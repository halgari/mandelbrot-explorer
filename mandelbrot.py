"""GPU-accelerated Mandelbrot Explorer with perturbation theory for infinite zoom."""

import math
import struct
import sys
import threading

import moderngl
import mpmath
import pygame


def auto_iterations(scale, base=128, factor=50):
    """Scale iterations logarithmically with zoom depth."""
    zoom = 1.5 / scale
    if zoom <= 1.0:
        return base
    return int(base + factor * math.log2(zoom))


# ── Shaders ──────────────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 430
in vec2 in_position;
out vec2 frag_coord;
void main() {
    frag_coord = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# Perturbation shader: reference orbit in SSBO, double-precision deltas
PERTURB_FRAGMENT_SHADER = """
#version 430

in vec2 frag_coord;
out vec4 fragColor;

layout(std430, binding = 0) buffer ReferenceOrbit {
    double orbit[];  // interleaved re, im pairs
};

uniform int orbit_len;
uniform int max_iter;
uniform double scale;
uniform dvec2 delta_center;  // offset from reference center to view center
uniform vec2 resolution;

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    double aspect = double(resolution.x) / double(resolution.y);

    // delta_c = pixel offset from reference center in fractal coords
    double dcr = double(frag_coord.x) * aspect * scale + delta_center.x;
    double dci = double(frag_coord.y) * scale + delta_center.y;

    // Perturbation iteration: delta_{n+1} = 2*Z_n*delta_n + delta_n^2 + delta_c
    double dr = 0.0, di = 0.0;
    int iter = 0;
    int len = min(orbit_len, max_iter);

    for (int i = 0; i < len; i++) {
        double Zr = orbit[i * 2];
        double Zi = orbit[i * 2 + 1];

        double new_dr = 2.0 * (Zr * dr - Zi * di) + dr * dr - di * di + dcr;
        double new_di = 2.0 * (Zr * di + Zi * dr) + 2.0 * dr * di + dci;
        dr = new_dr;
        di = new_di;

        // Escape check: |Z_n + delta_n|^2 > 4
        double fr = Zr + dr;
        double fi = Zi + di;
        double mag2 = fr * fr + fi * fi;
        if (mag2 > 4.0) break;
        iter = i + 1;
    }

    if (iter >= len) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        // Smooth coloring using final |z|^2
        int idx = min(iter, len - 1);
        double Zr = orbit[idx * 2];
        double Zi = orbit[idx * 2 + 1];
        double fr = Zr + dr;
        double fi = Zi + di;
        float mag2 = float(fr * fr + fi * fi);
        float smooth_iter = float(iter) + 1.0 - log(log(mag2) / 2.0) / log(2.0);
        float t = smooth_iter / float(max_iter);
        vec3 color = palette(t * 8.0);
        fragColor = vec4(color, 1.0);
    }
}
"""

# Fallback: direct computation, float only (for GPUs without GL 4.3 / fp64)
VERTEX_SHADER_330 = """
#version 330
in vec2 in_position;
out vec2 frag_coord;
void main() {
    frag_coord = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DIRECT_FRAGMENT_SHADER = """
#version 330
precision highp float;

in vec2 frag_coord;
out vec4 fragColor;

uniform vec2 center_f;
uniform float scale_f;
uniform int max_iter;
uniform vec2 resolution;

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    float aspect = resolution.x / resolution.y;
    float x = (frag_coord.x * aspect) * scale_f + center_f.x;
    float y = frag_coord.y * scale_f + center_f.y;

    float zr = 0.0, zi = 0.0;
    float zr2 = 0.0, zi2 = 0.0;
    int iter = 0;

    for (int i = 0; i < max_iter; i++) {
        zi = 2.0 * zr * zi + y;
        zr = zr2 - zi2 + x;
        zr2 = zr * zr;
        zi2 = zi * zi;
        if (zr2 + zi2 > 4.0) break;
        iter = i + 1;
    }

    if (iter == max_iter) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float smooth_iter = float(iter) + 1.0 - log(log(zr2 + zi2) / 2.0) / log(2.0);
        float t = smooth_iter / float(max_iter);
        vec3 color = palette(t * 8.0);
        fragColor = vec4(color, 1.0);
    }
}
"""

# ── HUD shaders ──────────────────────────────────────────────────────────────

HUD_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 uv;
uniform vec4 rect;
void main() {
    vec2 pos = rect.xy + in_position * rect.zw;
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = in_texcoord;
}
"""

HUD_FRAGMENT_SHADER = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D tex;
void main() {
    fragColor = texture(tex, uv);
}
"""


# ── Reference orbit computation ─────────────────────────────────────────────

def compute_reference_orbit(cx, cy, max_iter, zoom):
    """Compute reference orbit at arbitrary precision. Returns packed doubles."""
    # Scale precision with zoom depth
    precision = max(20, int(math.log10(max(zoom, 1))) + 15)
    mpmath.mp.dps = precision

    zr, zi = mpmath.mpf(0), mpmath.mpf(0)
    orbit_re = []
    orbit_im = []

    for _ in range(max_iter):
        orbit_re.append(float(zr))
        orbit_im.append(float(zi))
        zr, zi = zr * zr - zi * zi + cx, 2 * zr * zi + cy
        if float(zr * zr + zi * zi) > 1e30:
            break

    # Pack as interleaved doubles: re0, im0, re1, im1, ...
    data = struct.pack(f"{len(orbit_re) * 2}d",
                       *[v for pair in zip(orbit_re, orbit_im) for v in pair])
    return data, len(orbit_re)


# ── HUD overlay ──────────────────────────────────────────────────────────────

class HudOverlay:
    """Renders pygame surfaces as GL texture overlays."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=HUD_VERTEX_SHADER,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        vertices = struct.pack(
            "16f",
            0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
        )
        vbo = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(
            self.prog, [(vbo, "2f 2f", "in_position", "in_texcoord")]
        )
        self.texture = None

    def render(self, surface, screen_w, screen_h, x, y):
        tex_w, tex_h = surface.get_size()
        raw = pygame.image.tobytes(surface, "RGBA", False)

        if self.texture is None or self.texture.size != (tex_w, tex_h):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((tex_w, tex_h), 4, raw)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.texture.write(raw)

        ndc_x = (x / screen_w) * 2 - 1
        ndc_y = 1 - ((y + tex_h) / screen_h) * 2
        ndc_w = (tex_w / screen_w) * 2
        ndc_h = (tex_h / screen_h) * 2

        self.prog["rect"].value = (ndc_x, ndc_y, ndc_w, ndc_h)
        self.texture.use(0)
        self.prog["tex"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)


def render_hud_surface(font, fps, max_iter, iter_offset, scale, center_x, center_y,
                       use_perturbation, orbit_len, computing, show_help):
    zoom = 1.5 / scale
    # Format center with appropriate precision
    digits = max(6, int(math.log10(max(zoom, 1))) + 4)
    cx_str = mpmath.nstr(center_x, digits)
    cy_str = mpmath.nstr(center_y, digits)

    offset_str = f" ({iter_offset:+d})" if iter_offset != 0 else " (auto)"
    mode = "perturbation (fp64)" if use_perturbation else "direct (fp32)"
    status = " [computing orbit...]" if computing else ""

    lines = [
        f"FPS: {fps:.0f}  |  Iterations: {max_iter}{offset_str}  |  Zoom: {zoom:.2e}x",
        f"Center: ({cx_str}, {cy_str})",
        f"Mode: {mode}  |  Orbit: {orbit_len} pts{status}",
    ]
    if show_help:
        lines.append("")
        lines.append("--- Controls ---")
        lines.append("Scroll         Zoom in/out (toward cursor)")
        lines.append("Left drag      Pan")
        lines.append("+/-            Adjust iterations (auto-scales with zoom)")
        lines.append("G              Go to coordinates (x, y[, zoom])")
        lines.append("R              Reset view")
        lines.append("H              Toggle this help")
        lines.append("Q / Esc        Quit")

    padding = 8
    line_height = font.get_height() + 2
    text_w = max(font.size(line)[0] for line in lines) + padding * 2
    text_h = line_height * len(lines) + padding * 2

    surf = pygame.Surface((text_w, text_h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 160))

    for i, line in enumerate(lines):
        text_surf = font.render(line, True, (255, 255, 255))
        surf.blit(text_surf, (padding, padding + i * line_height))

    return surf


def render_input_surface(font, text, screen_w):
    padding = 10
    lines = [
        "Go to:  x, y[, zoom]",
        f"> {text}_",
        "Enter to jump, Esc to cancel",
    ]
    line_height = font.get_height() + 2
    box_w = min(500, screen_w - 20)
    box_h = line_height * len(lines) + padding * 2

    surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 200))
    pygame.draw.rect(surf, (100, 180, 255), (0, 0, box_w, box_h), 2)

    for i, line in enumerate(lines):
        color = (100, 180, 255) if i == 0 else (255, 255, 255)
        text_surf = font.render(line, True, color)
        surf.blit(text_surf, (padding, padding + i * line_height))

    return surf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    WIDTH, HEIGHT = 1280, 720

    pygame.init()
    pygame.display.set_caption("Mandelbrot Explorer")
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

    ctx = moderngl.create_context()

    # Try perturbation shader (requires GL 4.3 + fp64)
    use_perturbation = True
    try:
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=PERTURB_FRAGMENT_SHADER)
    except Exception:
        use_perturbation = False
        prog = ctx.program(vertex_shader=VERTEX_SHADER_330, fragment_shader=DIRECT_FRAGMENT_SHADER)

    # Full-screen quad
    vertices = struct.pack(
        "8f",
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    )
    vbo = ctx.buffer(vertices)
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_position")])

    hud = HudOverlay(ctx)
    font = pygame.font.SysFont("monospace", 15)

    # State — center stored at arbitrary precision
    center_x = mpmath.mpf("-0.5")
    center_y = mpmath.mpf("0")
    scale = 1.5
    iter_offset = 0

    # Reference orbit state
    ref_cx = mpmath.mpf("0")
    ref_cy = mpmath.mpf("0")
    ref_max_iter = 0
    orbit_data = None
    orbit_len = 0
    orbit_buffer = ctx.buffer(reserve=16) if use_perturbation else None  # placeholder
    orbit_dirty = True
    computing = False

    # Orbit computation in background thread
    orbit_result = [None]  # [data, length] or None
    orbit_lock = threading.Lock()

    def recompute_orbit_async(cx, cy, max_iter, zoom):
        nonlocal computing
        computing = True
        def worker():
            nonlocal computing
            data, length = compute_reference_orbit(cx, cy, max_iter, zoom)
            with orbit_lock:
                orbit_result[0] = (data, length, cx, cy, max_iter)
            computing = False
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    dragging = False
    drag_start = (0, 0)
    drag_center_start_x = mpmath.mpf("0")
    drag_center_start_y = mpmath.mpf("0")
    show_help = True
    input_mode = False
    input_text = ""

    clock = pygame.time.Clock()
    width, height = WIDTH, HEIGHT

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and input_mode:
                if event.key == pygame.K_ESCAPE:
                    input_mode = False
                    input_text = ""
                elif event.key == pygame.K_RETURN:
                    input_mode = False
                    try:
                        parts = [p.strip() for p in input_text.split(",")]
                        center_x = mpmath.mpf(parts[0])
                        center_y = mpmath.mpf(parts[1])
                        if len(parts) >= 3:
                            zoom = float(parts[2])
                            scale = 1.5 / zoom
                        orbit_dirty = True
                    except (ValueError, IndexError):
                        pass
                    input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode and event.unicode in "0123456789.,-+eE ":
                        input_text += event.unicode

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    center_x = mpmath.mpf("-0.5")
                    center_y = mpmath.mpf("0")
                    scale = 1.5
                    iter_offset = 0
                    orbit_dirty = True
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_g:
                    input_mode = True
                    input_text = ""
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    iter_offset += 64
                    orbit_dirty = True
                elif event.key == pygame.K_MINUS:
                    iter_offset -= 64
                    orbit_dirty = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start = event.pos
                    drag_center_start_x = center_x
                    drag_center_start_y = center_y
                elif event.button in (4, 5):
                    mx, my = event.pos
                    aspect = width / height
                    pixel_x = (mx / width) * 2 - 1
                    pixel_y = (1 - my / height) * 2 - 1
                    # Compute fractal coords at cursor (full precision)
                    fx = center_x + mpmath.mpf(pixel_x * aspect * scale)
                    fy = center_y + mpmath.mpf(pixel_y * scale)
                    if event.button == 4:
                        scale *= 0.85
                    else:
                        scale *= 1.15
                    center_x = fx - mpmath.mpf(pixel_x * aspect * scale)
                    center_y = fy - mpmath.mpf(pixel_y * scale)
                    orbit_dirty = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    orbit_dirty = True

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    aspect = width / height
                    center_x = drag_center_start_x - mpmath.mpf((dx / width) * 2 * aspect * scale)
                    center_y = drag_center_start_y + mpmath.mpf((dy / height) * 2 * scale)

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                aspect = width / height
                pixel_x = (mx / width) * 2 - 1
                pixel_y = (1 - my / height) * 2 - 1
                fx = center_x + mpmath.mpf(pixel_x * aspect * scale)
                fy = center_y + mpmath.mpf(pixel_y * scale)
                if event.y > 0:
                    scale *= 0.85
                elif event.y < 0:
                    scale *= 1.15
                center_x = fx - mpmath.mpf(pixel_x * aspect * scale)
                center_y = fy - mpmath.mpf(pixel_y * scale)
                orbit_dirty = True

            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                ctx.viewport = (0, 0, width, height)

        # Auto-scale iterations
        max_iter = max(64, auto_iterations(scale) + iter_offset)

        # Check for completed orbit computation
        if use_perturbation:
            with orbit_lock:
                if orbit_result[0] is not None:
                    data, length, cx, cy, mi = orbit_result[0]
                    orbit_result[0] = None
                    orbit_data = data
                    orbit_len = length
                    ref_cx = cx
                    ref_cy = cy
                    ref_max_iter = mi
                    # Recreate buffer with new data
                    orbit_buffer.release()
                    orbit_buffer = ctx.buffer(orbit_data)

            # Trigger recomputation if dirty and not already computing
            if orbit_dirty and not computing and not dragging:
                orbit_dirty = False
                zoom = 1.5 / scale
                recompute_orbit_async(center_x, center_y, max_iter, zoom)

        # Set uniforms and render
        if use_perturbation and orbit_len > 0:
            # Compute delta from reference center at full precision
            dcx = float(center_x - ref_cx)
            dcy = float(center_y - ref_cy)
            prog["delta_center"].value = (dcx, dcy)
            prog["scale"].value = float(scale)
            prog["orbit_len"].value = orbit_len
            prog["max_iter"].value = max_iter
            prog["resolution"].value = (float(width), float(height))
            orbit_buffer.bind_to_storage_buffer(0)
        else:
            if not use_perturbation:
                prog["center_f"].value = (float(center_x), float(center_y))
                prog["scale_f"].value = float(scale)
                prog["max_iter"].value = max_iter
                prog["resolution"].value = (float(width), float(height))

        # Render fractal
        ctx.clear(0.0, 0.0, 0.0)
        ctx.disable(moderngl.BLEND)
        if use_perturbation and orbit_len > 0:
            vao.render(moderngl.TRIANGLE_STRIP)
        elif not use_perturbation:
            vao.render(moderngl.TRIANGLE_STRIP)
        # else: orbit not ready yet, show black

        # Render HUD
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        fps = clock.get_fps()
        hud_surf = render_hud_surface(
            font, fps, max_iter, iter_offset, scale, center_x, center_y,
            use_perturbation, orbit_len, computing, show_help
        )
        hud.render(hud_surf, width, height, 10, 10)

        if input_mode:
            input_surf = render_input_surface(font, input_text, width)
            hud.render(input_surf, width, height, 10, height // 2 - 30)

        pygame.display.flip()
        clock.tick(0)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
