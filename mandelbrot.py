"""GPU-accelerated Mandelbrot Explorer using OpenGL fragment shaders."""

import math
import struct
import sys

import moderngl
import pygame


def auto_iterations(scale, base=128, factor=50):
    """Scale iterations logarithmically with zoom depth."""
    zoom = 1.5 / scale  # zoom = 1.0 at default view
    if zoom <= 1.0:
        return base
    return int(base + factor * math.log2(zoom))

VERTEX_SHADER_330 = """
#version 330
in vec2 in_position;
out vec2 frag_coord;
void main() {
    frag_coord = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

VERTEX_SHADER_400 = """
#version 400
in vec2 in_position;
out vec2 frag_coord;
void main() {
    frag_coord = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 400
precision highp float;

in vec2 frag_coord;
out vec4 fragColor;

uniform dvec2 center;
uniform double scale;
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
    double aspect = double(resolution.x) / double(resolution.y);
    double x = (double(frag_coord.x) * aspect) * scale + center.x;
    double y = double(frag_coord.y) * scale + center.y;

    double zr = 0.0, zi = 0.0;
    double zr2 = 0.0, zi2 = 0.0;
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
        float smooth_iter = float(iter) + 1.0 - log(log(float(zr2 + zi2)) / 2.0) / log(2.0);
        float t = smooth_iter / float(max_iter);
        vec3 color = palette(t * 8.0);
        fragColor = vec4(color, 1.0);
    }
}
"""

FRAGMENT_SHADER_FLOAT = """
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

# HUD overlay shaders - renders a texture quad with alpha blending
HUD_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 uv;
uniform vec4 rect;  // x, y, w, h in NDC
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


class HudOverlay:
    """Renders pygame surfaces as GL texture overlays."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=HUD_VERTEX_SHADER,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        # Unit quad: flip V texcoord so pygame top-down matches GL bottom-up
        vertices = struct.pack(
            "16f",
            0.0, 0.0, 0.0, 1.0,  # bottom-left pos, top texcoord
            1.0, 0.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 0.0,  # top-left pos, bottom texcoord
            1.0, 1.0, 1.0, 0.0,
        )
        vbo = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(
            self.prog, [(vbo, "2f 2f", "in_position", "in_texcoord")]
        )
        self.texture = None

    def render(self, surface, screen_w, screen_h, x, y):
        """Render a pygame surface at pixel position (x, y) from top-left."""
        tex_w, tex_h = surface.get_size()
        # Convert pygame surface to raw RGBA bytes
        raw = pygame.image.tobytes(surface, "RGBA", False)

        if self.texture is None or self.texture.size != (tex_w, tex_h):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((tex_w, tex_h), 4, raw)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.texture.write(raw)

        # Convert pixel coords to NDC (-1 to 1)
        ndc_x = (x / screen_w) * 2 - 1
        ndc_y = 1 - ((y + tex_h) / screen_h) * 2  # flip Y
        ndc_w = (tex_w / screen_w) * 2
        ndc_h = (tex_h / screen_h) * 2

        self.prog["rect"].value = (ndc_x, ndc_y, ndc_w, ndc_h)
        self.texture.use(0)
        self.prog["tex"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)


def render_hud_surface(font, fps, max_iter, iter_offset, scale, center_x, center_y, use_double, show_help):
    """Render the HUD text to a pygame surface with transparency."""
    offset_str = f" ({iter_offset:+d})" if iter_offset != 0 else " (auto)"
    lines = [
        f"FPS: {fps:.0f}  |  Iterations: {max_iter}{offset_str}  |  Zoom: {1.5/scale:.2e}x",
        f"Center: ({center_x:.12f}, {center_y:.12f})",
        f"Precision: {'double (fp64)' if use_double else 'float (fp32)'}",
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
    """Render the coordinate input box."""
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


def main():
    WIDTH, HEIGHT = 1280, 720

    pygame.init()
    pygame.display.set_caption("Mandelbrot Explorer")
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

    ctx = moderngl.create_context()

    # Try double-precision shader first, fall back to float
    use_double = True
    try:
        prog = ctx.program(vertex_shader=VERTEX_SHADER_400, fragment_shader=FRAGMENT_SHADER)
    except Exception:
        use_double = False
        prog = ctx.program(vertex_shader=VERTEX_SHADER_330, fragment_shader=FRAGMENT_SHADER_FLOAT)

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

    # Mandelbrot parameters
    center_x, center_y = -0.5, 0.0
    scale = 1.5
    iter_offset = 0  # manual +/- adjustment on top of auto

    dragging = False
    drag_start = (0, 0)
    drag_center_start = (0.0, 0.0)
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
                    # Parse: "x, y" or "x, y, zoom"
                    input_mode = False
                    try:
                        parts = [p.strip() for p in input_text.split(",")]
                        center_x = float(parts[0])
                        center_y = float(parts[1])
                        if len(parts) >= 3:
                            zoom = float(parts[2])
                            scale = 1.5 / zoom
                    except (ValueError, IndexError):
                        pass  # invalid input, just dismiss
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
                    center_x, center_y = -0.5, 0.0
                    scale = 1.5
                    iter_offset = 0
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_g:
                    input_mode = True
                    input_text = ""
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    iter_offset += 64
                elif event.key == pygame.K_MINUS:
                    iter_offset -= 64

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start = event.pos
                    drag_center_start = (center_x, center_y)
                elif event.button == 4:  # Scroll up - zoom in
                    mx, my = event.pos
                    aspect = width / height
                    fx = ((mx / width) * 2 - 1) * aspect * scale + center_x
                    fy = ((1 - my / height) * 2 - 1) * scale + center_y
                    scale *= 0.85
                    center_x = fx - ((mx / width) * 2 - 1) * aspect * scale
                    center_y = fy - ((1 - my / height) * 2 - 1) * scale
                elif event.button == 5:  # Scroll down - zoom out
                    mx, my = event.pos
                    aspect = width / height
                    fx = ((mx / width) * 2 - 1) * aspect * scale + center_x
                    fy = ((1 - my / height) * 2 - 1) * scale + center_y
                    scale *= 1.15
                    center_x = fx - ((mx / width) * 2 - 1) * aspect * scale
                    center_y = fy - ((1 - my / height) * 2 - 1) * scale

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    aspect = width / height
                    center_x = drag_center_start[0] - (dx / width) * 2 * aspect * scale
                    center_y = drag_center_start[1] + (dy / height) * 2 * scale

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                aspect = width / height
                fx = ((mx / width) * 2 - 1) * aspect * scale + center_x
                fy = ((1 - my / height) * 2 - 1) * scale + center_y
                if event.y > 0:
                    scale *= 0.85
                elif event.y < 0:
                    scale *= 1.15
                center_x = fx - ((mx / width) * 2 - 1) * aspect * scale
                center_y = fy - ((1 - my / height) * 2 - 1) * scale

            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                ctx.viewport = (0, 0, width, height)

        # Auto-scale iterations with zoom depth
        max_iter = max(64, auto_iterations(scale) + iter_offset)

        # Set uniforms
        if use_double:
            prog["center"].value = (center_x, center_y)
            prog["scale"].value = scale
        else:
            prog["center_f"].value = (float(center_x), float(center_y))
            prog["scale_f"].value = float(scale)
        prog["max_iter"].value = max_iter
        prog["resolution"].value = (float(width), float(height))

        # Render fractal
        ctx.clear(0.0, 0.0, 0.0)
        ctx.disable(moderngl.BLEND)
        vao.render(moderngl.TRIANGLE_STRIP)

        # Render HUD overlay
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        fps = clock.get_fps()
        hud_surf = render_hud_surface(
            font, fps, max_iter, iter_offset, scale, center_x, center_y, use_double, show_help
        )
        hud.render(hud_surf, width, height, 10, 10)

        # Render input overlay if active
        if input_mode:
            input_surf = render_input_surface(font, input_text, width)
            hud.render(input_surf, width, height, 10, height // 2 - 30)

        pygame.display.flip()
        clock.tick(0)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
