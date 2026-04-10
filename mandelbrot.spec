# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Mandelbrot Explorer
# Run on Windows: uv run pyinstaller mandelbrot.spec

a = Analysis(
    ['mandelbrot.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['glcontext'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MandelbrotExplorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    icon=None,
)
