"""
NACA 0012 airfoil obstacle — symmetric airfoil at a fixed angle of attack.

The airfoil chord runs along the x-axis. Each x position maps to a
half-thickness y offset using the NACA 00xx formula.

Tunable constants:
    CX_FRAC     — leading-edge x position as fraction of nx  (default 1/5)
    CY_FRAC     — vertical center as fraction of ny          (default 1/2)
    CHORD_FRAC  — chord length as fraction of nx             (default 1/4)
    THICKNESS   — max thickness as fraction of chord         (default 0.12)
    ANGLE_DEG   — angle of attack in degrees                 (default 8)
"""

import numpy as np

CX_FRAC    = 1 / 5
CY_FRAC    = 1 / 2
CHORD_FRAC = 1 / 4
THICKNESS  = 0.12
ANGLE_DEG  = 8


def _naca_half_thickness(xc, t):
    """NACA 00xx half-thickness at normalised chord position xc in [0,1]."""
    xc = np.clip(xc, 0, 1)
    return 5 * t * (0.2969 * np.sqrt(xc)
                    - 0.1260 * xc
                    - 0.3516 * xc**2
                    + 0.2843 * xc**3
                    - 0.1015 * xc**4)


def generate(nx, ny):
    chord  = int(nx * CHORD_FRAC)
    cx     = int(nx * CX_FRAC)          # leading edge x
    cy     = int(ny * CY_FRAC)
    angle  = np.radians(ANGLE_DEG)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    obstacle = np.zeros((nx, ny), dtype=bool)

    # Build a dense set of (x, y) airfoil surface points, then fill interior
    n_pts = chord * 4
    xc    = np.linspace(0, 1, n_pts)
    yt    = _naca_half_thickness(xc, THICKNESS) * chord

    # Upper and lower surface in local (unrotated) coords
    local_x = xc * chord
    upper   = list(zip(local_x,  yt))
    lower   = list(zip(local_x, -yt))
    contour = upper + lower[::-1]

    # Rotate and translate into grid coords
    def to_grid(lx, ly):
        gx = cx + lx * cos_a - ly * sin_a
        gy = cy + lx * sin_a + ly * cos_a
        return int(round(gx)), int(round(gy))

    grid_pts = [to_grid(lx, ly) for lx, ly in contour]

    # Fill using scanline: for each ix, mark between min and max iy
    from collections import defaultdict
    col_ys = defaultdict(list)
    for gx, gy in grid_pts:
        if 0 <= gx < nx and 0 <= gy < ny:
            col_ys[gx].append(gy)

    for gx, ys in col_ys.items():
        obstacle[gx, min(ys):max(ys) + 1] = True

    return obstacle
