"""
Square obstacle — axis-aligned square centered in the left third of the domain.

Tunable constants:
    CX_FRAC   — horizontal center as fraction of nx  (default 1/4)
    CY_FRAC   — vertical center as fraction of ny    (default 1/2)
    SIDE_FRAC — side length as fraction of ny        (default 2/9)
"""

import numpy as np

CX_FRAC   = 1 / 4
CY_FRAC   = 1 / 2
SIDE_FRAC = 2 / 9


def generate(nx, ny):
    cx   = int(nx * CX_FRAC)
    cy   = int(ny * CY_FRAC)
    half = int(ny * SIDE_FRAC / 2)

    obstacle = np.zeros((nx, ny), dtype=bool)
    x0, x1 = max(cx - half, 0), min(cx + half, nx)
    y0, y1 = max(cy - half, 0), min(cy + half, ny)
    obstacle[x0:x1, y0:y1] = True
    return obstacle
