"""
Cylinder obstacle — circular cross-section centered in the left third of the domain.

Parameters (via generate signature):
    nx, ny : grid dimensions

Tunable constants at the top of this file:
    CX_FRAC  — horizontal center as fraction of nx  (default 1/4)
    CY_FRAC  — vertical center as fraction of ny    (default 1/2)
    R_FRAC   — radius as fraction of ny             (default 1/9)
"""

import numpy as np

CX_FRAC = 1 / 4
CY_FRAC = 1 / 2
R_FRAC  = 1 / 9


def generate(nx, ny):
    cx     = int(nx * CX_FRAC)
    cy     = int(ny * CY_FRAC)
    radius = ny * R_FRAC

    obstacle = np.zeros((nx, ny), dtype=bool)
    for ix in range(int(cx - radius - 1), int(cx + radius + 2)):
        for iy in range(int(cy - radius - 1), int(cy + radius + 2)):
            if 0 <= ix < nx and 0 <= iy < ny:
                if np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2) < radius:
                    obstacle[ix, iy] = True
    return obstacle
