"""
Triple cylinder obstacle — two cylinders side by side, then a third
centered between them further downstream. Creates a staggered array
that produces interesting vortex interactions.

Layout (flow goes left → right):

    ┌─────────────────────────────────┐
    │    ●                            │
    │         ●                       │
    │    ●                            │
    └─────────────────────────────────┘

Tunable constants:
    CX1_FRAC     — x position of front two cylinders as fraction of nx (default 1/5)
    CX2_FRAC     — x position of rear cylinder as fraction of nx       (default 1/3)
    CY_GAP_FRAC  — half-spacing between front cylinders as fraction of ny (default 1/4)
    RADIUS_FRAC  — radius of each cylinder as fraction of ny           (default 1/12)
"""

import numpy as np

CX1_FRAC    = 1 / 5
CX2_FRAC    = 1 / 3
CY_GAP_FRAC = 1 / 4
RADIUS_FRAC = 1 / 12


def _fill_circle(obstacle, cx, cy, radius, nx, ny):
    for ix in range(max(int(cx - radius) - 1, 0), min(int(cx + radius) + 2, nx)):
        for iy in range(max(int(cy - radius) - 1, 0), min(int(cy + radius) + 2, ny)):
            if (ix - cx)**2 + (iy - cy)**2 < radius**2:
                obstacle[ix, iy] = True


def generate(nx, ny):
    cx1    = int(nx * CX1_FRAC)
    cx2    = int(nx * CX2_FRAC)
    cy_mid = ny // 2
    gap    = int(ny * CY_GAP_FRAC)
    radius = ny * RADIUS_FRAC

    obstacle = np.zeros((nx, ny), dtype=bool)

    _fill_circle(obstacle, cx1, cy_mid - gap, radius, nx, ny)  # top front
    _fill_circle(obstacle, cx1, cy_mid + gap, radius, nx, ny)  # bottom front
    _fill_circle(obstacle, cx2, cy_mid,       radius, nx, ny)  # centre rear

    return obstacle
