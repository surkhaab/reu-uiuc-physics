"""
Double rectangle obstacle — two vertically stacked rectangular bars,
creating a channel/gap flow effect.

Layout (flow goes left → right):

    ┌─────────────────────────────┐
    │  ████████                   │
    │  ████████   ← gap           │
    │                             │
    │  ████████   ← gap           │
    │  ████████                   │
    └─────────────────────────────┘

Tunable constants:
    CX_FRAC      — horizontal center as fraction of nx  (default 1/4)
    CY_GAP_FRAC  — half-gap between bars as fraction of ny (default 1/8)
    WIDTH_FRAC   — bar width  as fraction of nx         (default 1/12)
    HEIGHT_FRAC  — bar height as fraction of ny         (default 1/4)
"""

import numpy as np

CX_FRAC     = 1 / 4
CY_GAP_FRAC = 1 / 8
WIDTH_FRAC  = 1 / 12
HEIGHT_FRAC = 1 / 8


def generate(nx, ny):
    cx = int(nx * CX_FRAC)
    cy = ny // 2

    half_gap = int(ny * CY_GAP_FRAC)

    half_w = int(nx * WIDTH_FRAC / 2)
    half_h = int(ny * HEIGHT_FRAC / 2)

    obstacle = np.zeros((nx, ny), dtype=bool)

    for bar_cy in [cy - half_gap - half_h,
                   cy + half_gap + half_h]:

        x0 = max(cx - half_w, 0)
        x1 = min(cx + half_w, nx)

        y0 = max(bar_cy - half_h, 0)
        y1 = min(bar_cy + half_h, ny)

        obstacle[x0:x1, y0:y1] = True

    return obstacle