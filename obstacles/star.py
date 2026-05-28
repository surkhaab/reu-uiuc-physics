"""
Star obstacle — a filled n-pointed star centered in the left third of the domain.

The star is built by alternating outer (spike) and inner (valley) vertices
around a circle, then filling the interior using a ray-casting test.

Tunable constants:
    CX_FRAC      — horizontal center as fraction of nx  (default 1/4)
    CY_FRAC      — vertical center as fraction of ny    (default 1/2)
    N_POINTS     — number of star points                (default 5)
    R_OUTER_FRAC — outer spike radius as fraction of ny (default 1/6)
    R_INNER_FRAC — inner valley radius as fraction of ny (default 1/14)
"""

import numpy as np

CX_FRAC      = 1 / 4
CY_FRAC      = 1 / 2
N_POINTS     = 5
R_OUTER_FRAC = 1 / 6
R_INNER_FRAC = 1 / 14


def _point_in_polygon(px, py, poly_x, poly_y):
    """Ray-casting test: is point (px, py) inside the polygon?"""
    n      = len(poly_x)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def generate(nx, ny):
    cx      = int(nx * CX_FRAC)
    cy      = int(ny * CY_FRAC)
    r_outer = ny * R_OUTER_FRAC
    r_inner = ny * R_INNER_FRAC

    # Build star polygon vertices by alternating outer and inner radii
    n_verts = N_POINTS * 2
    angles  = np.linspace(0, 2 * np.pi, n_verts, endpoint=False) - np.pi / 2
    radii   = np.where(np.arange(n_verts) % 2 == 0, r_outer, r_inner)
    poly_x  = cx + radii * np.cos(angles)
    poly_y  = cy + radii * np.sin(angles)

    # Fill every grid cell whose centre is inside the polygon
    obstacle = np.zeros((nx, ny), dtype=bool)
    ix_min = max(int(cx - r_outer) - 1, 0)
    ix_max = min(int(cx + r_outer) + 2, nx)
    iy_min = max(int(cy - r_outer) - 1, 0)
    iy_max = min(int(cy + r_outer) + 2, ny)

    for ix in range(ix_min, ix_max):
        for iy in range(iy_min, iy_max):
            if _point_in_polygon(ix, iy, poly_x, poly_y):
                obstacle[ix, iy] = True

    return obstacle
