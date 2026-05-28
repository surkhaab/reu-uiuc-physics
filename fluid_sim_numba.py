#!/usr/bin/env python3
"""
Lattice Boltzmann Method (LBM) Flow Simulation
-----------------------------------------------
Usage:
    python fluid_dynamics.py [--obstacle cylinder] [--steps 2000] [--runs 20] [--output flow.gif]

Arguments:
    --obstacle   Name of obstacle module inside the obstacles/ folder (default: cylinder)
                 Built-in options: cylinder, square, airfoil, square, double_rectangle, star
                 Add your own: drop a new .py file in obstacles/ with a generate(nx, ny) function
    --steps      Number of LBM steps per run segment (default: 2000)
    --runs       Number of run segments (default: 20)
    --output     Output filename for animation, e.g. flow.mp4 or flow.gif (default: flow.gif)
    --vmax       Color scale max velocity (default: 0.07)
    --nx         Grid width  (default: 520)
    --ny         Grid height (default: 180)
    --no-numba   Disable Numba JIT even if numba is installed
"""

import argparse
import importlib
import itertools
import os
import sys
import time
import subprocess

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# ──────────────────────────────────────────────
# Numba — optional, graceful fallback to NumPy
# ──────────────────────────────────────────────
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Transparent no-op decorator so the rest of the file is unchanged
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator

# ──────────────────────────────────────────────
# Grid dimensions (set in main after arg parse)
# ──────────────────────────────────────────────
nx: int
ny: int

# D2Q9 velocity vectors (int64 for numba compatibility)
v = np.array([
    [ 0,  0],
    [ 0,  1],
    [ 0, -1],
    [ 1,  0],
    [-1,  0],
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1],
], dtype=np.int64)

# D2Q9 weights
WEIGHTS = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=np.float64)

# BGK relaxation parameter
OMEGA = 1.9572953736654806


# ══════════════════════════════════════════════
# NUMBA-JIT CORE  — compiled on first call
# All heavy maths lives in one @njit function so
# the Python loop overhead is just 40k trivial calls.
# ══════════════════════════════════════════════

@njit(cache=True)
def _lbm_step(n, n_init, obstacle, v, weights, omega):
    """
    One full LBM timestep:
      FixBoundary → Collision (BGK) → Bounce-back → Stream (roll)

    Compiled to native code by Numba. When Numba is absent the same
    function runs as plain Python/NumPy (slow but correct).

    Returns the updated distribution n_out (same shape as n).
    """
    nk, nx, ny = n.shape

    # ── 1. Fix boundary ──────────────────────────────────────
    # Left inlet: restore equilibrium
    for k in range(nk):
        for j in range(ny):
            n[k, 0, j] = n_init[k, 0, j]
    # Right outlet: zero-gradient (copy from neighbour)
    for k in range(4, 7):
        for j in range(ny):
            n[k, nx-1, j] = n[k, nx-2, j]

    # ── 2. Collision (BGK) ───────────────────────────────────
    # Compute rho and u
    rho = np.zeros((nx, ny))
    u   = np.zeros((2, nx, ny))
    for k in range(nk):
        for i in range(nx):
            for j in range(ny):
                rho[i, j]    += n[k, i, j]
                u[0, i, j]   += v[k, 0] * n[k, i, j]
                u[1, i, j]   += v[k, 1] * n[k, i, j]
    for i in range(nx):
        for j in range(ny):
            u[0, i, j] /= rho[i, j]
            u[1, i, j] /= rho[i, j]

    # Compute equilibrium and relax
    n_out = np.empty_like(n)
    for k in range(nk):
        for i in range(nx):
            for j in range(ny):
                vdotu = v[k, 0]*u[0, i, j] + v[k, 1]*u[1, i, j]
                udotu = u[0, i, j]**2 + u[1, i, j]**2
                t     = 1.0 + 3.0*vdotu + 4.5*vdotu**2 - 1.5*udotu
                n_eq  = weights[k] * rho[i, j] * t
                if obstacle[i, j]:
                    n_out[k, i, j] = n[k, i, j]        # no collision inside solid
                else:
                    n_out[k, i, j] = n[k, i, j]*(1.0 - omega) + omega*n_eq

    # ── 3. Bounce-back ───────────────────────────────────────
    n_bounce = n_out.copy()
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                n_bounce[1, i, j] = n_out[2, i, j]
                n_bounce[2, i, j] = n_out[1, i, j]
                n_bounce[3, i, j] = n_out[4, i, j]
                n_bounce[4, i, j] = n_out[3, i, j]
                n_bounce[5, i, j] = n_out[8, i, j]
                n_bounce[8, i, j] = n_out[5, i, j]
                n_bounce[6, i, j] = n_out[7, i, j]
                n_bounce[7, i, j] = n_out[6, i, j]

    # ── 4. Stream (advect) ───────────────────────────────────
    # Equivalent to np.roll but done manually so Numba can compile it.
    # result[k, i, j] = n_bounce[k, (i - vx) % nx, (j - vy) % ny]
    n_new = np.empty_like(n_bounce)
    for k in range(nk):
        vx = v[k, 0]
        vy = v[k, 1]
        for i in range(nx):
            src_i = (i - vx) % nx
            for j in range(ny):
                src_j = (j - vy) % ny
                n_new[k, i, j] = n_bounce[k, src_i, src_j]

    return n_new, rho, u


# ──────────────────────────────────────────────
# NumPy fallback step (readable, no Numba needed)
# Used when --no-numba is passed.
# ──────────────────────────────────────────────
def _micro2macro_np(n):
    rho = np.sum(n, axis=0)
    u   = np.einsum("kd,kij->dij", v, n) / rho
    return rho, u

def _macro2eq_np(rho, u):
    vdotu = np.einsum("kd,dij->kij", v, u)
    udotu = np.einsum("dij,dij->ij",  u, u)
    t     = 1 + 3*vdotu + 4.5*vdotu**2 - 1.5*udotu
    return WEIGHTS[:, np.newaxis, np.newaxis] * rho[np.newaxis] * t

def _lbm_step_numpy(n, n_init, obstacle, v, weights, omega):
    # Boundary
    n[:, 0, :]    = n_init[:, 0, :]
    n[4:7, -1, :] = n[4:7, -2, :]
    # Collision
    rho, u = _micro2macro_np(n)
    n_eq   = _macro2eq_np(rho, u)
    n_out  = n * (1 - omega) + omega * n_eq
    n_out[:, obstacle] = n[:, obstacle]
    # Bounce
    n_b = n_out.copy()
    for a, b in [(1,2),(3,4),(5,8),(6,7)]:
        n_b[a, obstacle] = n_out[b, obstacle]
        n_b[b, obstacle] = n_out[a, obstacle]
    # Stream
    for k in range(9):
        n_b[k] = np.roll(n_b[k], v[k, 0], axis=0)
        n_b[k] = np.roll(n_b[k], v[k, 1], axis=1)
    return n_b, rho, u


# ──────────────────────────────────────────────
# Setup & Run
# ──────────────────────────────────────────────
def Macro2Equilibrium(rho, u):
    """Used only at initialisation (not in the hot loop)."""
    return _macro2eq_np(rho, u)

def Setup():
    rho  = np.ones((nx, ny))
    u    = np.zeros((2, nx, ny))
    y    = np.arange(ny)[np.newaxis, :]
    u[0] = 0.04 * (1.0 + 1e-4 * np.sin(y / ny * 2 * np.pi))
    return Macro2Equilibrium(rho, u)


def Run(steps, n, n_init, obstacle, record_every=100, use_numba=True):
    rhos, us = [], []
    step_fn  = _lbm_step if (use_numba and NUMBA_AVAILABLE) else _lbm_step_numpy

    for step in range(steps + 1):
        n, rho, u = step_fn(n, n_init, obstacle, v, WEIGHTS, OMEGA)
        if step % record_every == 0:
            u2 = np.sqrt(u[0]**2 + u[1]**2)
            rhos.append(rho)
            us.append(u2.T)
    return rhos, us, n


# ──────────────────────────────────────────────
# Dynamic obstacle loader
# ──────────────────────────────────────────────
def load_obstacle(name, nx, ny):
    obstacles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacles")
    available     = sorted(
        f[:-3] for f in os.listdir(obstacles_dir)
        if f.endswith(".py") and not f.startswith("_")
    )
    if name not in available:
        print(f"\n  ✗ Obstacle '{name}' not found.")
        print(f"    Available: {', '.join(available)}")
        print(f"    Create obstacles/{name}.py with a generate(nx, ny) function.\n")
        sys.exit(1)
    parent = os.path.dirname(os.path.abspath(__file__))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"obstacles.{name}")
    return module.generate(nx, ny)


# ──────────────────────────────────────────────
# Animation
# ──────────────────────────────────────────────
def build_animation(us_flat, vmax, fps=10):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cax = ax.imshow(us_flat[0], cmap="RdBu_r", vmin=0, vmax=vmax,
                    origin="lower", aspect="auto")
    fig.colorbar(cax, ax=ax, label="Speed |u|")
    ax.set_title("LBM Flow Simulation")
    ax.set_xlabel("y"); ax.set_ylabel("x")

    def animate(i):
        cax.set_array(us_flat[i])
        return (cax,)

    anim = FuncAnimation(fig, animate, frames=len(us_flat),
                         interval=1000 // fps, blit=True)
    return fig, anim


def save_animation(anim, output_path, fps=10):
    ext = output_path.rsplit(".", 1)[-1].lower()
    print(f"  Saving animation to '{output_path}' …", flush=True)
    if ext == "mp4":
        writer = FFMpegWriter(fps=fps)
    elif ext == "gif":
        writer = PillowWriter(fps=fps)
    else:
        print(f"  ⚠  Unknown extension '.{ext}', defaulting to GIF.")
        output_path = output_path.rsplit(".", 1)[0] + ".gif"
        writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=120)
    print(f"  ✓ Saved: {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    global nx, ny

    parser = argparse.ArgumentParser(
        description="Lattice Boltzmann flow simulation."
    )
    parser.add_argument("--obstacle", default="cylinder",
                        help="Obstacle name (must match a file in obstacles/)")
    parser.add_argument("--steps",    type=int,   default=2000)
    parser.add_argument("--runs",     type=int,   default=10)
    parser.add_argument("--output",   default="flow.gif")
    parser.add_argument("--vmax",     type=float, default=0.07)
    parser.add_argument("--nx",       type=int,   default=260)
    parser.add_argument("--ny",       type=int,   default=90)
    parser.add_argument("--no-numba", action="store_true",
                        help="Disable Numba JIT, use pure NumPy instead")
    args = parser.parse_args()

    nx, ny     = args.nx, args.ny
    use_numba  = (not args.no_numba) and NUMBA_AVAILABLE

    print("=" * 60)
    print("  Lattice Boltzmann Method — Flow Simulation")
    print("=" * 60)
    print(f"  Grid      : {nx} × {ny}")
    print(f"  Obstacle  : {args.obstacle}")
    print(f"  Steps/run : {args.steps}   |   Runs: {args.runs}")
    print(f"  Output    : {args.output}")
    if use_numba:
        print("  Backend   : Numba JIT  ✓  (first run includes compile time)")
    elif NUMBA_AVAILABLE:
        print("  Backend   : NumPy  (Numba disabled via --no-numba)")
    else:
        print("  Backend   : NumPy  (install numba for a big speedup)")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────
    t0 = time.perf_counter()
    print("\n[1/3] Setting up lattice …", flush=True)
    fin      = Setup()
    feq_init = fin.copy()
    obstacle = load_obstacle(args.obstacle, nx, ny)
    print(f"      Obstacle cells : {obstacle.sum():,}")

    if use_numba:
        print("      Warming up Numba JIT …", end=" ", flush=True)
        warm_start = time.perf_counter()
        # Compile on a tiny grid so the warm-up is fast
        _lbm_step(
            np.ones((9, 4, 4)),
            np.ones((9, 4, 4)),
            np.zeros((4, 4), dtype=bool),
            v, WEIGHTS, OMEGA,
        )
        print(f"done ({time.perf_counter() - warm_start:.1f}s)", flush=True)

    # ── Simulation ─────────────────────────────────────
    print(f"\n[2/3] Running {args.runs} segments × {args.steps} steps …", flush=True)
    fins = [None] * (args.runs + 1)
    rhos = [None] * args.runs
    us   = [None] * args.runs
    fins[0]   = fin.copy()
    sim_start = time.perf_counter()

    for i in range(args.runs):
        seg_start = time.perf_counter()
        rhos[i], us[i], fins[i+1] = Run(
            args.steps, fins[i], feq_init, obstacle,
            use_numba=use_numba,
        )
        seg_t   = time.perf_counter() - seg_start
        elapsed = time.perf_counter() - sim_start
        eta     = elapsed / (i + 1) * (args.runs - i - 1)
        pct     = (i + 1) / args.runs * 100
        print(f"  Segment {i+1:>3}/{args.runs}  |  {seg_t:5.1f}s  |  "
              f"elapsed {elapsed:6.1f}s  |  ETA {eta:5.1f}s  [{pct:.0f}%]",
              flush=True)

    sim_time = time.perf_counter() - sim_start
    total_steps = args.runs * args.steps
    print(f"\n  ✓ Simulation done in {sim_time:.2f}s  "
          f"({total_steps:,} steps,  {total_steps/sim_time:,.0f} steps/s)")

    # ── Animation ──────────────────────────────────────
    print("\n[3/3] Building & saving animation …", flush=True)
    anim_start = time.perf_counter()
    us_flat    = list(itertools.chain.from_iterable(us[:-1]))
    fig, anim  = build_animation(us_flat, vmax=args.vmax)
    save_animation(anim, args.output)
    # plt.close(fig)
    plt.show()
    anim_time = time.perf_counter() - anim_start
    print(f"  {len(us_flat)} frames saved in {anim_time:.2f}s")

    # ── Summary ────────────────────────────────────────
    total_time = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"  Total wall time : {total_time:.2f}s")
    print(f"  Setup + compile : {sim_start - t0:.2f}s")
    print(f"  Simulation      : {sim_time:.2f}s")
    print(f"  Animation I/O   : {anim_time:.2f}s")
    print("=" * 60)
    print(f"\n  Opening '{args.output}' …")
    #print(f"\n  Open '{args.output}' to view the animation.")

    # if sys.platform == "darwin":
    #     subprocess.run(["open", args.output])
    # elif sys.platform.startswith("linux"):
    #     subprocess.run(["xdg-open", args.output])
    # elif sys.platform == "win32":
    #     subprocess.run(["start", args.output], shell=True)

if __name__ == "__main__":
    main()