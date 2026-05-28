#!/usr/bin/env python3

"""
Lattice Boltzmann Method (LBM) Flow Simulation
-----------------------------------------------
Usage:
    python fluid_dynamics.py [--obstacle cylinder] [--steps 2000] [--runs 20] [--output flow.gif]

Arguments:
    --obstacle   Name of obstacle module inside the obstacles/ folder (default: cylinder)
                 Built-in options: cylinder, square, airfoil
                 Add your own: drop a new .py file in obstacles/ with a generate(nx, ny) function
    --steps      Number of LBM steps per run segment (default: 2000)
    --runs       Number of run segments (default: 20)
    --output     Output filename for animation, e.g. flow.mp4 or flow.gif (default: flow.gif)
    --vmax       Color scale max velocity (default: 0.07)
    --nx         Grid width  (default: 520)
    --ny         Grid height (default: 180)
"""

import argparse
import importlib
import time
import itertools
import sys
import os
import subprocess

import numpy as np
import matplotlib
#matplotlib.use("Agg")          # non-interactive backend — safe for terminal use
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# ──────────────────────────────────────────────
# Global lattice parameters (set after arg parse)
# ──────────────────────────────────────────────
nx: int
ny: int

# D2Q9 velocity vectors
v = np.zeros((9, 2), dtype="int")
v[0, :] = [0,  0]
v[1, :] = [0,  1]
v[2, :] = [0, -1]
v[3, :] = [1,  0]
v[4, :] = [-1, 0]
v[5, :] = [-1,-1]
v[6, :] = [-1, 1]
v[7, :] = [1, -1]
v[8, :] = [1,  1]


# ──────────────────────────────────────────────
# Dynamic obstacle loader
# ──────────────────────────────────────────────
def load_obstacle(name, nx, ny):
    """
    Import obstacles/<name>.py and call its generate(nx, ny) function.
    Students can add new obstacles by dropping a .py file in the obstacles/ folder.
    """
    obstacles_dir = os.path.join(os.path.dirname(__file__), "obstacles")
    available     = [f[:-3] for f in os.listdir(obstacles_dir)
                     if f.endswith(".py") and not f.startswith("_")]

    if name not in available:
        print(f"\n  ✗ Obstacle '{name}' not found.")
        print(f"    Available obstacles: {', '.join(sorted(available))}")
        print(f"    Add obstacles/{name}.py with a generate(nx, ny) function to create it.\n")
        sys.exit(1)

    # Make sure the obstacles folder is importable
    if obstacles_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))

    module = importlib.import_module(f"obstacles.{name}")
    return module.generate(nx, ny)


# ──────────────────────────────────────────────
# LBM physics
# ──────────────────────────────────────────────
def Micro2Macro(n):
    rho = np.sum(n, axis=0)
    u   = np.einsum("kd,kij->dij", v, n) / rho
    return rho, u


def Macro2Equilibrium(rho, u):
    weights  = np.array([4/9] + [1/9]*4 + [1/36]*4)
    vdotu    = np.einsum("kd,dij->kij", v, u)
    udotu    = np.einsum("dij,dij->ij",  u, u)
    t        = 1 + 3*vdotu + 4.5*vdotu**2 - 1.5*udotu
    n_eq     = weights[:, np.newaxis, np.newaxis] * rho[np.newaxis] * t
    return n_eq


def Collision(n, obstacle):
    rho, u  = Micro2Macro(n)
    n_eq    = Macro2Equilibrium(rho, u)
    omega   = 1.9572953736654806
    n_out   = n * (1 - omega) + omega * n_eq
    n_out[:, obstacle] = n[:, obstacle]
    return n_out


def Bounce(n, obstacle):
    n_out = n.copy()
    pairs = [(1,2), (3,4), (5,8), (6,7)]
    for a, b in pairs:
        n_out[a, obstacle] = n[b, obstacle]
        n_out[b, obstacle] = n[a, obstacle]
    return n_out


def MoveDensity(n):
    for k in range(n.shape[0]):
        n[k] = np.roll(n[k], v[k], (0, 1))
    return n


def Move(n, obstacle):
    n = Bounce(n, obstacle)
    n = MoveDensity(n)
    return n


def FixBoundary(n, n_init):
    n[:, 0, :]  = n_init[:, 0, :]
    n[4:7, -1, :] = n[4:7, -2, :]
    return n


def Setup():
    rho = np.ones((nx, ny))
    u   = np.zeros((2, nx, ny))
    y   = np.arange(ny)[np.newaxis, :]
    u[0] = 0.04 * (1.0 + 1e-4 * np.sin(y / ny * 2 * np.pi))
    return Macro2Equilibrium(rho, u)


def Run(steps, n, n_init, obstacle, record_every=100):
    rhos, us = [], []
    for step in range(steps + 1):
        n = FixBoundary(n, n_init)
        n = Collision(n, obstacle)
        n = Move(n, obstacle)
        if step % record_every == 0:
            rho, u = Micro2Macro(n)
            u2 = np.sqrt(np.einsum("dij,dij->ij", u, u))
            rhos.append(rho)
            us.append(u2.T)
    return rhos, us, n


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
    global nx, ny, v

    parser = argparse.ArgumentParser(
        description="Lattice Boltzmann flow simulation with animation output."
    )
    parser.add_argument("--obstacle", default="cylinder",
                        help="Obstacle name — must match a file in obstacles/ (default: cylinder)")
    parser.add_argument("--steps",   type=int, default=2000,
                        help="LBM steps per run segment (default: 2000)")
    parser.add_argument("--runs",    type=int, default=10,
                        help="Number of run segments (default: 10)")
    parser.add_argument("--output",  default="flow.gif",
                        help="Output file for animation, e.g. flow.gif or flow.mp4")
    parser.add_argument("--vmax",    type=float, default=0.07,
                        help="Colour scale max velocity (default: 0.07)")
    parser.add_argument("--nx",      type=int, default=260,
                        help="Grid width  (default: 260)")
    parser.add_argument("--ny",      type=int, default=90,
                        help="Grid height (default: 90)")
    args = parser.parse_args()

    # Apply grid size globally
    nx, ny = args.nx, args.ny

    print("=" * 60)
    print("  Lattice Boltzmann Method — Flow Simulation")
    print("=" * 60)
    print(f"  Grid      : {nx} × {ny}")
    print(f"  Obstacle  : {args.obstacle}")
    print(f"  Steps/run : {args.steps}   |   Runs: {args.runs}")
    print(f"  Output    : {args.output}")
    print("=" * 60)

    # ── Setup ──────────────────────────────────
    t0 = time.perf_counter()
    print("\n[1/3] Setting up lattice …", flush=True)
    fin       = Setup()
    feq_init  = fin.copy()
    obstacle  = load_obstacle(args.obstacle, nx, ny)
    print(f"      Done. Obstacle cells: {obstacle.sum():,}")

    # ── Simulation loop ────────────────────────
    print(f"\n[2/3] Running {args.runs} segments × {args.steps} steps …", flush=True)
    fins = [None] * (args.runs + 1)
    rhos = [None] *  args.runs
    us   = [None] *  args.runs
    fins[0] = fin.copy()

    total_steps = args.runs * args.steps
    sim_start   = time.perf_counter()

    for i in range(args.runs):
        seg_start = time.perf_counter()
        rhos[i], us[i], fins[i + 1] = Run(
            args.steps, fins[i], feq_init, obstacle
        )
        seg_time = time.perf_counter() - seg_start
        elapsed  = time.perf_counter() - sim_start
        done_pct = (i + 1) / args.runs * 100
        eta      = elapsed / (i + 1) * (args.runs - i - 1)
        print(f"  Segment {i+1:>3}/{args.runs}  |  {seg_time:5.1f}s  |  "
              f"elapsed {elapsed:6.1f}s  |  ETA {eta:5.1f}s  [{done_pct:.0f}%]",
              flush=True)

    sim_time = time.perf_counter() - sim_start
    print(f"\n  ✓ Simulation complete in {sim_time:.2f}s "
          f"({total_steps:,} total steps, "
          f"{total_steps/sim_time:,.0f} steps/s)")

    # ── Animation ─────────────────────────────
    print("\n[3/3] Building & saving animation …", flush=True)
    anim_start = time.perf_counter()
    us_flat    = list(itertools.chain.from_iterable(us[:-1]))
    fig, anim  = build_animation(us_flat, vmax=args.vmax)
    # save_animation(anim, args.output)
    # plt.close(fig)
    plt.show()
    anim_time = time.perf_counter() - anim_start
    print(f"  Animation built & saved in {anim_time:.2f}s  ({len(us_flat)} frames)")

    # ── Summary ───────────────────────────────
    total_time = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"  Total wall time : {total_time:.2f}s")
    print(f"  Setup           : {sim_start - t0:.2f}s")
    print(f"  Simulation      : {sim_time:.2f}s")
    print(f"  Animation I/O   : {anim_time:.2f}s")
    print("=" * 60)
    print(f"\n  Opening '{args.output}' …")
    #print(f"\n  Open '{args.output}' to view the animation.")

    if sys.platform == "darwin":
        subprocess.run(["open", args.output])
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", args.output])
    elif sys.platform == "win32":
        subprocess.run(["start", args.output], shell=True)


if __name__ == "__main__":
    main()
