# reu-uiuc-physics

# LBM Flow Simulation
Simulates 2D fluid flow around obstacles using the Lattice Boltzmann Method (LBM),
a computational fluid dynamics technique that models fluid behavior at the mesoscopic
scale. The simulation produces animated visualizations of the resulting flow fields for a variety of obstacle geometries.

## Installation

```bash
conda create -n reu python=3.11
conda activate reu
conda install numpy matplotlib pillow numba
conda install -c conda-forge ffmpeg  # optional, only needed for .mp4 output
```

## Usage

Run a single simulation:
```bash
./fluid_sim_numba.py # runs with cylinder obstacle by default
./fluid_sim_numba.py --obstacle square # specify obstacle
```

Common options:
```bash
./fluid_dynamics.py --obstacle cylinder --steps 2000 --output flow.gif
```

Run all obstacles in parallel:
```bash
./run_all.sh
```

## Available Obstacles

| Name               | Description                        |
|--------------------|------------------------------------|
| `cylinder`         | Single circular cylinder           |
| `square`           | Single square                      |
| `airfoil`          | NACA 0012 airfoil                  |
| `double_rect`      | Two stacked rectangular bars       |
| `triple_cylinder`  | A triangle of cylinders            |
| `star`             | Five-pointed star                  |

## Optional - Adding Your Own Obstacle

Create a new file in the `obstacles/` folder:
```python
# obstacles/myshape.py
import numpy as np

def generate(nx, ny):
    obstacle = np.zeros((nx, ny), dtype=bool)
    # mark cells as True to block flow
    return obstacle
```

Then run it with:
```bash
./fluid_dynamics.py --obstacle myshape
```

## Credits

Originally developed as a course exercise (PHYS 246) by Bryan Clark (UIUC), adapted and extended by Surkhab Kaur, 2026.
