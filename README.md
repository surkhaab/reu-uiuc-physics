# reu-uiuc-physics

# Single Pendulum Simulation
Simulates a driven damped pendulum across three physical regimes: undamped, damped/driven, and chaotic.

---
 
## Setup

```bash
conda create -n reu python=3.11
conda activate reu
conda install numpy matplotlib pillow numba  # pillow and numba only needed for fluid dynamics
conda install -c conda-forge ffmpeg  # optional, only needed for .mp4 output
```

```bash
git clone git@github.com:surkhaab/reu-uiuc-physics.git
```
 
---
 
## Project Structure
 
```
workshop/
├── simulate.py          # simulation script (do not modify)
├── configs/             # one .txt config file per run
│   ├── version1_undamped.txt
│   ├── version2_damped.txt
│   └── version3_chaotic.txt
└── plots/               # output phase space plots (created automatically)
```
 
---
 
## Running a Simulation
 
Pass a config file as the argument:
 
```bash
python3 simulate.py configs/version1_undamped.txt
```
 
The script reads the config, runs the simulation, and saves a phase space plot to the path specified in `output_plot`.
 
---
 
## Config File Format
 
Plain text, one `key = value` per line. Lines starting with `#` are comments.
 
```
# example config
name = my_run
A = 0.1          # damping coefficient
B = 1.0          # restoring force
C = 2.0          # driving force amplitude
OMEGA = 1.2      # driving frequency
init_pos = 0.0   # initial angle (radians)
init_vel = 0.1   # initial angular velocity (rad/s)
T = 500          # total simulation time (s)
dt = 0.01        # timestep (s)
l1 = 1.0         # pendulum length (m)
output_plot = plots/my_run.png
```
 
---
 
## The Three Configs
 
| File | Regime | Key parameters |
|---|---|---|
| `version1_undamped.txt` | Undamped, undriven | A=0, C=0 — simple harmonic motion |
| `version2_damped.txt` | Damped and driven | A=0.1, C=0.1 — settles into periodic motion |
| `version3_chaotic.txt` | Chaotic | A=0.1, C=2.0 — never settles, sensitive to initial conditions |
 
---
 

# LBM Flow Simulation
Simulates 2D fluid flow around obstacles using the Lattice Boltzmann Method (LBM),
a computational fluid dynamics technique that models fluid behavior at the mesoscopic
scale. The simulation produces animated visualizations of the resulting flow fields for a variety of obstacle geometries.

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

## Acknowledgements
 
Both simulations were originally developed as course exercises for PHYS 246 (UIUC), and adapted and extended by Surkhab Kaur, 2026.
 
- Single Pendulum — Bryan Clark, George Gollin, and Ryan Levy
- LBM Fluid Dynamics — Bryan Clark
