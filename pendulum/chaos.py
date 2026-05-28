#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# --- Read config ---
if len(sys.argv) < 2:
    print("Usage: python3 simulate.py <config_file>")
    sys.exit(1)

params = {}
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            key, val = line.split("=")
            params[key.strip()] = val.strip()

A     = float(params["A"])
B     = float(params["B"])
C     = float(params["C"])
OMEGA = float(params["OMEGA"])
pos0  = float(params["init_pos"])
vel0  = float(params["init_vel"])
T     = float(params["T"])
dt    = float(params["dt"])
name  = params["name"]
outfile = params["output"]


# --- Physics ---
def acc(t, pos, vel):
    return -A * vel - B * np.sin(pos) + C * np.sin(OMEGA * t)

def step(t, pos, vel):
    a        = acc(t, pos, vel)
    mid_vel  = vel + a * dt / 2
    mid_pos  = pos + vel * dt / 2
    mid_a    = acc(t + dt / 2, mid_pos, mid_vel)
    new_pos  = pos + mid_vel * dt
    new_vel  = vel + mid_a  * dt
    return t + dt, new_pos, new_vel

def wrap_theta(arr):
    return (arr + np.pi) % (2 * np.pi) - np.pi


# --- Run simulation ---
t, pos, vel = 0.0, pos0, vel0
ts, poss, vels = [t], [pos], [vel]

while t < T:
    t, pos, vel = step(t, pos, vel)
    ts.append(t)
    poss.append(pos)
    vels.append(vel)

poss = wrap_theta(np.array(poss))
vels = np.array(vels)


# --- Save phase space plot ---
os.makedirs(os.path.dirname(outfile), exist_ok=True)

# Save final state to txt
state_file = outfile.replace(".png", "_state.txt")
state = np.column_stack([
    np.array(ts),
    np.array(poss),
    np.array(vels)
])
np.savetxt(state_file, state, header="t  pos  vel", comments="#")
print(f"Saved state → {state_file}")

plt.figure(figsize=(6, 5))
plt.scatter(poss, vels, s=0.5, alpha=0.6, color="steelblue")
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\omega$ (rad/s)")
plt.title(f"Phase Space — {name}")
plt.tight_layout()
plt.savefig(outfile, dpi=120)
print(f"Saved → {outfile}")
plt.show()
