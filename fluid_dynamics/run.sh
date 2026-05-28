#!/bin/bash

OBSTACLE=${1:-"cylinder"}
STEPS=2000
RUNS=10
OUTPUT="flow.gif" #or "flow.mp4"

./fluid_sim_numba.py --obstacle ${OBSTACLE} --output ${OUTPUT}
