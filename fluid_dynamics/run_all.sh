#!/bin/bash

# List of obstacles to run — add new ones here as students create them
STEPS=2000
RUNS=10

OBSTACLES_DIR="./obstacles"
 
# Auto-detect all obstacle files — any .py file that isn't __init__.py
OBSTACLES=()
for f in ${OBSTACLES_DIR}/*.py; do
    OBSTACLES+=("$(basename "$f" .py)")
done
 
echo "========================================"
echo "  Found ${#OBSTACLES[@]} obstacles: ${OBSTACLES[*]}"
echo "========================================"
echo ""

for OBSTACLE in "${OBSTACLES[@]}"; do
    OUTPUT="${OBSTACLE}_flow.gif"
    echo "  Launching: ${OBSTACLE} → ${OUTPUT}"
    ./fluid_sim_numba.py --obstacle ${OBSTACLE} --steps ${STEPS} --runs ${RUNS} --output ${OUTPUT} &
done

echo ""
echo "  All jobs launched. Waiting for them to finish..."
wait
echo ""
echo "  Done! Output files:"
for OBSTACLE in "${OBSTACLES[@]}"; do
    echo "    ${OBSTACLE}_flow.gif"
done
