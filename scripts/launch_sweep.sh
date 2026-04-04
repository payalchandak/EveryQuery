#!/bin/bash
# ── Launch a W&B sweep on the cluster ─────────────────────────
#
# Usage (from the repo root):
#   bash scripts/launch_sweep.sh              # 8 agents, default config
#   bash scripts/launch_sweep.sh 12           # 12 agents
#   bash scripts/launch_sweep.sh 5 sweep_tier2.yaml
#
# What it does:
#   1. Creates a W&B sweep from the config YAML
#   2. Submits N sbatch jobs, each running 1 trial
#
# ───────────────────────────────────────────────────────────────

set -euo pipefail

NUM_AGENTS="${1:-8}"
SWEEP_CONFIG="${2:-sweep_tier1.yaml}"

cd "$(git rev-parse --show-toplevel)"

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

echo "Creating W&B sweep from $SWEEP_CONFIG ..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Parse the sweep ID from wandb output (last line contains the ID)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K\S+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Error: could not parse sweep ID from wandb output."
    echo "Create the sweep manually and run:"
    echo "  for i in \$(seq 1 $NUM_AGENTS); do sbatch scripts/sweep_worker.sh <SWEEP_ID>; done"
    exit 1
fi

echo ""
echo "Sweep ID: $SWEEP_ID"
echo "Submitting $NUM_AGENTS agents ..."

for i in $(seq 1 "$NUM_AGENTS"); do
    sbatch scripts/sweep_worker.sh "$SWEEP_ID"
done

echo ""
echo "Done. $NUM_AGENTS jobs submitted."
echo "Monitor at: https://wandb.ai/$SWEEP_ID"
