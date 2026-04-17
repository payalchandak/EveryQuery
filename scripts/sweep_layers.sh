#!/bin/bash
# Sweep over num_hidden_layers: 6, 12, 16, 22
# Submits all jobs with a 30s stagger between submissions.

set -euo pipefail

LAYER_SIZES=(6 12 16 22)
NUM_LAYERS=${#LAYER_SIZES[@]}

for i in "${!LAYER_SIZES[@]}"; do
    N_LAYERS="${LAYER_SIZES[$i]}"
    echo "Submitting job with num_hidden_layers=$N_LAYERS"
    sbatch --job-name="eq_layers_${N_LAYERS}" \
        scripts/train_eq.sh \
        "lightning_module.model.num_hidden_layers=$N_LAYERS"

    # If this is NOT the last job, wait 30 seconds.
    if [ $((i + 1)) -lt "$NUM_LAYERS" ]; then
        echo "Waiting 30 seconds before next submission..."
        sleep 30
    fi
done

echo "All jobs submitted."
