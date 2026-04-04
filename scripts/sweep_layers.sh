#!/bin/bash
# Sweep over num_hidden_layers: 6, 12, 16, 22
# Submits all jobs simultaneously via sbatch

set -euo pipefail

for N_LAYERS in 6 12 16 22; do
    echo "Submitting job with lr=$N_LAYERS"
    sbatch --job-name="eq_layers_${N_LAYERS}" \
        scripts/train_eq.sh \
        lightning_module.model.num_hidden_layers=$N_LAYERS \
        only_preprocess=false

    # If this is NOT the last job, wait 30 seconds
    if [ $((i + 1)) -lt $NUM_LAYERS ]; then
        echo "Waiting 30 seconds before next submission..."
        sleep 30
    fi
done

echo "All jobs submitted."
