#!/bin/bash
# Sweep over num_hidden_layers: 6, 12, 16, 22
# Submits all jobs simultaneously via sbatch

set -euo pipefail

for N_LAYERS in 6 12 16 22; do
    echo "Submitting job with num_hidden_layers=$N_LAYERS"
    sbatch --job-name="eq_layers_${N_LAYERS}" \
        scripts/train_eq.sh \
        lightning_module.model.num_hidden_layers=$N_LAYERS \
        only_preprocess=false
done

echo "All jobs submitted."
