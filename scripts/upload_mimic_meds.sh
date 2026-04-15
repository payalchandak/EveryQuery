#!/bin/bash
#SBATCH --job-name=mimic_meds_upload
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

SRC_ROOT="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
BUCKET="gs://every-query-runs"
DEST="${BUCKET}/meds/MIMIC_MEDS/MEDS_cohort"

SUBDIRS=(
    "intermediate/data"
    "intermediate/metadata"
    "processed/data"
    "processed/metadata"
)

mkdir -p logs
echo "Starting MIMIC_MEDS upload on $(hostname) at $(date)"
echo "Source: ${SRC_ROOT}"
echo "Destination: ${DEST}"

for sub in "${SUBDIRS[@]}"; do
    src="${SRC_ROOT}/${sub}"
    dst="${DEST}/${sub}"
    if [[ ! -d "${src}" ]]; then
        echo "ERROR: source directory missing: ${src}" >&2
        exit 1
    fi
    echo "---"
    echo "Syncing ${src} -> ${dst}"
    gsutil -m rsync -r "${src}" "${dst}"
done

echo "Finished at $(date)"
