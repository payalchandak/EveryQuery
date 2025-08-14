#!/bin/bash
#SBATCH --job-name=cross-attn
#SBATCH --output=logs/n1_%j.out
#SBATCH --error=logs/n1_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=payal_chandak@hst.harvard.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate eq

cd ~/EveryQuery

source .env

python src/train.py experiment=experiment.yaml paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR "hydra.searchpath=[$MEDS_TORCH_CONFIG_DIR,$CONFIG_DIR]"