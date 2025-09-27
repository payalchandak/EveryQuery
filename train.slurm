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

cd /n/data1/hms/dbmi/zaklab/payal/EveryQuery

source .env

set -a; . ./.env; set +a; 

python source/train.py