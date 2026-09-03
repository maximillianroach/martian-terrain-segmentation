#!/bin/bash
#SBATCH --job-name=ai4mars_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ai4mars_%j.out
#SBATCH --error=logs/ai4mars_%j.err

module load miniconda
conda activate rl-lab

export WANDB_API_KEY=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

cd martian-terrain-segmentation
python -m train
