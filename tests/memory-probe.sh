#!/bin/bash --login
#SBATCH -J mrisr-vae-memory
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-1
#SBATCH -G 1
#SBATCH --mem=32G

# A few full-size training steps on real subjects: peak GPU memory, seconds per
# step and per-item loading time, for n_filters 32 and 64 at batch size 2.
# Submit from the repo root: sbatch tests/memory-probe.sh
module purge

srun uv run python tests/memory_probe.py --config configs/kaggle_cmr.yaml --n_filters 32 64 --batch_size 2
