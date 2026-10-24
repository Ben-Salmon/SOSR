#!/bin/bash --login
#SBATCH -J mrisr-vae-train
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuA
#SBATCH -n 1
#SBATCH -c 8                # CPUs for DataLoader workers
#SBATCH -t 2-0              # train: max_hours in the config stops it cleanly before this
#SBATCH -G 1
#SBATCH --mem=32G

# Submit from the repo root: sbatch slurm/train.sh <config> [mrisr-train options]
# e.g. sbatch slurm/train.sh configs/kaggle_cmr.yaml --resume
# With --resume, resubmit to carry on from <checkpoint_dir>/last.pt.
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/train.sh <config> [mrisr-train options]" >&2
    exit 1
fi
config=$1
shift
module purge

srun uv run mrisr-train --config "$config" "$@"
