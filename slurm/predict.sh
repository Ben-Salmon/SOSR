#!/bin/bash --login
#SBATCH -J mrisr-vae-predict
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-1
#SBATCH -G 1
#SBATCH --mem=8G

# Submit from the repo root: sbatch slurm/predict.sh <config> [mrisr-predict options]
# e.g. sbatch slurm/predict.sh configs/kaggle_cmr.yaml --max_subjects 2
# The config's `predict` section sets the checkpoint (best or last), out_dir,
# data_dir, split and method (direct, sampling or both); options given here
# override it.
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/predict.sh <config> [mrisr-predict options]" >&2
    exit 1
fi
config=$1
shift
module purge

srun uv run mrisr-predict --config "$config" "$@"
