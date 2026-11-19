#!/bin/bash --login
#SBATCH -J mrisr-vae-breathhold
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-12
#SBATCH -G 1
#SBATCH --mem=16G

# Step 1 (CMR only): breathhold correction of every subject in one directory.
# Writes <subject>/<view>_affine_correction.npy. Afterwards rerun
# slurm/register.sh: long-axis corrections only reach the model through it.
# Submit from the repo root, once per directory, e.g.
#   sbatch slurm/breathhold.sh --data_dir ../scratch/data/kaggle_niftis/train
#   sbatch slurm/breathhold.sh --data_dir ../scratch/data/kaggle_niftis/test
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/breathhold.sh --data_dir <dir> [mrisr-breathhold options]" >&2
    exit 1
fi
module purge

srun uv run mrisr-breathhold "$@"
