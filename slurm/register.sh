#!/bin/bash --login
#SBATCH -J mrisr-vae-register
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p multicore
#SBATCH -n 16
#SBATCH -t 0-2

# Step 3: registration_params.pickle for every subject (after breathhold
# correction, for CMR). Submit from the repo root, e.g.
#   sbatch slurm/register.sh --dataset cmr --data_dirs ../scratch/data/kaggle_niftis/train ../scratch/data/kaggle_niftis/test
#   sbatch slurm/register.sh --dataset brain --data_dirs ../scratch/data/fastmri/nifti
# Add --check to write nothing and compare with the existing pickles.
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/register.sh --dataset <cmr|brain> --data_dirs <dir>... [mrisr-register options]" >&2
    exit 1
fi
module purge

uv run mrisr-register --n_jobs 16 "$@"
