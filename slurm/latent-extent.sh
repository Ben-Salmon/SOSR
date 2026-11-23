#!/bin/bash --login
#SBATCH -J mrisr-vae-latent-extent
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p multicore
#SBATCH -n 16
#SBATCH -t 0-2

# Step 4: the ROI's extent over the training subjects, and the latent grid it
# suggests, to copy into the geometry yaml. Submit from the repo root, e.g.
#   sbatch slurm/latent-extent.sh --dataset cmr --data_dir ../scratch/data/kaggle_niftis/train --su_per_lvox 1.8
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/latent-extent.sh --dataset <cmr|brain> --data_dir <dir> [mrisr-latent-extent options]" >&2
    exit 1
fi
module purge

uv run mrisr-latent-extent --n_jobs 16 "$@"
