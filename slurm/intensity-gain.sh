#!/bin/bash --login
#SBATCH -J mrisr-vae-intensity-gain
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-4
#SBATCH -G 1
#SBATCH --mem=16G

# Step 2 (CMR only): each long-axis view's intensity gain relative to the
# short-axis slices, after the breathhold correction (step 1). Writes
# <subject>/<view>_intensity_gain.json, which the dataset applies. Rerun it
# whenever the breathhold correction is rerun.
# Submit from the repo root, once per directory, e.g.
#   sbatch slurm/intensity-gain.sh --data_dir ../scratch/data/kaggle_niftis/train
#   sbatch slurm/intensity-gain.sh --data_dir ../scratch/data/kaggle_niftis/test
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/intensity-gain.sh --data_dir <dir> [mrisr-intensity-gain options]" >&2
    exit 1
fi
module purge

srun uv run mrisr-intensity-gain "$@"
