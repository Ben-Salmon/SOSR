#!/bin/bash --login
#SBATCH -J mrisr-vae-evaluate
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH --partition=gpuq
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Scores a model on the held-out view(s) (data: heldout_views) and saves the
# predictions on each view's own grid, as the config's `evaluate` section sets.
# Submit from the repo root: sbatch slurm/evaluate.sh <config> [mrisr-evaluate options]
# e.g. sbatch slurm/evaluate.sh configs/kaggle_cmr.yaml
#      sbatch slurm/evaluate.sh configs/kaggle_cmr_supervised.yaml
#      sbatch slurm/evaluate.sh configs/kaggle_cmr.yaml --max_subjects 2 --out_dir <dir>   (timing)
# Options given here override the config. Models evaluated with the same
# data_dir, frame and conditioning_views are scored on the same subjects,
# frames and voxels, so their metrics.csv files compare row by row.
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/evaluate.sh <config> [mrisr-evaluate options]" >&2
    exit 1
fi
config=$1
shift

uv run mrisr-evaluate --config "$config" "$@"

# To score another method's predictions, saved beside these as
# <conditioning>/<subject>/la_4ch_pred_<name>.nii.gz:
#   uv run mrisr-score --evaluation_dir <out_dir> --name <name>
