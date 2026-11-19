#!/bin/bash --login
#SBATCH -J mrisr-vae-score
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-2
#SBATCH --mem=8G

# Scores every prediction in an evaluation protocol's out_dir by the same code
# and writes scores.csv and summary.csv there (the table comparing methods).
# Submit from the repo root once the methods and baselines are done:
#   sbatch slurm/score.sh configs/evaluation/kaggle_cmr_test.yaml
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/score.sh <protocol> [mrisr-score options]" >&2
    exit 1
fi
protocol=$1
shift
module purge

srun uv run mrisr-score --protocol "$protocol" "$@"
