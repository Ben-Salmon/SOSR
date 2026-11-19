#!/bin/bash --login
#SBATCH -J mrisr-vae-evaluate
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-4
#SBATCH -G 1
#SBATCH --mem=16G

# One method of an evaluation protocol: its predictions of the held-out view,
# written into the protocol's out_dir beside every other method's. Submit
# from the repo root, one job per method (they can run at the same time):
#   sbatch slurm/evaluate.sh configs/evaluation/kaggle_cmr_test.yaml vae
#   sbatch slurm/evaluate.sh configs/evaluation/kaggle_cmr_test.yaml vae_direct
#   sbatch slurm/evaluate.sh configs/evaluation/kaggle_cmr_test.yaml supervised
# Timing run: ... vae --max_subjects 2 --out_dir ../scratch/evaluation/timing
# Then, once the baselines are done too: sbatch slurm/score.sh <protocol>
if [ $# -lt 2 ]; then
    echo "Usage: sbatch slurm/evaluate.sh <protocol> <method> [mrisr-evaluate options]" >&2
    exit 1
fi
protocol=$1
method=$2
shift 2
module purge

srun uv run mrisr-evaluate --protocol "$protocol" --method "$method" "$@"
