#!/bin/bash --login
#SBATCH -J mrisr-vae-baselines
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-2
#SBATCH -G 1
#SBATCH --mem=16G

# Interpolation and SMORE baselines of an evaluation protocol (see
# src/mrisr_vae/evaluation/baselines.py). Submit from the repo root, in order:
#   p=configs/evaluation/kaggle_cmr_test.yaml
#   1. sbatch slurm/baselines.sh export $p
#   2. sbatch --array=1-N%20 slurm/smore.sh <work_dir>/stacks.txt   (export prints N)
#   3. sbatch slurm/baselines.sh check $p --max_subjects 2           (SMORE's grid is read right)
#   4. sbatch slurm/baselines.sh project $p
# Then: sbatch slurm/score.sh $p
if [ $# -lt 2 ]; then
    echo "Usage: sbatch slurm/baselines.sh <export|project|check> <protocol> [mrisr-baselines options]" >&2
    exit 1
fi
command=$1
protocol=$2
shift 2
module purge

srun uv run mrisr-baselines "$command" --protocol "$protocol" "$@"
