#!/bin/bash --login
#SBATCH -J mrisr-vae-baselines
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-2
#SBATCH -G 1
#SBATCH --mem=16G

# Interpolation and SMORE baselines on the held-out view (see
# src/mrisr_vae/baselines.py). Submit from the repo root:
#   sbatch slurm/baselines.sh <export|project|check> [mrisr-baselines options]
#
# The whole pipeline, with the direct predictor's checkpoint for the latent
# geometry (every model shares it):
#   ckpt=../scratch/checkpoints/kaggle_cmr_direct/best.pt
#   test=/mnt/iusers01/fatpou01/bmh01/r99257bs/scratch/data/kaggle_niftis/test
#   1. sbatch slurm/evaluate.sh configs/kaggle_cmr_direct.yaml \
#          --conditioning_views sa --save_predictions --out_dir ../scratch/evaluation_baselines
#   2. sbatch slurm/baselines.sh export --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test
#   3. sbatch --array=1-N%20 slurm/smore.sh ../scratch/smore/test/stacks.txt
#   4. sbatch slurm/baselines.sh check --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test \
#          --max_subjects 2
#   5. sbatch slurm/baselines.sh project --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test \
#          --evaluation_dir ../scratch/evaluation_baselines
#   6. uv run mrisr-score --evaluation_dir ../scratch/evaluation_baselines --name <direct|linear|cubic|smore>
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/baselines.sh <export|project|check> [mrisr-baselines options]" >&2
    exit 1
fi
module purge

srun uv run mrisr-baselines "$@"
