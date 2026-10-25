#!/bin/bash --login
#SBATCH -J mrisr-vae-baselines
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH --partition=gpuq
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Interpolation and SMORE baselines on the held-out view, from each subject's
# lowest-numbered short-axis stack (see src/mrisr_vae/baselines.py). Submit
# from the repo root:
#   sbatch slurm/baselines.sh <export|project|check> [mrisr-baselines options]
#
# The whole pipeline, with the direct predictor's checkpoint for the latent
# geometry (every model shares it):
#   ckpt=../data/kaggle_cmr/checkpoints/kaggle_cmr_direct/best.pt
#   test=../data/kaggle_cmr/data/kaggle_niftis/test
#   1. sbatch slurm/evaluate.sh configs/kaggle_cmr_direct.yaml \
#          --conditioning_views sa_first --region sa_first --save_predictions --out_dir ../data/kaggle_cmr/evaluation_baselines
#   2. sbatch slurm/baselines.sh export --checkpoint $ckpt --data_dir $test --out_dir ../data/kaggle_cmr/smore/test
#   3. sbatch --array=1-N%20 slurm/smore.sh ../data/kaggle_cmr/smore/test/stacks.txt
#      (N = wc -l < stacks.txt; SMORE is pip-installed for module python/3.11.9:
#      see slurm/smore.sh. This job and the others run in the repo's uv environment.)
#   4. sbatch slurm/baselines.sh check --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test \
#          --max_subjects 2
#   5. sbatch slurm/baselines.sh project --checkpoint $ckpt --data_dir $test --out_dir ../data/kaggle_cmr/smore/test \
#          --evaluation_dir ../data/kaggle_cmr/evaluation_baselines
#      (linear and cubic alone need no GPU: slurm/baselines-cpu.sh, with --sources linear cubic)
#   6. uv run mrisr-score --evaluation_dir ../data/kaggle_cmr/evaluation_baselines --name <linear|cubic|smore>
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/baselines.sh <export|project|check> [mrisr-baselines options]" >&2
    exit 1
fi

uv run mrisr-baselines "$@"
