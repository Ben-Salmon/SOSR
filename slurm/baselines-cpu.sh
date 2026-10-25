#!/bin/bash --login
#SBATCH -J mrisr-vae-baselines-cpu
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-4             # time 2 subjects first (--max_subjects 2) and adjust
#SBATCH --mem=16G

# The interpolation baselines (linear, cubic) on a CPU node: the same
# mrisr-baselines as slurm/baselines.sh, which is only needed for SMORE's
# `project`. --device falls back to the CPU here, as there is no GPU.
# Submit from the repo root:
#   sbatch slurm/baselines-cpu.sh <export|project|check> [mrisr-baselines options]
#
#   ckpt=../scratch/checkpoints/kaggle_cmr_direct/best.pt
#   test=/mnt/iusers01/fatpou01/bmh01/r99257bs/scratch/data/kaggle_niftis/test
#   1. sbatch slurm/baselines-cpu.sh export --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test
#   2. sbatch slurm/baselines-cpu.sh project --checkpoint $ckpt --data_dir $test --out_dir ../scratch/smore/test \
#          --evaluation_dir ../scratch/evaluation_baselines --sources linear cubic
#   3. uv run mrisr-score --evaluation_dir ../scratch/evaluation_baselines --name <linear|cubic>
# --evaluation_dir must hold an mrisr-evaluate --save_predictions run that
# includes the `sa_first` conditioning set, scored with --region sa_first
# (step 1 of slurm/baselines.sh). Without --sources linear cubic, `project`
# also looks for SMORE's outputs.
if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/baselines-cpu.sh <export|project|check> [mrisr-baselines options]" >&2
    exit 1
fi
module purge

srun uv run mrisr-baselines "$@"
