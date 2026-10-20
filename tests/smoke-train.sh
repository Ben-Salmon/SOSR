#!/bin/bash --login
#SBATCH -J mrisr-vae-smoke
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-2
#SBATCH -G 1
#SBATCH --mem=32G

# End to end on real data at toy scale: 2 epochs on 8 subjects (validating on
# 2), a resumed third epoch, then predictions and held-out-view (la_4ch)
# evaluation for 2 validation subjects.
# Writes only to the smoke directory below, which it clears first, and logs to
# wandb offline.
# Submit from the repo root: sbatch tests/smoke-train.sh
module purge
set -ex

SMOKE_DIR=../scratch/checkpoints/mrisr_vae_smoke
CONFIG=configs/kaggle_cmr.yaml
rm -rf "$SMOKE_DIR"

srun uv run mrisr-train --config "$CONFIG" --checkpoint_dir "$SMOKE_DIR" \
    --max_subjects 8 --max_epochs 2 --wandb_mode offline
srun uv run mrisr-train --config "$CONFIG" --checkpoint_dir "$SMOKE_DIR" \
    --max_subjects 8 --max_epochs 3 --wandb_mode offline --resume
srun uv run mrisr-predict --checkpoint "$SMOKE_DIR/last.pt" --out_dir "$SMOKE_DIR/predictions" \
    --split val --max_subjects 2 --n_samples 2 --save_samples

ls -R "$SMOKE_DIR/predictions"

# Held-out view (la_4ch) scores and predictions on the same 2 subjects.
srun uv run mrisr-evaluate --checkpoint "$SMOKE_DIR/last.pt" --out_dir "$SMOKE_DIR/evaluation" \
    --split val --max_subjects 2 --conditioning_views all sa la_2ch --n_samples 2 --save_predictions

ls -R "$SMOKE_DIR/evaluation"
cat "$SMOKE_DIR/evaluation/metrics.csv"
