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

# Held-out view (la_4ch) predictions and scores on 2 validation subjects,
# with a protocol for the smoke run: the sampled mean and the direct
# predictor, inside sa_first, as configs/evaluation/kaggle_cmr_test.yaml.
cat > "$SMOKE_DIR/protocol.yaml" <<PROTOCOL
out_dir: $SMOKE_DIR/evaluation
data_dir: $(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['data_dir'])")
geometry: configs/kaggle_cmr_geometry.yaml
heldout_views: [la_4ch]
region: sa_first
conditioning_views: [all, sa, sa_first, la_2ch]
split: val
methods:
  vae: {checkpoint: $SMOKE_DIR/last.pt, n_samples: 2}
  vae_direct: {checkpoint: $SMOKE_DIR/last.pt}
PROTOCOL
for method in vae vae_direct; do
    srun uv run mrisr-evaluate --protocol "$SMOKE_DIR/protocol.yaml" --method "$method" --max_subjects 2
done
srun uv run mrisr-score --protocol "$SMOKE_DIR/protocol.yaml"

ls -R "$SMOKE_DIR/evaluation"
cat "$SMOKE_DIR/evaluation/summary.csv"
