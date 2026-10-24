#!/bin/bash --login
#SBATCH -J mrisr-vae-smore
#SBATCH -o logs/%x-%A_%a.out   # one log per stack, relative to the submission directory (the repo root)
#SBATCH -p gpuL
#SBATCH -n 1
#SBATCH -c 8               # SMORE's DataLoader uses 8 workers
#SBATCH -t 0-2             # per stack; time one stack first (--array=1) and adjust
#SBATCH -G 1
#SBATCH --mem=16G

# SMORE (../smore, unmodified, default options) on one short-axis stack per
# array task: line $SLURM_ARRAY_TASK_ID of the stacks.txt that
# `mrisr-baselines export` writes. The output goes to
# <subject>/smore/<stack>/<stack>_smore4.nii.gz beside the exported stack,
# where `mrisr-baselines project` looks for it. Stacks already done are
# skipped, so the array can be resubmitted after failures.
#
# SMORE lives in its own environment:
#   uv venv ../smore/.venv && uv pip install --python ../smore/.venv -e ../smore
#
# Submit from the repo root, e.g. (N = wc -l < stacks.txt):
#   sbatch --array=1 slurm/smore.sh ../scratch/smore/test/stacks.txt        (timing)
#   sbatch --array=1-N%20 slurm/smore.sh ../scratch/smore/test/stacks.txt
if [ $# -lt 1 ]; then
    echo "Usage: sbatch --array=1-N slurm/smore.sh <stacks.txt>" >&2
    exit 1
fi
stacks=$1
smore_bin=$(realpath ../smore/.venv/bin)

stack=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$stacks")
if [ -z "$stack" ]; then
    echo "No line $SLURM_ARRAY_TASK_ID in $stacks" >&2
    exit 1
fi
name=$(basename "$stack" .nii.gz)
out_dir=$(dirname "$stack")/smore
out="$out_dir/$name/${name}_smore4.nii.gz"
if [ -f "$out" ]; then
    echo "$out exists; skipping"
    exit 0
fi
module purge

# run-smore calls smore-train and smore-test by name, so they must be on PATH.
export PATH="$smore_bin:$PATH"
echo "SMORE on $stack ($(git -C ../smore rev-parse --short HEAD))"
start=$(date +%s)
srun "$smore_bin/run-smore" --in-fpath "$stack" --out-dir "$out_dir"
echo "Elapsed: $(( $(date +%s) - start )) s"

# run-smore does not pass on its steps' exit codes.
if [ ! -f "$out" ]; then
    echo "SMORE wrote no $out" >&2
    exit 1
fi
