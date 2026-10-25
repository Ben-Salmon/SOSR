#!/bin/bash --login
#SBATCH -J mrisr-vae-smore
#SBATCH -o logs/%x-%A_%a.out   # one log per stack, relative to the submission directory (the repo root)
#SBATCH -n 1
#SBATCH -c 8               # SMORE's DataLoader uses 8 workers
#SBATCH --time=01:00:00           # per stack; time one stack first (--array=1) and adjust
#SBATCH --mem=16G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

# SMORE (unmodified, default options) on one short-axis stack per array task:
# line $SLURM_ARRAY_TASK_ID of the stacks.txt that `mrisr-baselines export`
# writes. The output goes to <subject>/smore/<stack>/<stack>_smore4.nii.gz
# beside the exported stack, where `mrisr-baselines project` looks for it.
# Stacks already done are skipped, so the array can be resubmitted after
# failures.
#
# SMORE is pip-installed for the cluster's Python module, not into this repo's
# environment. Install it from GitLab: the `smore` on PyPI is an unrelated package.
#   module load python/3.11.9
#   pip install --user git+https://gitlab.com/iacl/smore
#
# Submit from the repo root, e.g. (N = wc -l < stacks.txt):
#   sbatch --array=1 slurm/smore.sh ../scratch/smore/test/stacks.txt        (timing)
#   sbatch --array=1-N%20 slurm/smore.sh ../scratch/smore/test/stacks.txt
if [ $# -lt 1 ]; then
    echo "Usage: sbatch --array=1-N slurm/smore.sh <stacks.txt>" >&2
    exit 1
fi
stacks=$1

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
module load python/3.11.9

# SMORE's commands are console scripts, not modules, so `python -m` cannot
# run them. run-smore calls smore-train and smore-test by name, so all three
# must be on PATH; `pip install --user` puts them in the user base's bin.
export PATH="$(python -m site --user-base)/bin:$PATH"
for cmd in run-smore smore-train smore-test; do
    if ! command -v "$cmd" > /dev/null; then
        echo "$cmd not found: pip install --user git+https://gitlab.com/iacl/smore (see above)" >&2
        exit 1
    fi
done
version=$(python -c "from importlib.metadata import version; print(version('smore'))")
echo "SMORE $version ($(command -v run-smore)) on $stack"
start=$(date +%s)
srun run-smore --in-fpath "$stack" --out-dir "$out_dir"
echo "Elapsed: $(( $(date +%s) - start )) s"

# run-smore does not pass on its steps' exit codes.
if [ ! -f "$out" ]; then
    echo "SMORE wrote no $out" >&2
    exit 1
fi
