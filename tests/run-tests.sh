#!/bin/bash --login
#SBATCH -J mrisr-vae-tests
#SBATCH -o logs/%x-%j.out   # stdout and stderr, relative to the submission directory (the repo root)
#SBATCH -p serial
#SBATCH -t 0-1
#SBATCH --mem=5G

# CPU unit tests on synthetic data, a few minutes: the forward model against
# diffusion-priors' (skipped if ../diffusion-priors is missing), its adjoint,
# the backprojection, the Ladder VAE, the ELBO, the trainer and the dataset.
# Submit from the repo root: sbatch tests/run-tests.sh
module purge

srun uv run pytest -v -rs tests
