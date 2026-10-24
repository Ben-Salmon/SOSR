# mrisr_vae

Through-plane MRI super-resolution without high-resolution training data. A
Ladder VAE learns an empirical-Bayes prior over high-resolution volumes from
low-resolution observations, using the forward model of
[diffusion-priors](../diffusion-priors).

This repo replaces that repo's expectation-maximisation loop (posterior
sampling, then retraining a diffusion model, lap after lap) with a single
training run.

## Model

As in Hierarchical DivNoising (Prakash et al., ICLR 2022), the decoder ends in
a known measurement model instead of a learned one. Here that model is the MRI
forward model of each view:

```
z ~ p(z)                                Ladder VAE prior (top-down path)
x = g(z)                                volume on the latent grid
y_v ~ N(A_v x, sigma_v^2 I)             each view: Lanczos resampling, Gaussian
                                        slice profile, sampling at the view's voxels
```

- **Training objective.** Maximising the ELBO over the dataset fits
  p(x) = ∫ p(x|z) p(z) dz so that, pushed through the forward models, it
  reproduces the observations. Because each scan's slices sit at a random
  sub-voxel offset, the forward models differ between subjects, which is what
  makes the prior identifiable.
- **Encoder input.** The encoder q(z|y) sees every available view
  backprojected onto the latent grid. Channel 0 is the precision-weighted
  average A^T W y / A^T W 1; channel 1 is the relative coverage. This lets it
  take any combination of views, at any resolution.
- **View dropout.** In training each view is left out of the encoder's input
  with probability `view_drop_probability` (keeping at least one), but stays
  in the likelihood. q(z | some views) still bounds log p(all views), so this
  is still an ELBO, and it makes the latents predict views they were not
  shown instead of copying the input.
- **KL warm-up.** The KL's weight rises linearly from 0 to 1 over
  `kl_warmup_steps` optimiser steps, so the upper latent layers are put to
  use before the KL can collapse them. The learning-rate plateaus, best
  checkpoint and early stopping wait for it to end.
- **Super-resolution.** Draw z ~ q(z|y) and decode x = g(z). The mean of
  several samples is the MMSE estimate.
- **Direct predictor** (COSDD's direct denoiser, `model: direct_predictor`).
  A UNet on the same encoder input is trained alongside, by squared error
  against the VAE's detached posterior samples. It therefore learns their
  mean, E[x|y], and gives the MMSE estimate in one pass (`sr_direct.nii.gz`).
  It has its own optimiser and never affects the VAE. With `train:
  direct_only` and `pretrained_vae`, a fresh one is trained alone on a
  trained, frozen VAE (`configs/kaggle_cmr_direct.yaml`).

Noise model (`likelihood: noise`). No noise is added to the data; σ only
enters the likelihood, where it absorbs model error such as misalignment left
over after breathhold correction.

- `per_group`: σ_v² = σ_measured² + (s_g · extra_noise_std_v)². The scale s_g
  is fixed at 1 or learned once per view group (`sa`, `la`).
- `global`: σ_v² = σ_measured² + (r_v · σ)². One σ, in normalised units, is
  learned for every view. r_v is a fixed ratio per view (`relative_std`,
  e.g. `la_2ch: 4`; 1 if not listed), so per voxel a view's residuals count
  1 / r_v² as much. See `configs/kaggle_cmr.yaml`.

## Layout

```
src/mrisr_vae/
  geometry.py        latent grid (explicit object; replaces init_geometry's globals)
  forward_model.py   Lanczos operator copied from diffusion-priors, per-view builder, adjoint
  data.py            one frame of one subject per item; operators built in the workers
  backprojection.py  encoder input
  likelihood.py      Gaussian likelihood: per-group or global learned noise
  networks/          Ladder VAE and UNet trimmed from COSDD (GroupNorm instead of BatchRenorm)
  model.py           MRIVAE: ELBO, held-out-view error, posterior and prior samples
  trainer.py         Adamax with warm-up, early stopping, resumable checkpoints, wandb
  train.py           entry point mrisr-train (config options and defaults: DEFAULTS)
  predict.py         entry point mrisr-predict
configs/             geometry (copied from diffusion-priors) and training config
slurm/               training and prediction jobs
tests/               CPU unit tests, GPU memory probe, smoke run
```

## Setup

```
uv sync
```

## Running

Submit every job from the repository root. Each one writes `logs/<job-name>-<job-id>.out`. In order:

1. `sbatch tests/run-tests.sh` (CPU, minutes). Runs the unit tests on synthetic data, including a check that the forward model matches diffusion-priors'.
2. `sbatch tests/memory-probe.sh` (GPU). Runs full-size training steps on real subjects and reports peak memory, time per step and loading time per item. Use it to set `batch_size` and `n_filters` in `configs/kaggle_cmr.yaml`.
3. `sbatch tests/smoke-train.sh` (GPU). Trains 2 epochs on 8 subjects, resumes for a third, then predicts.
4. `sbatch slurm/train.sh configs/kaggle_cmr.yaml --resume`. Resubmit to carry on from `last.pt`. The supervised baseline trains the same way, from `configs/kaggle_cmr_supervised.yaml` (see below).
5. `sbatch slurm/predict.sh configs/kaggle_cmr.yaml`. The config's `predict` section picks the checkpoint (`best` or `last`), data, subjects, output directory and method (`direct`, `sampling` or `both`). Writes `sr_direct.nii.gz`, and/or `sr_mean.nii.gz` and `sr_std.nii.gz`, per subject; the supervised baseline writes `sr_supervised.nii.gz`.
6. `sbatch slurm/evaluate.sh configs/kaggle_cmr.yaml`, and likewise `kaggle_cmr_direct.yaml` and `kaggle_cmr_supervised.yaml`. The config's `evaluate` section sets the checkpoint, data, conditioning sets, frame, samples and seed. Scores predictions of the held-out view on the test set (see below).

## Held-out view

`data: heldout_views` (here `la_4ch`) lists views that training and validation
never load. They must not be in the geometry's `VIEWS`, and `mrisr-train`
refuses to start if they are. `la_2ch` stays in training, to show the model
handles mixed views.

`mrisr-evaluate` then conditions on chosen sets of training views. For
example, `--conditioning_views all sa la_2ch` evaluates short axis plus
`la_2ch`, short axis alone, and `la_2ch` alone, all on the same frame. Each set
is normalised from its own views alone, so nothing else leaks into the input.
For each set it projects the super-resolved volume through the held-out view's
forward model, and scores it in raw intensities on the voxels that forward
model supports, with MSE, PSNR and SSIM (definitions in `metrics.py`). It
writes `metrics.csv`.

The CMR configs set `evaluate: region: sa` (`--region sa`): only the
`la_4ch` voxels inside the subject's tallest short-axis stack are scored (the
highest index breaking ties), within its field of view and from half a slice
thickness below its first slice to half a thickness above its last. The latent
cube reaches above the ventricular base, but no training view images the
atria there (`la_2ch` crosses `la_4ch` along one line only), so the VAE has no
data for them, while the supervised baseline is trained on them. The region
depends only on the stack's geometry, so every model and conditioning set is
scored on the same voxels; `metrics.csv` gives their number (`n_voxels`) and
the stack (`region_stack`). With `--save_predictions` it also writes the
predictions, target and scoring mask on the view's own image grid, in
`<conditioning>/<subject>/`.

To score another method, e.g. a network trained to predict `la_4ch` directly,
save its predictions there as `la_4ch_pred_<name>.nii.gz` and run
`mrisr-score --evaluation_dir <dir> --name <name>`. It uses the same metric
code on the same voxels.

The three configs evaluate the same way, so their `metrics.csv` rows match on
(conditioning, subject, frame, view):

- the test set, which none of the models has seen (the supervised baseline
  was early-stopped on the validation subjects' `la_4ch`);
- one frame per subject, fixed by its name, so phases vary across subjects
  but not between models or runs;
- conditioning on `all`, `sa` and `la_2ch`;
- 32 posterior samples for the VAE, reseeded per subject and conditioning
  set from `seed`. The other two runs draw none: `kaggle_cmr_direct`'s VAE is
  `kaggle_cmr`'s, frozen, and the baseline has no posterior.
- `kaggle_cmr` sets `evaluate: direct: false` (`--no-direct`), so its scores
  are those of the samples' mean alone, the estimate the std describes; the
  direct predictor is scored by the `kaggle_cmr_direct` run.

With 2 or more samples, `mean_std` and `spearman_std_error` give the
uncertainty of the sample mean in the held-out view and its rank correlation
with the absolute error, per subject, over the scored voxels. The std is that
of the samples after the forward model, which by linearity is the posterior's
spread in the view; pushing the latent's per-voxel std through the forward
model would ignore the correlations between neighbouring voxels.
`--save_predictions` also writes it as `la_4ch_sample_std.nii.gz`.

## Supervised baseline

`model: type: supervised` (`configs/kaggle_cmr_supervised.yaml`) trains the
VAE's direct predictor alone, from scratch, as a supervised baseline: the same
UNet on the same encoder input, with the same view dropout. Its output, a
volume on the latent grid, goes through the forward model of `data:
target_view` (`la_4ch`), and the loss is the mean squared error against that
view on the voxels its forward model supports. The target is never in the
input and never sets its normalisation. Subjects without it are skipped
(18 of the 701 training subjects). Optimiser, warm-up, gradient accumulation,
clipping, learning-rate plateaus, early stopping (on `val/target_mse`, the
validation subjects' `la_4ch` error) and checkpoints are as for the VAE.

The baseline trains on `la_4ch`, so it is scored on the test set, which
neither model has seen: `sbatch slurm/evaluate.sh
configs/kaggle_cmr_supervised.yaml` writes its scores as
`*_supervised` in `metrics.csv`, on the same subjects, frames and voxels as the
VAE's.

## Validation metrics

- `val/direct_mse` and `val/direct_heldout_mse`: the direct predictor's error against posterior samples, and its held-out-view error.

- `val/nelbo`: negative ELBO in nats per observed voxel, with fixed frames and seed.
- `val/heldout_mse`: drops each subject's last training view in VIEWS order (`la_2ch`, not the held-out `la_4ch`), encodes the rest, and measures the error of predicting the dropped view. This is the same metric, on the same validation subjects and frames, as `em/heldout_error` in diffusion-priors, so the two models can be compared directly.
