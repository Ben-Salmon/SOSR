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
  preprocess/        before training
    breathhold/      mrisr-breathhold: slice alignment (Stolt-Ansó et al., see its README)
    intensity_gain.py mrisr-intensity-gain: long-axis intensity gains
    registration.py  mrisr-register: registration_params.pickle per subject
    heart_roi.py     the heart's bounding box (Zambal et al. 2008), for the centre of mass
    latent_extent.py mrisr-latent-extent: how big to make the latent grid
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
  metrics.py         masked mse, psnr, ssim and Spearman, used by every score
  evaluation/        after training: one protocol for every method
    protocol.py      the protocol yaml (configs/evaluation/)
    heldout.py       mrisr-evaluate: a model's predictions of the held-out view
    baselines.py     mrisr-baselines: interpolation and SMORE predictions
    score.py         mrisr-score: scores.csv and summary.csv over every method
    conditioning.py  conditioning sets and the scored region
    files.py         the evaluation directory's layout
configs/             training configs, latent geometries, evaluation/ protocols
slurm/               one job per step below
tests/               CPU unit tests, GPU memory probe, smoke run
```

## Setup

```
uv sync
```

## Running

Submit every job from the repository root. Each one writes
`logs/<job-name>-<job-id>.out`. Nothing heavy runs on the login node.

### 1. Preprocess

1. **Breathhold correction** (CMR only). Aligns every short- and long-axis
   slice of a subject by matching intensities where they intersect; writes
   `<subject>/<view>_affine_correction.npy`. Once per directory:
   `sbatch slurm/breathhold.sh --data_dir ../scratch/data/kaggle_niftis/train` (and `test`).
2. **Intensity gains** (CMR only). Each series is scaled on its own, so a
   long-axis view can be ~10-15% darker than the short-axis slices at the
   same anatomy, while the forward model slices one volume into both.
   Estimates each long-axis view's gain relative to the short-axis slices
   where they intersect, with the breathhold correction applied; writes
   `<subject>/<view>_intensity_gain.json`, which the dataset multiplies the
   view and its noise level by. After step 1, once per directory:
   `sbatch slurm/intensity-gain.sh --data_dir ../scratch/data/kaggle_niftis/train` (and `test`).
   Rerun it whenever the breathhold correction is rerun.
3. **Registration.** Writes `<subject>/registration_params.pickle`: each
   view's voxel-to-mm affine (with the correction), voxel size, slice
   thickness and size, and `mm_to_su_affine`, which puts the ROI's centre of
   mass (the heart's bounding box in the first CMR view; the bright voxels
   of a brain) at the origin of the latent's frame.
   `sbatch slurm/register.sh --dataset cmr --data_dirs <train> <test>`
   (`--dataset brain` for fastMRI). Rerun it whenever the breathhold
   correction is rerun: the dataset refuses a pickle built with other
   corrections than those on disk. `--check` writes nothing and lists the
   subjects whose parameters would change.
4. **Latent grid size.** `sbatch slurm/latent-extent.sh --dataset cmr --data_dir <train> --su_per_lvox 1.8`
   prints the ROI's extent over the training subjects (10th percentile of
   the lows, 90th of the highs), the grid it suggests (`x_size_su`,
   `X_MIN_COORDS_SU`) and the size the Ladder VAE pads it to. Copy it,
   rounded as you like, into the geometry yaml (`configs/kaggle_cmr_geometry.yaml`).

### 2. Train

1. `sbatch tests/run-tests.sh` (CPU, minutes). Runs the unit tests on synthetic data, including a check that the forward model matches diffusion-priors', and that the preprocessing matches diffusion-priors' original scripts.
2. `sbatch tests/memory-probe.sh` (GPU). Runs full-size training steps on real subjects and reports peak memory, time per step and loading time per item. Use it to set `batch_size` and `n_filters` in `configs/kaggle_cmr.yaml`.
3. `sbatch tests/smoke-train.sh` (GPU). Trains 2 epochs on 8 subjects, resumes for a third, then predicts.
4. `sbatch slurm/train.sh configs/kaggle_cmr.yaml --resume`. Resubmit to carry on from `last.pt`. Likewise `configs/kaggle_cmr_direct.yaml` (the direct predictor retrained on the frozen VAE) and `configs/kaggle_cmr_supervised.yaml` (the supervised baseline, below).

### 3. Predict (optional)

`sbatch slurm/predict.sh configs/kaggle_cmr.yaml` writes super-resolved
volumes on the latent grid. The config's `predict` section picks the
checkpoint (`best` or `last`), data, subjects, output directory and method
(`direct`, `sampling` or `both`). Writes `sr_direct.nii.gz`, and/or
`sr_mean.nii.gz` and `sr_std.nii.gz`, per subject; the supervised baseline
writes `sr_supervised.nii.gz`. Evaluation does not need it.

### 4. Evaluate

Every method is scored by one protocol,
`configs/evaluation/kaggle_cmr_test.yaml`, into one directory (its
`out_dir`), and compared in one table. In order (`p=configs/evaluation/kaggle_cmr_test.yaml`):

1. The models, one job per method of the protocol (they can run at once):
   `sbatch slurm/evaluate.sh $p vae`, `... $p vae_direct`, `... $p supervised`.
2. The baselines: `sbatch slurm/baselines.sh export $p`, then the SMORE array
   it prints (`sbatch --array=1-N%20 slurm/smore.sh <work_dir>/stacks.txt`;
   install SMORE once first, see the top of `slurm/smore.sh`), then
   `sbatch slurm/baselines.sh check $p --max_subjects 2` (SMORE's grid is
   read right: the unshifted output should score best) and
   `sbatch slurm/baselines.sh project $p`.
3. `sbatch slurm/score.sh $p` writes `scores.csv` (one row per method,
   conditioning set, subject and view) and `summary.csv` (per conditioning
   set and method, mean, std and median over the subjects every method in
   that set has) in `out_dir`, and prints the table.

Steps 1 and 2 can run in either order, and a method can be added or rerun
later: rerun step 3 after.

**What the protocol fixes.** The held-out view (`heldout_views`, here
`la_4ch`) is one training and validation never load: it must not be in the
geometry's `VIEWS`, and `mrisr-train` refuses to start if it is. `la_2ch`
stays in training, to show the model handles mixed views. Then, for every
method alike:

- the test set (`data_dir`), which none of the models has seen (the
  supervised baseline was early-stopped on the validation subjects' `la_4ch`);
- one frame per subject, fixed by its name, so phases vary across subjects
  but not between methods or runs;
- the conditioning sets: `all` (every training view), `sa` (every short-axis
  stack), `sa_first` (each subject's lowest-numbered short-axis stack, the
  one the baselines are given) and `la_2ch`. Each set is normalised from its
  own views alone, so nothing else leaks into the input;
- the scored voxels: those of `la_4ch` its forward model supports, inside
  each subject's lowest-numbered short-axis stack (`region: sa_first`):
  within its field of view, and from half a slice thickness below its first
  slice to half a thickness above its last. The latent cube reaches above
  the ventricular base, but no training view images the atria there
  (`la_2ch` crosses `la_4ch` along one line only), so the VAE has no data
  for them, while the supervised baseline is trained on them. The region
  depends only on the subject's stacks, so it is the same for every method
  and conditioning set.

Each model method gives one prediction: with `n_samples` (the VAE's 32), the
mean of that many posterior samples, reseeded per subject and conditioning
set from `seed`; without, the model's direct predictor (or the supervised
baseline's output). For each, the super-resolved volume is pushed through
`la_4ch`'s forward model and saved on `la_4ch`'s own image grid as
`<conditioning>/<subject>/la_4ch_pred_<method>.nii.gz`, beside the target and
scoring mask, which every method checks are the ones the others used.
`mrisr-score` then scores them all by the same code, in raw intensities: MSE,
PSNR and SSIM (definitions in `metrics.py`). To score another method, save
its predictions there as `la_4ch_pred_<name>.nii.gz` and rerun it.

With 2 or more samples, the samples' spread after the forward model is saved
as `la_4ch_std_<method>.nii.gz`, which by linearity is the posterior's spread
in the view (pushing the latent's per-voxel std through the forward model
would ignore the correlations between neighbouring voxels). `mrisr-score`
reports its mean over the scored voxels (`mean_std`) and its rank
correlation with the absolute error (`spearman_std_error`), per subject.
`runs/<method>.csv` records what each run did per subject (frame,
conditioning views, normalisation, and the average score of single samples).

The baselines (`mrisr-baselines`) are linear and cubic interpolation of each
subject's lowest-numbered short-axis stack, and SMORE (Remedios et al. 2023)
run unmodified on it, resampled onto the latent grid and pushed through the
same forward model; they appear under `sa_first`.

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

The baseline trains on `la_4ch`, so it is scored on the test set only,
which neither model has seen: method `supervised` of the evaluation protocol,
on the same subjects, frames and voxels as the VAE.

## Validation metrics

- `val/direct_mse` and `val/direct_heldout_mse`: the direct predictor's error against posterior samples, and its held-out-view error.

- `val/nelbo`: negative ELBO in nats per observed voxel, with fixed frames and seed.
- `val/heldout_mse`: drops each subject's last training view in VIEWS order (`la_2ch`, not the held-out `la_4ch`), encodes the rest, and measures the error of predicting the dropped view. This is the same metric, on the same validation subjects and frames, as `em/heldout_error` in diffusion-priors, so the two models can be compared directly.
