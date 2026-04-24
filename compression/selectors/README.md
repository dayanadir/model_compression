# Selector Baselines for Model Compression

This folder contains baseline **channel/unit selection strategies** used during structured model compression.  
Each selector chooses which teacher indices to keep for an `AxisGroup` when shrinking width from `H` to `Hs`.

## Available Baselines

- `uniform`
  - Picks `Hs` indices spaced uniformly across `[0, H)`.
  - Deterministic and data-free.
  - Useful as a simple geometry-only baseline.

- `random_consistent`
  - Samples a random subset of size `Hs` from `[0, H)` using the provided RNG seed.
  - Deterministic for a fixed seed, but otherwise random.
  - Useful as a stochastic baseline.

- `magnitude`
  - Computes per-unit score as the **sum of absolute weights** across group members (`in`/`out`, excluding `affine`).
  - Keeps top-`Hs` units by score.
  - Data-free, weight-only importance baseline.

- `l1_structured`
  - Uses one representative outgoing multi-dimensional tensor and computes structured L1 importance per unit.
  - Reduces all non-selected axes by sum of absolute values.
  - Keeps top-`Hs` units by this structured score.

- `in_out_meanabs`
  - Computes per-unit mean absolute weight for incoming members plus outgoing members:
    - score = `mean_abs(in)` + `mean_abs(out)`
  - Optional variant includes affine gamma:
    - score += `mean_abs(affine)`
  - Keeps top-`Hs` units by score.

- `activation`
  - Uses precomputed calibration activation scores per `AxisGroup`.
  - Requires `CalibrationContext` and fails if group scores are missing.
  - Keeps top-`Hs` units by activation-derived importance.

## Determinism and Tie-Breaking

Selectors that rank units (`magnitude`, `l1_structured`, `in_out_meanabs`, `activation`) use stable top-k selection with deterministic lower-index tie-breaking.

## Calibration Requirement Summary

- Requires calibration: `activation`
- No calibration required: `uniform`, `random_consistent`, `magnitude`, `l1_structured`, `in_out_meanabs`
