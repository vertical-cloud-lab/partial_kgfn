# Changelog

## Unreleased

### Changed
- Made the BOFN Actions harness pass `noisy=True` explicitly to mirror the AckleyS paper runner.
- Reduced the `full-smoke` workflow gate to a single pKGFN iteration via the new `PKGFN_MAX_ITERATIONS` env-var cap, dropping the redundant Random cell.

### Added
- Added a BOFN benchmark GitHub Actions workflow with smoke, matrix, and combine jobs.
- Added an action-friendly BOFN/pKGFN benchmark harness and pip requirements for CI installation.
- Added ignore rules for generated benchmark results and figures.
- Added `PKGFN_MAX_ITERATIONS` env-var cap on the BO loop in `partial_kgfn/run_one_trial.py` for CI smoke-gating.

### Removed
- Removed the unused `model_trial_<seed>.pt` weights checkpoint from `run_one_trial`. The data-only `trial_<seed>.pt` is sufficient — the resume path refits the GP from `train_X`/`train_Y` and nothing else read the model file.
