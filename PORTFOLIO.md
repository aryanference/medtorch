# MedTorch Portfolio Notes

## Project identity

- **Name:** MedTorch
- **Base:** MedTorch architecture and APIs
- **Focus:** CT-aware augmentation, transform quality analytics, and practical research tooling

## Implemented extension areas

- CT augmentation modules under `src/torchio/transforms/augmentation/intensity/ct`
- Temporal augmentation modules under `src/torchio/transforms/augmentation/intensity/temporal`
- Metrics package under `src/torchio/metrics`
- Multi-subject transform wrapper in `src/torchio/data/multi_subject_transform.py`

## Recent revamp work

- Rebranded package metadata in `pyproject.toml` to `medtorch`
- Replaced README with a portfolio-oriented narrative and feature highlights
- Improved `GridAggregator` uncertainty tracking and removed duplicated output logic
- Added `Queue.memory_usage_bytes` for easier runtime memory estimation

## Suggested next milestones

- Add docs/examples that compare CT artifact transforms qualitatively
- Add unit tests for `get_uncertainty_map()` across `crop`, `average`, and `hann`
- Add benchmark scripts demonstrating `Transform.benchmark(...)` and `TransformAudit`
- Create a small demo notebook showing end-to-end CT augmentation + metric evaluation
