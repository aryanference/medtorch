# MedTorch

MedTorch is a modern 3D medical imaging toolkit for deep learning workflows in CT and MRI.

It provides a complete pipeline for loading volumetric scans, applying robust augmentations, sampling training patches, and evaluating transform quality with practical metrics.

## Core capabilities

- 3D image IO and subject-based dataset abstractions
- patch-based sampling and queue-driven training support
- intensity and spatial augmentations for medical volumes
- CT-specific artifact simulation
- temporal augmentation primitives for 4D-style data
- quality metrics and transform auditing utilities

## New and revamped features

- **CT artifact augmentation**
  - `RandomBeamHardening`
  - `RandomMetalArtifact`
  - `RandomRingArtifact`
  - `RandomCTNoise`
- **Temporal augmentations**
  - `RandomTemporalShift`
  - `RandomBOLDNoise`
- **Inference reliability**
  - `GridAggregator` uncertainty estimation using `track_uncertainty=True`
  - per-voxel uncertainty retrieval via `get_uncertainty_map()`
- **Data pipeline utilities**
  - `SubjectsDataset.k_fold(...)`
  - `SubjectsDataset.train_val_split(...)`
  - `Queue.memory_usage_bytes`
- **Subject lifecycle helpers**
  - `Subject.to_dict(...)`
  - `Subject.from_dict(...)`
- **Benchmarking and analytics**
  - `Transform.benchmark(...)`
  - `metrics.TransformAudit`

## Installation

```bash
pip install -e .
```

## Quick usage

```python
import torchio as tio

transform = tio.Compose(
    [
        tio.RandomBeamHardening(strength=(0.05, 0.2), p=0.4),
        tio.RandomCTNoise(std=(0.0, 0.05), p=0.5),
        tio.RandomTemporalShift(shift=(-2, 2), p=0.3),
    ]
)
```

## Example workflow areas

- CT segmentation model robustness training
- low-dose CT augmentation experiments
- patch-based 3D UNet training pipelines
- transform profiling and preprocessing validation

## Project goals

- make advanced medical augmentation practical and reproducible
- provide a strong base for research and portfolio demonstrations
- keep APIs simple for rapid experimentation
