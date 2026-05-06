<div align="center">

```
███╗   ███╗███████╗██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
████╗ ████║██╔════╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
██╔████╔██║█████╗  ██║  ██║   ██║   ██║   ██║██████╔╝██║     ███████║
██║╚██╔╝██║██╔══╝  ██║  ██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
██║ ╚═╝ ██║███████╗██████╔╝   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
╚═╝     ╚═╝╚══════╝╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**3D Medical Imaging · Deep Learning · CT & MRI Pipelines**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-D22128?style=for-the-badge)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-22c55e?style=for-the-badge)]()
[![Format](https://img.shields.io/badge/Code%20Style-Ruff-F7C948?style=for-the-badge)]()

*A modern, research-grade toolkit for volumetric medical image augmentation, patch-based training, and transform auditing — built for practitioners who work with real scanners.*

</div>

---

## ⚡ What is MedTorch?

MedTorch is a **complete 3D medical imaging pipeline** for deep learning workflows, covering everything from raw DICOM/NIfTI loading to production-ready training loops. It specializes in:

- Simulating real-world **CT scanner artifacts** (beam hardening, metal, ring noise)
- **4D temporal augmentations** for dynamic imaging sequences
- **Uncertainty-aware inference** with patch-based aggregation
- **Transform auditing & benchmarking** for reproducible research

> Built on top of `torchio`, MedTorch extends the ecosystem with clinical realism and research-grade tooling.

---

## 🧠 Architecture at a Glance

```
                    ┌─────────────────────────────────────────┐
                    │              MedTorch Pipeline           │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────┬───────────────┼───────────────┬──────────────┐
          ▼             ▼               ▼               ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐
    │  IO &    │  │  Patch   │  │  CT Artifact│  │ Temporal │  │  Metrics & │
    │ Subjects │  │ Sampling │  │ Simulation  │  │   Augs   │  │  Auditing  │
    └──────────┘  └──────────┘  └────────────┘  └──────────┘  └────────────┘
         │              │               │               │              │
    Loads NIfTI   Queue-driven    Beam Hardening   BOLD Noise    TransformAudit
    DICOM, MHD    training loops  Metal Artifacts  Temporal      Benchmarking
    Subject API   k-fold splits   Ring Artifacts   Shifts        Per-voxel QA
```

---

## 🔬 Feature Highlights

### CT Artifact Simulation

Real scanners produce real noise. Train models that don't break in production.

| Augmentation | What It Simulates | Clinical Use Case |
|---|---|---|
| `RandomBeamHardening` | X-ray beam energy absorption | Dense bone / contrast agent robustness |
| `RandomMetalArtifact` | Metallic implant streak artifacts | Surgical implant imaging |
| `RandomRingArtifact` | Detector ring failures | Low-dose CT generalization |
| `RandomCTNoise` | Quantum & electronic noise | Dose-invariant model training |

### Temporal Augmentations (4D)

```python
import torchio as tio

transform = tio.Compose([
    tio.RandomTemporalShift(shift=(-2, 2), p=0.3),   # cardiac / respiratory motion
    tio.RandomBOLDNoise(p=0.5),                        # fMRI BOLD signal perturbation
])
```

### Uncertainty-Aware GridAggregator

```python
aggregator = tio.data.GridAggregator(sampler, track_uncertainty=True)

for patches_batch in patch_loader:
    predictions = model(patches_batch['mri'][tio.DATA])
    aggregator.add_batch(predictions, patches_batch[tio.LOCATION])

output_tensor  = aggregator.get_output_tensor()
uncertainty_map = aggregator.get_uncertainty_map()   # per-voxel epistemic uncertainty
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repo
git clone https://github.com/aryanference/medtorch.git
cd medtorch

# Install in editable mode
pip install -e .
```

### Minimal Example

```python
import torchio as tio

# Build a clinically-realistic augmentation pipeline
transform = tio.Compose([
    tio.RandomBeamHardening(strength=(0.05, 0.2), p=0.4),
    tio.RandomCTNoise(std=(0.0, 0.05), p=0.5),
    tio.RandomTemporalShift(shift=(-2, 2), p=0.3),
])

# Load a subject and apply
subject = tio.Subject(
    ct=tio.ScalarImage('scan.nii.gz'),
    label=tio.LabelMap('seg.nii.gz'),
)
augmented = transform(subject)
```

### Dataset Splits

```python
dataset = tio.SubjectsDataset(subjects)

# K-Fold cross-validation
folds = dataset.k_fold(k=5)

# Train/Val split
train_set, val_set = dataset.train_val_split(val_ratio=0.2)
```

---

## 📊 Benchmarking & Auditing

Stop guessing about your preprocessing. Measure it.

```python
# Benchmark a transform
stats = tio.RandomBeamHardening().benchmark(subjects, n_runs=50)
# → mean time, std dev, memory usage

# Audit transform quality across a dataset
audit = tio.metrics.TransformAudit(transform, subjects)
report = audit.run()
# → per-transform quality scores, outlier detection, intensity drift
```

---

## 🗂️ Project Structure

```
medtorch/
├── src/torchio/           # Core library — augmentations, IO, sampling
│   ├── transforms/        # CT artifacts, temporal, spatial, intensity
│   ├── data/              # Subject, SubjectsDataset, Queue, Aggregators
│   └── metrics/           # TransformAudit, quality utilities
├── tests/                 # Pytest test suite
├── tutorials/             # Jupyter notebooks & workflow examples
├── docs/                  # Sphinx documentation source
└── pyproject.toml         # Build config & dependencies
```

---

## 🎯 Use Cases

```
✦ CT segmentation robustness training
✦ Low-dose CT augmentation experiments
✦ Patch-based 3D U-Net training pipelines
✦ fMRI / 4D sequence preprocessing
✦ Transform profiling & preprocessing validation
✦ Research portfolio demonstrations
```

---

## 🧪 Running Tests

```bash
# Using tox
tox

# Or directly with pytest
pytest tests/ -v
```

---

## 📖 Documentation

Full API reference and tutorials live in [`/docs`](./docs) and [`/tutorials`](./tutorials).

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before submitting a PR. This project follows a standard fork → branch → pull request workflow.

---

## 📄 License

Licensed under the **Apache 2.0 License** — see [`LICENSE`](./LICENSE) for details.

---

## 🔖 Citation

If you use MedTorch in your research, please cite:

```bibtex
@software{medtorch,
  author  = {aryanference},
  title   = {MedTorch: 3D Medical Imaging Toolkit for Deep Learning},
  url     = {https://github.com/aryanference/medtorch},
  year    = {2024}
}
```

---

<div align="center">

*Built for researchers who treat their data pipelines as seriously as their models.*

**[⭐ Star this repo](https://github.com/aryanference/medtorch)** · **[🐛 Report a Bug](https://github.com/aryanference/medtorch/issues)** · **[💡 Request a Feature](https://github.com/aryanference/medtorch/issues)**

</div>
