# MedTorch

[![PyPI downloads](https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white)](https://pypi.org/project/torchio/)
[![PyPI version](https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white)](https://pypi.org/project/torchio/)
[![Conda version](https://img.shields.io/conda/v/conda-forge/torchio.svg?label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/torchio)
[![Google Colab notebooks](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/medtorch/medtorch/blob/main/tutorials/README.md)
[![Documentation status](https://github.com/medtorch/medtorch/actions/workflows/docs.yml/badge.svg)](https://github.com/medtorch/medtorch/actions/workflows/docs.yml)
[![Tests status](https://github.com/medtorch/medtorch/actions/workflows/tests.yml/badge.svg)](https://github.com/medtorch/medtorch/actions/workflows/tests.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://docs.astral.sh/ruff/)
[![Coverage status](https://codecov.io/gh/medtorch/medtorch/branch/main/graphs/badge.svg)](https://app.codecov.io/github/medtorch/medtorch)
[![Code quality](https://img.shields.io/scrutinizer/g/medtorch/medtorch.svg?label=Code%20quality&logo=scrutinizer)](https://scrutinizer-ci.com/g/medtorch/medtorch/?branch=main)
[![YouTube](https://img.shields.io/youtube/views/UEUVSw5-M9M?label=watch&style=social)](https://www.youtube.com/watch?v=UEUVSw5-M9M)

MedTorch is an open-source Python library for efficient loading, preprocessing,
augmentation and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and
preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
[MRI magnetic field inhomogeneity (bias)](https://mriquestions.com/why-homogeneity.html)
or [k-space motion artifacts](http://proceedings.mlr.press/v102/shaw19a.html).

MedTorch is part of the broader [PyTorch Ecosystem](https://pytorch.org/ecosystem/),
and was featured at
the [PyTorch Ecosystem Day 2021](https://pytorch.org/blog/ecosystem_day_2021) and
the [PyTorch Developer Day 2021](https://pytorch.org/blog/pytorch-developer-day-2021).

Many groups have used this toolkit for their research.
The complete list of citations is available on [Google Scholar](https://scholar.google.co.uk/scholar?cites=8711392719159421861&sciodt=0,5&hl=en), and the
[dependents list](https://github.com/medtorch/medtorch/network/dependents) is
available on GitHub.

The code is available on [GitHub](https://github.com/medtorch/medtorch).
If you like MedTorch, star the repository.

See [Getting started](getting-started.md) for installation instructions and a
usage overview.

Contributions are welcome.
Please check the [contributing guide](https://github.com/medtorch/medtorch/blob/main/CONTRIBUTING.md)
if you would like to contribute.

If you have questions, feel free to ask in the
[discussions tab](https://github.com/medtorch/medtorch/discussions).

If you found a bug or have a feature request, please
[open an issue](https://github.com/medtorch/medtorch/issues).
