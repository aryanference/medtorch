from __future__ import annotations

import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomCTNoise(RandomTransform, IntensityTransform):
    r"""Simulate dose-dependent CT quantum noise.

    CT noise differs fundamentally from MRI noise. The dominant source is
    quantum noise: photon counting statistics at the detector follow a Poisson
    distribution, giving noise whose standard deviation scales inversely with
    the square root of dose. An additive Gaussian electronic noise floor
    is also included.

    Args:
        dose_level: Relative dose in ``(0, 1]``.  Lower values = noisier.
            Accepts a ``(min, max)`` tuple for random sampling.
        quantum_sigma_ref: Noise std (HU) at full dose. Default ``10`` HU.
        electronic_sigma: Electronic noise floor std (HU). Default ``2`` HU.
        **kwargs: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomCTNoise(dose_level=(0.1, 0.8), p=0.6)
    """

    def __init__(
        self,
        dose_level: float | tuple[float, float] = (0.2, 1.0),
        quantum_sigma_ref: float = 10.0,
        electronic_sigma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dose_range = self._parse_range(
            dose_level, 'dose_level', min_constraint=1e-3, max_constraint=1.0
        )
        self.quantum_sigma_ref = quantum_sigma_ref
        self.electronic_sigma = electronic_sigma

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject
        dose_by_name: dict[str, float] = {}
        for name in images_dict:
            dose_by_name[name] = self.sample_uniform(*self.dose_range)
        transform = CTNoise(
            dose=dose_by_name,
            quantum_sigma_ref=self.quantum_sigma_ref,
            electronic_sigma=self.electronic_sigma,
            **self._get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class CTNoise(IntensityTransform):
    """Apply deterministic dose-dependent CT quantum + electronic noise.

    Args:
        dose: Float or dict mapping image names to dose levels in ``(0, 1]``.
        quantum_sigma_ref: Noise std at full dose (HU).
        electronic_sigma: Electronic noise floor std (HU).
        **kwargs: See :class:`~torchio.transforms.Transform`.
    """

    def __init__(
        self,
        dose: float | dict[str, float],
        quantum_sigma_ref: float = 10.0,
        electronic_sigma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dose = dose
        self.quantum_sigma_ref = quantum_sigma_ref
        self.electronic_sigma = electronic_sigma
        self.args_names = ['dose', 'quantum_sigma_ref', 'electronic_sigma']

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            dose = self.get_parameter(self.dose, name)
            data = image.data.float()
            q_sigma = self.quantum_sigma_ref / (dose ** 0.5 + 1e-8)
            quantum_noise = torch.randn_like(data) * q_sigma
            electronic_noise = torch.randn_like(data) * self.electronic_sigma
            noisy = data + quantum_noise + electronic_noise
            image.set_data(noisy.to(image.data.dtype))
        return subject
