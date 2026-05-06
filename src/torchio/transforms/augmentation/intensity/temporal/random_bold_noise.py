from __future__ import annotations

import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomBOLDNoise(RandomTransform, IntensityTransform):
    r"""Simulate physiological noise in BOLD fMRI time-series.

    Functional MRI (fMRI) BOLD signals are contaminated by structured
    physiological noise from cardiac pulsation and respiratory motion.
    These appear as sinusoidal fluctuations at known frequency bands added
    on top of the neural signal of interest.

    This transform treats the **channel dimension** as the time axis
    (as is conventional when a 4D ``(1, X, Y, Z)`` NIfTI is stored with
    time in channels after loading). It adds:

    * **Cardiac noise** — sinusoidal at ~1 Hz (one cycle per ~60 TRs
      at TR = 1 s), spatially smooth.
    * **Respiratory noise** — sinusoidal at ~0.3 Hz, spatially smooth.
    * **Thermal noise** — i.i.d. Gaussian.

    Args:
        cardiac_amplitude: Max amplitude of cardiac noise as a fraction of
            the signal std. Accepts a ``(min, max)`` tuple.
        respiratory_amplitude: Max amplitude of respiratory noise as a
            fraction of the signal std. Accepts a ``(min, max)`` tuple.
        thermal_sigma: Std of additive Gaussian thermal noise (same units
            as the image). Accepts a ``(min, max)`` tuple.
        cardiac_freq: Approximate cardiac frequency in cycles/timepoint.
            Default ``0.016`` (≈ 1 Hz at TR = 1 s).
        respiratory_freq: Approximate respiratory frequency in
            cycles/timepoint. Default ``0.005`` (≈ 0.3 Hz at TR = 1 s).
        **kwargs: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> # 4-D fMRI: shape (T, X, Y, Z) after loading
        >>> transform = tio.RandomBOLDNoise(cardiac_amplitude=0.05, p=0.5)
    """

    def __init__(
        self,
        cardiac_amplitude: float | tuple[float, float] = (0.01, 0.08),
        respiratory_amplitude: float | tuple[float, float] = (0.01, 0.06),
        thermal_sigma: float | tuple[float, float] = (0.0, 5.0),
        cardiac_freq: float = 0.016,
        respiratory_freq: float = 0.005,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cardiac_amp_range = self._parse_range(
            cardiac_amplitude, 'cardiac_amplitude', min_constraint=0.0
        )
        self.resp_amp_range = self._parse_range(
            respiratory_amplitude, 'respiratory_amplitude', min_constraint=0.0
        )
        self.thermal_sigma_range = self._parse_range(
            thermal_sigma, 'thermal_sigma', min_constraint=0.0
        )
        self.cardiac_freq = cardiac_freq
        self.respiratory_freq = respiratory_freq

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            data = image.data.float()  # (T, X, Y, Z)
            n_timepoints = data.shape[0]
            if n_timepoints < 4:
                continue  # not a time-series
            sig_std = float(data.std()) + 1e-8
            c_amp = self.sample_uniform(*self.cardiac_amp_range) * sig_std
            r_amp = self.sample_uniform(*self.resp_amp_range) * sig_std
            t_sigma = self.sample_uniform(*self.thermal_sigma_range)
            t = torch.arange(n_timepoints, dtype=torch.float32)
            # Random phase offsets for biological realism
            c_phase = torch.rand(1).item() * 6.2832
            r_phase = torch.rand(1).item() * 6.2832
            cardiac = c_amp * torch.sin(
                2 * 3.14159 * self.cardiac_freq * t + c_phase
            )
            respiratory = r_amp * torch.sin(
                2 * 3.14159 * self.respiratory_freq * t + r_phase
            )
            # Shape: (T,) -> broadcast over (T, X, Y, Z)
            phys_noise = (cardiac + respiratory).view(n_timepoints, 1, 1, 1)
            thermal = torch.randn_like(data) * t_sigma
            noisy = data + phys_noise + thermal
            image.set_data(noisy.to(image.data.dtype))
        return subject
