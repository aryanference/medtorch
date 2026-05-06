from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomBeamHardening(RandomTransform, IntensityTransform):
    r"""Simulate CT beam-hardening (cupping) artifact.

    In X-ray CT, lower-energy photons are preferentially absorbed as the beam
    passes through dense tissue, causing the beam to "harden". This produces a
    characteristic cupping artifact: voxels near the centre of a uniform object
    appear darker than those at the periphery.

    The artifact is modelled as a spatially-varying multiplicative bias field
    :math:`B(\mathbf{r})`:

    .. math::
        I'(\mathbf{r}) = I(\mathbf{r}) \cdot B(\mathbf{r}), \quad
        B(\mathbf{r}) = 1 - s \cdot \left(\frac{d(\mathbf{r})}{d_{\max}}
        \right)^{\gamma}

    where :math:`d(\mathbf{r})` is the Euclidean distance from voxel
    :math:`\mathbf{r}` to the volume centre, :math:`d_{\max}` is the maximum
    such distance, :math:`s` is the cupping strength, and :math:`\gamma`
    controls the spatial fall-off.

    Args:
        strength: Scalar or ``(min, max)`` tuple controlling the maximum
            fractional signal drop at the volume centre.
            If a single value ``s`` is given it is interpreted as
            ``(0, s)``. Typical values are in ``[0.05, 0.3]``.
        gamma: Exponent controlling the spatial profile of the artifact.
            Higher values confine the cupping closer to the centre.
            Default is ``2.0`` (parabolic profile).
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomBeamHardening(strength=0.15, p=0.5)
        >>> transformed = transform(subject)
    """

    def __init__(
        self,
        strength: float | tuple[float, float] = 0.15,
        gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strength_range = self._parse_range(strength, 'strength', min_constraint=0.0)
        if gamma <= 0:
            raise ValueError(f'gamma must be positive, not {gamma}')
        self.gamma = gamma

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        strength_by_name: dict[str, float] = {}
        for name in images_dict:
            strength_by_name[name] = self.sample_uniform(*self.strength_range)

        transform = BeamHardening(
            strength=strength_by_name,
            gamma=self.gamma,
            **self._get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class BeamHardening(IntensityTransform):
    """Apply a deterministic CT beam-hardening (cupping) artifact.

    Args:
        strength: A ``float`` or a ``dict`` mapping image names to ``float``
            values specifying the cupping strength per image.
        gamma: Spatial exponent of the cupping profile.
        **kwargs: See :class:`~torchio.transforms.Transform`.
    """

    def __init__(
        self,
        strength: float | dict[str, float],
        gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strength = strength
        self.gamma = gamma
        self.args_names = ['strength', 'gamma']

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            strength = self.get_parameter(self.strength, name)
            bias = self._make_cupping_field(image.data, strength, self.gamma)
            image.set_data(image.data * bias)
        return subject

    @staticmethod
    def _make_cupping_field(
        data: torch.Tensor,
        strength: float,
        gamma: float,
    ) -> torch.Tensor:
        """Build the spatially-varying cupping bias field tensor."""
        _, w, h, d = data.shape
        # Normalised coordinate grids in [-1, 1]
        cx, cy, cz = w / 2.0, h / 2.0, d / 2.0
        xs = (torch.arange(w, dtype=torch.float32) - cx) / cx
        ys = (torch.arange(h, dtype=torch.float32) - cy) / cy
        zs = (torch.arange(d, dtype=torch.float32) - cz) / cz
        # Euclidean distance from centre, normalised to [0, 1]
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2)
        dist = dist / (dist.max() + 1e-8)
        # Cupping bias: darkest at centre, no change at periphery
        bias = 1.0 - strength * (1.0 - dist ** gamma)
        bias = bias.clamp(min=0.01).unsqueeze(0)  # (1, W, H, D)
        return bias
