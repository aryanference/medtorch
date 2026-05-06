from __future__ import annotations

import numpy as np
import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomRingArtifact(RandomTransform, IntensityTransform):
    r"""Simulate CT ring artifacts caused by miscalibrated detector elements.

    Ring artifacts appear as concentric bright or dark circles centred on the
    rotation isocentre. They arise when one or more detector cells have a
    different gain from the rest, which — after filtered back-projection —
    produces a ring at the corresponding radial distance.

    The simulation adds thin annular bands of perturbed intensity to each
    axial slice.  Each ring is placed at a randomly sampled radius and has a
    randomly sampled polarity (bright or dark) and magnitude.

    Args:
        num_rings: Number of rings to add. If a ``(min, max)`` tuple is given,
            the count is drawn uniformly from ``[min, max]``.
        intensity: Maximum absolute intensity perturbation added to ring
            voxels.  If a ``(min, max)`` tuple is given, the magnitude is
            drawn uniformly per ring.
        ring_width: Width of each ring in voxels.  A value of ``1`` produces
            a very fine ring; larger values produce broader banding.
        **kwargs: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomRingArtifact(num_rings=3, intensity=50.0)
        >>> transformed = transform(subject)
    """

    def __init__(
        self,
        num_rings: int | tuple[int, int] = 3,
        intensity: float | tuple[float, float] = 50.0,
        ring_width: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_rings_range = self._parse_range(
            num_rings, 'num_rings', min_constraint=1, type_constraint=int
        ) if isinstance(num_rings, int) else self._parse_range(
            num_rings, 'num_rings', min_constraint=1
        )
        self.intensity_range = self._parse_range(intensity, 'intensity', min_constraint=0.0)
        if not isinstance(ring_width, int) or ring_width < 1:
            raise ValueError(f'ring_width must be a positive int, not {ring_width}')
        self.ring_width = ring_width

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        params_by_name: dict[str, dict] = {}
        for name, image in images_dict.items():
            num_rings = int(self.sample_uniform(*self.num_rings_range))
            max_radius = min(image.data.shape[1], image.data.shape[2]) // 2
            radii = [
                int(torch.randint(2, max_radius, (1,)).item())
                for _ in range(num_rings)
            ]
            intensities = [
                self.sample_uniform(*self.intensity_range)
                * (1 if torch.rand(1).item() > 0.5 else -1)
                for _ in range(num_rings)
            ]
            params_by_name[name] = {
                'radii': radii,
                'intensities': intensities,
                'ring_width': self.ring_width,
            }

        transform = RingArtifact(params=params_by_name, **self._get_base_args())
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class RingArtifact(IntensityTransform):
    """Apply a deterministic CT ring artifact.

    Args:
        params: Dict mapping image names to ring parameters (radii,
            intensities, ring_width).
        **kwargs: See :class:`~torchio.transforms.Transform`.
    """

    def __init__(self, params: dict[str, dict], **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.args_names = ['params']

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            if name not in self.params:
                continue
            p = self.params[name]
            data = image.data.clone().float()
            mask = self._make_ring_mask(
                data.shape,
                p['radii'],
                p['ring_width'],
            )
            for i, (ring_mask, inten) in enumerate(
                zip(mask, p['intensities'], strict=True)
            ):
                data += inten * ring_mask.float()
            image.set_data(data.to(image.data.dtype))
        return subject

    @staticmethod
    def _make_ring_mask(
        shape: tuple,
        radii: list[int],
        ring_width: int,
    ) -> list[torch.Tensor]:
        """Return one boolean mask per ring for all slices simultaneously."""
        _, w, h, d = shape
        cx, cy = w / 2.0, h / 2.0
        xs = torch.arange(w, dtype=torch.float32) - cx + 0.5
        ys = torch.arange(h, dtype=torch.float32) - cy + 0.5
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        r = torch.sqrt(grid_x ** 2 + grid_y ** 2)  # (W, H)

        masks = []
        for radius in radii:
            annulus = (r >= radius - ring_width) & (r <= radius + ring_width)
            # Broadcast across channels and depth: (1, W, H, 1) -> (1, W, H, D)
            annulus = annulus.unsqueeze(0).unsqueeze(-1).expand(1, w, h, d)
            masks.append(annulus)
        return masks
