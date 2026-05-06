from __future__ import annotations

import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomMetalArtifact(RandomTransform, IntensityTransform):
    r"""Simulate CT metal streak artifacts.

    High-density metallic implants (surgical screws, dental fillings, hip
    prostheses) cause severe streaking in CT reconstructions. The streaks
    radiate outward from the metal region in the axial plane due to photon
    starvation and beam hardening along projection rays passing through metal.

    Simulation steps per axial slice:

    1. Threshold the image to find a metal-like seed region.
    2. Generate radial streak lines passing through that region.
    3. Add high-intensity streaks that decay with distance.

    Args:
        metal_threshold: Percentile of the image intensity distribution
            used to define the "metal" seed region. Default ``99.0`` means
            the brightest 1 % of voxels act as metal.
        num_streaks: Number of radial streaks per slice. If a tuple
            ``(min, max)`` is given, the count is drawn uniformly.
        streak_intensity: Amplitude of the streak signal (in the same
            units as the image, e.g. HU). Accepts a ``(min, max)`` tuple.
        **kwargs: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomMetalArtifact(p=0.3)
        >>> transformed = transform(subject)
    """

    def __init__(
        self,
        metal_threshold: float = 99.0,
        num_streaks: int | tuple[int, int] = 12,
        streak_intensity: float | tuple[float, float] = 200.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 < metal_threshold < 100:
            raise ValueError(f'metal_threshold must be in (0,100), got {metal_threshold}')
        self.metal_threshold = metal_threshold
        self.num_streaks_range = self._parse_range(
            num_streaks, 'num_streaks', min_constraint=1
        )
        self.streak_intensity_range = self._parse_range(
            streak_intensity, 'streak_intensity', min_constraint=0.0
        )

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        params_by_name: dict[str, dict] = {}
        for name in images_dict:
            params_by_name[name] = {
                'metal_threshold': self.metal_threshold,
                'num_streaks': int(self.sample_uniform(*self.num_streaks_range)),
                'streak_intensity': self.sample_uniform(*self.streak_intensity_range),
            }

        transform = MetalArtifact(params=params_by_name, **self._get_base_args())
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class MetalArtifact(IntensityTransform):
    """Apply deterministic CT metal streak artifact.

    Args:
        params: Dict mapping image names to parameter dicts.
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
            data = image.data.float().clone()
            data = self._add_streaks(
                data,
                p['metal_threshold'],
                p['num_streaks'],
                p['streak_intensity'],
            )
            image.set_data(data.to(image.data.dtype))
        return subject

    @staticmethod
    def _add_streaks(
        data: torch.Tensor,
        metal_threshold: float,
        num_streaks: int,
        streak_intensity: float,
    ) -> torch.Tensor:
        _, w, h, d = data.shape
        # Work slice by slice along the depth axis
        for k in range(d):
            slc = data[0, :, :, k]
            thresh = torch.quantile(slc, metal_threshold / 100.0)
            metal_mask = slc >= thresh
            if not metal_mask.any():
                continue
            # Find centroid of metal region
            idxs = metal_mask.nonzero(as_tuple=False).float()
            cx = idxs[:, 0].mean().item()
            cy = idxs[:, 1].mean().item()
            # Draw radial streaks
            angles = torch.linspace(0, 3.14159, num_streaks)
            for angle in angles:
                cos_a = angle.cos().item()
                sin_a = angle.sin().item()
                max_r = int((w ** 2 + h ** 2) ** 0.5)
                for r in range(-max_r, max_r):
                    xi = int(cx + r * cos_a)
                    yi = int(cy + r * sin_a)
                    if 0 <= xi < w and 0 <= yi < h:
                        decay = 1.0 / (abs(r) + 1) ** 0.5
                        data[0, xi, yi, k] += streak_intensity * decay
        return data
