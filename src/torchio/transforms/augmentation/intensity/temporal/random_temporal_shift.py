from __future__ import annotations

import torch

from .....data.subject import Subject
from ....intensity_transform import IntensityTransform
from ... import RandomTransform


class RandomTemporalShift(RandomTransform, IntensityTransform):
    r"""Apply a random sub-TR temporal shift to fMRI/4D image channels.

    In multi-band or interleaved fMRI acquisitions, different brain slices
    are acquired at slightly different times within the repetition time (TR).
    If slice-timing correction is not applied (or is imperfect), each slice
    experiences a temporal offset. This transform simulates that effect by
    circularly shifting voxel time-series by a random integer number of
    timepoints, independently per spatial slice along the depth axis.

    The **channel dimension** (first axis of the ``(C, W, H, D)`` tensor)
    is treated as the time axis.

    Args:
        max_shift: Maximum absolute shift in timepoints. Each depth slice
            is shifted by an integer drawn uniformly from
            ``[-max_shift, max_shift]``. Must be a positive integer.
        **kwargs: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomTemporalShift(max_shift=3, p=0.5)
    """

    def __init__(
        self,
        max_shift: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(max_shift, int) or max_shift < 1:
            raise ValueError(f'max_shift must be a positive int, not {max_shift}')
        self.max_shift = max_shift

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            data = image.data.clone()  # (T, W, H, D)
            n_timepoints = data.shape[0]
            if n_timepoints < 4:
                continue
            d = data.shape[3]
            shifted = data.clone()
            for k in range(d):
                shift = int(
                    torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
                )
                if shift == 0:
                    continue
                # Circular shift along time (channel) axis
                shifted[:, :, :, k] = torch.roll(data[:, :, :, k], shifts=shift, dims=0)
            image.set_data(shifted)
        return subject
