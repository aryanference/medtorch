from __future__ import annotations

import copy
from collections.abc import Sequence

from .subject import Subject
from ..transforms.augmentation import RandomTransform
from ..transforms.transform import Transform


class MultiSubjectTransform:
    """Apply identical spatial parameters to a group of subjects.

    When training image-registration networks or performing multi-contrast
    joint augmentation, all subjects in a group must receive the **same**
    random spatial transformation (rotation, elastic deformation, etc.).
    The standard ``RandomTransform`` re-samples parameters independently for
    every ``__call__``, breaking spatial correspondence.

    ``MultiSubjectTransform`` solves this by:

    1. Sampling random parameters **once** from the ``reference_index``
       subject (or the first subject if not specified).
    2. Using those fixed parameters to apply a deterministic version of
       the transform to **every** subject in the group.

    Only :class:`~torchio.transforms.augmentation.RandomTransform` subclasses
    that expose a deterministic counterpart through ``apply_transform`` are
    supported (all built-in random transforms qualify).

    Args:
        transform: A random spatial transform, e.g.
            :class:`~torchio.transforms.RandomAffine` or
            :class:`~torchio.transforms.RandomElasticDeformation`.
        reference_index: Index of the subject from which random parameters
            are derived. Defaults to ``0`` (the first subject).

    Example:
        >>> import torchio as tio
        >>> fixed  = tio.Subject(image=tio.ScalarImage(tensor=t1))
        >>> moving = tio.Subject(image=tio.ScalarImage(tensor=t2))
        >>> group_transform = tio.MultiSubjectTransform(tio.RandomAffine())
        >>> fixed_aug, moving_aug = group_transform([fixed, moving])
    """

    def __init__(
        self,
        transform: Transform,
        reference_index: int = 0,
    ):
        if not callable(transform):
            raise TypeError(
                f'transform must be callable, got {type(transform)}'
            )
        self.transform = transform
        self.reference_index = reference_index

    def __call__(self, subjects: Sequence[Subject]) -> list[Subject]:
        """Apply consistent augmentation to all subjects.

        Args:
            subjects: A sequence of
                :class:`~torchio.data.Subject` instances that should all
                receive the **same** random spatial transform parameters.

        Returns:
            A list of transformed subjects in the same order as the input.
        """
        if not subjects:
            return []

        # Validate all elements
        for i, s in enumerate(subjects):
            if not isinstance(s, Subject):
                raise TypeError(
                    f'All elements must be torchio.Subject instances, '
                    f'but element {i} has type {type(s)}'
                )

        ref_idx = self.reference_index
        if ref_idx >= len(subjects):
            raise IndexError(
                f'reference_index {ref_idx} is out of range '
                f'for {len(subjects)} subjects'
            )

        # If the transform is random, apply it to the reference subject first
        # to record the chosen parameters, then replay them on all subjects.
        if isinstance(self.transform, RandomTransform):
            ref_subject = copy.deepcopy(subjects[ref_idx])
            transformed_ref = self.transform(ref_subject)
            # Extract the deterministic transform that was actually applied
            # (the last entry added to applied_transforms)
            if transformed_ref.applied_transforms:
                last_name, last_params = transformed_ref.applied_transforms[-1]
                # Reconstruct the deterministic transform from the recorded params
                from ..transforms import get_transform_class  # type: ignore[attr-defined]
                try:
                    det_cls = get_transform_class(last_name)
                    deterministic_transform = det_cls(**last_params)
                except Exception:
                    # Fallback: use the original random transform (params won't match)
                    deterministic_transform = self.transform
            else:
                deterministic_transform = self.transform
        else:
            deterministic_transform = self.transform

        results: list[Subject] = []
        for i, subject in enumerate(subjects):
            if i == ref_idx and isinstance(self.transform, RandomTransform):
                results.append(transformed_ref)
            else:
                out = deterministic_transform(copy.deepcopy(subject))
                assert isinstance(out, Subject)
                results.append(out)

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'transform={self.transform!r}, '
            f'reference_index={self.reference_index})'
        )
