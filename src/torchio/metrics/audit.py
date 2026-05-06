"""Transform pipeline auditing utilities.

:class:`TransformAudit` wraps a :class:`~torchio.transforms.Compose`
pipeline and measures per-transform wall-clock time and memory usage.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

import torch

from ..data.subject import Subject
from ..transforms.transform import Transform


@dataclass
class TransformStats:
    """Per-transform timing and memory statistics."""
    name: str
    calls: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float('inf')
    max_seconds: float = 0.0
    mean_output_mb: float = 0.0
    _mb_accum: float = field(default=0.0, repr=False)

    def update(self, elapsed: float, output_mb: float) -> None:
        self.calls += 1
        self.total_seconds += elapsed
        self.min_seconds = min(self.min_seconds, elapsed)
        self.max_seconds = max(self.max_seconds, elapsed)
        self._mb_accum += output_mb
        self.mean_output_mb = self._mb_accum / self.calls

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / max(self.calls, 1)

    def __str__(self) -> str:
        return (
            f'{self.name}: '
            f'mean={self.mean_seconds*1000:.1f} ms, '
            f'min={self.min_seconds*1000:.1f} ms, '
            f'max={self.max_seconds*1000:.1f} ms, '
            f'output={self.mean_output_mb:.2f} MB '
            f'(n={self.calls})'
        )


class TransformAudit:
    """Instrument a transform pipeline to measure per-step performance.

    Wraps a sequence of transforms and intercepts each call to record
    wall-clock time and output tensor memory.

    Args:
        transforms: List of :class:`~torchio.transforms.Transform` objects
            to audit, in order.

    Example:
        >>> import torchio as tio
        >>> pipeline = [tio.ToCanonical(), tio.ZNormalization(), tio.RandomFlip()]
        >>> audit = tio.metrics.TransformAudit(pipeline)
        >>> _ = audit(subject)
        >>> audit.report()
    """

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)
        self.stats: dict[str, TransformStats] = {
            t.__class__.__name__: TransformStats(name=t.__class__.__name__)
            for t in transforms
        }

    def __call__(self, subject: Subject) -> Subject:
        """Run the pipeline on a subject, recording timing per transform.

        Args:
            subject: Input :class:`~torchio.data.Subject`.

        Returns:
            The subject after all transforms have been applied.
        """
        current = copy.deepcopy(subject)
        for transform in self.transforms:
            name = transform.__class__.__name__
            t0 = time.perf_counter()
            current = transform(current)
            elapsed = time.perf_counter() - t0
            assert isinstance(current, Subject)
            # Measure total tensor memory in the output
            mb = sum(
                img.data.element_size() * img.data.nelement() / 1_048_576
                for img in current.get_images(intensity_only=False)
            )
            self.stats[name].update(elapsed, mb)
        return current

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        for name in self.stats:
            self.stats[name] = TransformStats(name=name)

    def report(self) -> str:
        """Return a formatted report of per-transform statistics.

        Also prints to stdout.

        Returns:
            Multi-line string report.
        """
        lines = ['TransformAudit Report', '=' * 50]
        total = sum(s.total_seconds for s in self.stats.values())
        for stats in self.stats.values():
            pct = 100.0 * stats.total_seconds / (total + 1e-12)
            lines.append(f'  {stats}  [{pct:.1f}% of total]')
        lines.append('-' * 50)
        lines.append(
            f'  Total wall time: {total*1000:.1f} ms over '
            f'{next(iter(self.stats.values())).calls} run(s)'
        )
        text = '\n'.join(lines)
        print(text)  # noqa: T201
        return text

    def bottleneck(self) -> str:
        """Return the name of the slowest transform."""
        return max(self.stats, key=lambda k: self.stats[k].mean_seconds)
