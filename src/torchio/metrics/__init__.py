"""torchio.metrics — 3-D medical image quality and segmentation metrics.

All functions operate on :class:`torch.Tensor` instances with shape
``(C, W, H, D)`` or ``(W, H, D)``.  No additional package dependencies
beyond NumPy, SciPy and PyTorch are required.
"""

from .audit import TransformAudit
from .image_quality import nmi3d
from .image_quality import psnr3d
from .image_quality import ssim3d
from .segmentation import dice_coefficient
from .segmentation import hausdorff_distance_3d

__all__ = [
    'ssim3d',
    'psnr3d',
    'nmi3d',
    'dice_coefficient',
    'hausdorff_distance_3d',
    'TransformAudit',
]
