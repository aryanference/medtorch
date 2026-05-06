"""3-D image quality metrics for medical imaging.

All metrics accept 3-D spatial tensors ``(W, H, D)`` or 4-D tensors
``(C, W, H, D)``. When a 4-D tensor is provided the metric is averaged
across channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_3d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure input is 3-D ``(W, H, D)`` by averaging channels."""
    if tensor.ndim == 3:
        return tensor.float()
    if tensor.ndim == 4:
        return tensor.float().mean(dim=0)
    raise ValueError(f'Expected 3-D or 4-D tensor, got {tensor.ndim}-D')


def ssim3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 7,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    r"""Compute the 3-D Structural Similarity Index (SSIM).

    Extends the 2-D SSIM [Wang et al., 2004] to volumetric data by using
    3-D Gaussian-weighted local statistics.

    Args:
        img1: Reference image tensor, shape ``(W, H, D)`` or ``(C, W, H, D)``.
        img2: Distorted image tensor, same shape as ``img1``.
        window_size: Side length of the cubic local window. Default ``7``.
        data_range: Dynamic range of the images. If ``None``, inferred from
            ``img1`` as ``img1.max() - img1.min()``.
        k1: Stability constant for luminance. Default ``0.01``.
        k2: Stability constant for contrast. Default ``0.03``.

    Returns:
        SSIM value in ``[-1, 1]``. Perfect agreement yields ``1.0``.

    Example:
        >>> import torch, torchio as tio
        >>> a = torch.rand(1, 64, 64, 64)
        >>> b = a + 0.01 * torch.randn_like(a)
        >>> score = tio.metrics.ssim3d(a, b)
    """
    a = _to_3d(img1).unsqueeze(0).unsqueeze(0)  # (1,1,W,H,D)
    b = _to_3d(img2).unsqueeze(0).unsqueeze(0)

    if data_range is None:
        data_range = float(img1.float().max() - img1.float().min()) + 1e-8

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    p = window_size // 2

    # Build 3-D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - p
    gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,w,w,w)

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, kernel, padding=p)

    mu1 = conv(a)
    mu2 = conv(b)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = conv(a ** 2) - mu1_sq
    sigma2_sq = conv(b ** 2) - mu2_sq
    sigma12 = conv(a * b) - mu12

    ssim_map = (
        (2 * mu12 + c1) * (2 * sigma12 + c2)
        / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    )
    return float(ssim_map.mean().item())


def psnr3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float | None = None,
) -> float:
    r"""Compute 3-D Peak Signal-to-Noise Ratio (PSNR) in decibels.

    .. math::
        \mathrm{PSNR} = 10 \log_{10}\!\left(\frac{V^2}{\mathrm{MSE}}\right)

    where :math:`V` is the dynamic range and :math:`\mathrm{MSE}` is the
    mean squared error between the two volumes.

    Args:
        img1: Reference image tensor ``(W, H, D)`` or ``(C, W, H, D)``.
        img2: Reconstructed image tensor, same shape as ``img1``.
        max_val: Dynamic range. If ``None``, inferred as
            ``img1.max() - img1.min()``.

    Returns:
        PSNR in dB.  Higher is better.

    Example:
        >>> score = tio.metrics.psnr3d(reference, noisy)
    """
    a = img1.float()
    b = img2.float()
    if max_val is None:
        max_val = float(a.max() - a.min()) + 1e-8
    mse = float(((a - b) ** 2).mean().item())
    if mse == 0:
        return float('inf')
    import math
    return 10 * math.log10(max_val ** 2 / mse)


def nmi3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    bins: int = 64,
) -> float:
    r"""Compute Normalised Mutual Information (NMI) between two volumes.

    NMI is a registration quality metric that measures statistical dependency
    between image intensities, invariant to linear intensity changes:

    .. math::
        \mathrm{NMI}(A, B) =
        \frac{H(A) + H(B)}{H(A, B)}

    where :math:`H(\cdot)` denotes Shannon entropy.

    Args:
        img1: First image tensor ``(W, H, D)`` or ``(C, W, H, D)``.
        img2: Second image tensor, same shape as ``img1``.
        bins: Number of histogram bins for density estimation. Default ``64``.

    Returns:
        NMI value in ``[1, 2]``. Perfectly aligned identical volumes yield
        ``2.0``; independent images yield values close to ``1.0``.

    Example:
        >>> nmi = tio.metrics.nmi3d(fixed, moving)
    """
    a = _to_3d(img1).reshape(-1)
    b = _to_3d(img2).reshape(-1)

    a_min, a_max = float(a.min()), float(a.max())
    b_min, b_max = float(b.min()), float(b.max())

    # Marginal histograms
    ha = torch.histc(a, bins=bins, min=a_min, max=a_max).float() + 1e-10
    hb = torch.histc(b, bins=bins, min=b_min, max=b_max).float() + 1e-10

    # Joint histogram via 2-D binning
    a_idx = ((a - a_min) / (a_max - a_min + 1e-8) * (bins - 1)).long().clamp(0, bins - 1)
    b_idx = ((b - b_min) / (b_max - b_min + 1e-8) * (bins - 1)).long().clamp(0, bins - 1)
    joint = torch.zeros(bins, bins, dtype=torch.float32)
    joint.index_put_((a_idx, b_idx), torch.ones_like(a), accumulate=True)
    joint = joint + 1e-10

    # Normalise to probability distributions
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    pab = joint / joint.sum()

    # Shannon entropies
    ha_ent = -float((pa * pa.log()).sum().item())
    hb_ent = -float((pb * pb.log()).sum().item())
    hab_ent = -float((pab * pab.log()).sum().item())

    nmi = (ha_ent + hb_ent) / (hab_ent + 1e-10)
    return float(nmi)
