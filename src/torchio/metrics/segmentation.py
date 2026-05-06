"""3-D segmentation quality metrics.

All functions accept boolean or integer label tensors with shape
``(C, W, H, D)`` or ``(W, H, D)``.
"""

from __future__ import annotations

import torch


def dice_coefficient(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int | None = None,
    ignore_background: bool = True,
    smooth: float = 1e-6,
) -> dict[int, float]:
    r"""Compute the Dice Similarity Coefficient (DSC) per label class.

    .. math::
        \mathrm{DSC}_c = \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|}

    Args:
        pred: Predicted label tensor ``(W, H, D)`` with integer class labels,
            or ``(C, W, H, D)`` one-hot where ``C`` = number of classes.
        gt: Ground-truth label tensor, same shape as ``pred``.
        num_classes: Total number of classes.  Inferred from ``max(pred, gt)``
            if not given.
        ignore_background: If ``True`` (default), class ``0`` is excluded
            from the returned dictionary.
        smooth: Laplace smoothing term to avoid division by zero.

    Returns:
        Dictionary mapping class index (``int``) to DSC score (``float``).

    Example:
        >>> scores = tio.metrics.dice_coefficient(pred_seg, gt_seg)
        >>> print(f"Tumour DSC: {scores[1]:.3f}")
    """
    # Collapse one-hot to label map if needed
    if pred.ndim == 4 and pred.shape[0] > 1:
        pred = pred.argmax(dim=0)
    else:
        pred = pred.squeeze(0) if pred.ndim == 4 else pred

    if gt.ndim == 4 and gt.shape[0] > 1:
        gt = gt.argmax(dim=0)
    else:
        gt = gt.squeeze(0) if gt.ndim == 4 else gt

    pred = pred.long()
    gt = gt.long()

    if num_classes is None:
        num_classes = int(max(pred.max().item(), gt.max().item())) + 1

    results: dict[int, float] = {}
    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        p_c = pred == c
        g_c = gt == c
        intersection = float((p_c & g_c).sum().item())
        union = float(p_c.sum().item() + g_c.sum().item())
        results[c] = (2.0 * intersection + smooth) / (union + smooth)
    return results


def hausdorff_distance_3d(
    pred: torch.Tensor,
    gt: torch.Tensor,
    percentile: float = 95.0,
) -> float:
    r"""Compute the percentile Hausdorff Distance (HD) between two binary masks.

    The HD measures the worst-case boundary error between a predicted and a
    ground-truth segmentation:

    .. math::
        \mathrm{HD}_{p}(A, B) = \max\!\left(
            d_p(A \to B),\; d_p(B \to A)
        \right)

    where :math:`d_p` is the *p*-th percentile of pairwise distances from
    surface voxels of one mask to the nearest surface voxel of the other.

    Args:
        pred: Binary prediction tensor ``(W, H, D)`` or ``(1, W, H, D)``.
        gt: Binary ground-truth tensor, same shape as ``pred``.
        percentile: Percentile for robust HD. Default ``95.0``.

    Returns:
        Hausdorff distance in voxels (float).

    Example:
        >>> hd = tio.metrics.hausdorff_distance_3d(pred_mask, gt_mask)
    """
    p = pred.bool().squeeze()
    g = gt.bool().squeeze()

    if p.ndim != 3 or g.ndim != 3:
        raise ValueError('pred and gt must be 3-D after squeezing')

    p_pts = p.nonzero(as_tuple=False).float()
    g_pts = g.nonzero(as_tuple=False).float()

    if len(p_pts) == 0 or len(g_pts) == 0:
        return float('inf')

    # Compute pairwise L2 distances — chunked to avoid OOM on large masks
    def _percentile_dist(src: torch.Tensor, tgt: torch.Tensor) -> float:
        chunk = 512
        min_dists = []
        for i in range(0, len(src), chunk):
            s = src[i:i + chunk]  # (M, 3)
            # (M, N) distance matrix via broadcasting
            diffs = s.unsqueeze(1) - tgt.unsqueeze(0)  # (M, N, 3)
            dists = (diffs ** 2).sum(-1).sqrt()  # (M, N)
            min_dists.append(dists.min(dim=1).values)
        all_min = torch.cat(min_dists)
        k = max(1, int(len(all_min) * percentile / 100.0))
        return float(all_min.topk(k, largest=True).values[-1].item())

    d_p2g = _percentile_dist(p_pts, g_pts)
    d_g2p = _percentile_dist(g_pts, p_pts)
    return max(d_p2g, d_g2p)
