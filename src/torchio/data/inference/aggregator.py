from __future__ import annotations

import warnings

import numpy as np
import torch

from ...constants import CHANNELS_DIMENSION
from ..sampler import GridSampler


class GridAggregator:
    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of patches after
    inference of batches extracted by a [`GridSampler`](#torchio.data.GridSampler).

    Args:
        sampler: Instance of [`GridSampler`](#torchio.data.GridSampler) used to
            extract the patches.
        overlap_mode: If `'crop'`, the overlapping predictions will be
            cropped. If `'average'`, the predictions in the overlapping areas
            will be averaged with equal weights. If `'hann'`, the predictions
            in the overlapping areas will be weighted with a Hann window
            function. See the [grid aggregator tests](https://github.com/medtorch/medtorch/blob/main/tests/data/inference/test_aggregator.py) for a raw visualization
            of the three modes.
        downsampling_factor: Factor by which the output volume is expected to
            be smaller than the input volume in each spatial dimension. This is
            useful when the model downsamples the input (e.g., with strided
            convolutions or pooling layers). Currently, only a single integer
            is supported, which applies the same downsampling factor to all
            spatial dimensions.


    Note:
        Adapted from NiftyNet. See [this NiftyNet tutorial
        ](https://niftynet.readthedocs.io/en/dev/window_sizes.html) for more
        information about patch-based sampling.
    """

    def __init__(
        self,
        sampler: GridSampler,
        overlap_mode: str = 'crop',
        downsampling_factor: int = 1,  # TODO: support one per dimension
    ):
        subject = sampler.subject
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = subject.spatial_shape
        self._output_tensor: torch.Tensor | None = None
        self.patch_overlap = sampler.patch_overlap
        self.patch_size = sampler.patch_size
        self._parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor: torch.Tensor | None = None
        self._hann_window: torch.Tensor | None = None
        self._downsampling_factor = downsampling_factor
        shape_array = np.array(subject.spatial_shape) // self._downsampling_factor
        self.spatial_shape = tuple(shape_array.tolist())
        # Uncertainty tracking tensors, filled only if requested in add_batch().
        self._sum_tensor: torch.Tensor | None = None
        self._sq_tensor: torch.Tensor | None = None
        self._count_tensor: torch.Tensor | None = None

    @staticmethod
    def _parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average', 'hann'):
            message = (
                'Overlap mode must be "crop", "average" or "hann" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)

    def _crop_patch(
        self,
        patch: torch.Tensor,
        location: np.ndarray,
        overlap: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        half_overlap = overlap // 2  # overlap is always even in grid sampler
        index_ini, index_fin = location[:3], location[3:]

        # If the patch is not at the border, we crop half the overlap
        crop_ini: np.ndarray = half_overlap.copy()
        crop_fin: np.ndarray = half_overlap.copy()

        # If the volume has been padded, we don't need to worry about cropping
        if self.volume_padded:
            pass
        else:
            crop_ini *= index_ini > 0
            crop_fin *= index_fin != self.spatial_shape

        # Update the location of the patch in the volume
        new_index_ini = index_ini + crop_ini
        new_index_fin = index_fin - crop_fin
        new_location = np.hstack((new_index_ini, new_index_fin))

        patch_size = np.asarray(patch.shape[-3:], dtype=int)
        crop_fin = crop_fin.astype(int)
        i_ini, j_ini, k_ini = crop_ini
        i_fin, j_fin, k_fin = patch_size - crop_fin
        # Make type checkers happy
        i_ini = int(i_ini)
        j_ini = int(j_ini)
        k_ini = int(k_ini)
        i_fin = int(i_fin)
        j_fin = int(j_fin)
        k_fin = int(k_fin)
        cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        return cropped_patch, new_location

    def _initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._output_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def _initialize_uncertainty_tensors(self, batch: torch.Tensor) -> None:
        if self._sq_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._sum_tensor = torch.zeros(
            num_channels, *self.spatial_shape, dtype=torch.float32
        )
        self._sq_tensor = torch.zeros(
            num_channels, *self.spatial_shape, dtype=torch.float32
        )
        self._count_tensor = torch.zeros(
            num_channels, *self.spatial_shape, dtype=torch.float32
        )

    def _initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    @staticmethod
    def _get_hann_window(patch_size) -> torch.Tensor:
        hann_window_3d = torch.as_tensor([1])
        # create a n-dim hann window
        for spatial_dim, size in enumerate(patch_size):
            window_shape = np.ones_like(patch_size)
            window_shape[spatial_dim] = size
            hann_window_1d = torch.hann_window(
                size + 2,
                periodic=False,
            )
            hann_window_1d = hann_window_1d[1:-1].view(*window_shape)
            hann_window_3d = hann_window_3d * hann_window_1d
        return hann_window_3d

    def _initialize_hann_window(self) -> None:
        if self._hann_window is not None:
            return
        self._hann_window = self._get_hann_window(self.patch_size)

    def add_batch(
        self,
        batch_tensor: torch.Tensor,
        locations: torch.Tensor,
        track_uncertainty: bool = False,
    ) -> None:
        """Add batch processed by a network to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. `batch['image'][torchio.DATA]`.
            locations: 2D tensor with shape $(B, 6)$ representing the
                patch indices in the original image. They are typically
                extracted using `batch[torchio.LOCATION]`.
        """
        batch = batch_tensor.cpu()
        locations_array = locations.cpu().numpy() // self._downsampling_factor
        target_shapes = locations_array[:, 3:] - locations_array[:, :3]
        # There should be only one patch size
        assert len(np.unique(target_shapes, axis=0)) == 1
        input_spatial_shape = tuple(batch.shape[-3:])
        target_spatial_shape_array = target_shapes[0]
        target_spatial_shape = tuple(target_spatial_shape_array.tolist())
        if input_spatial_shape != target_spatial_shape:
            message = (
                f'The shape of the input batch, {input_spatial_shape},'
                ' does not match the shape of the target location,'
                f' which is {target_spatial_shape}'
            )
            raise RuntimeError(message)
        self._initialize_output_tensor(batch)
        assert isinstance(self._output_tensor, torch.Tensor)
        if track_uncertainty:
            self._initialize_uncertainty_tensors(batch)
        if self.overlap_mode == 'crop':
            for patch, location in zip(batch, locations_array, strict=True):
                cropped_patch, new_location = self._crop_patch(
                    patch,
                    location,
                    self.patch_overlap,
                )
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = new_location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] = cropped_patch
                if track_uncertainty and self._sq_tensor is not None:
                    f = cropped_patch.float()
                    assert self._sum_tensor is not None
                    self._sum_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += f
                    self._sq_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += f ** 2
                    self._count_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += 1
        elif self.overlap_mode == 'average':
            self._initialize_avgmask_tensor(batch)
            assert isinstance(self._avgmask_tensor, torch.Tensor)
            for patch, location in zip(batch, locations, strict=True):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += 1
                if track_uncertainty and self._sq_tensor is not None:
                    f = patch.float()
                    assert self._sum_tensor is not None
                    self._sum_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += f
                    self._sq_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += f ** 2
                    self._count_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += 1
        elif self.overlap_mode == 'hann':
            # To handle edge and corners avoid numerical problems, we save the
            # hann window in a different tensor
            # At the end, it will be filled with ones (or close values) where
            # there is overlap and < 1 where there is not
            # When we divide, the multiplication will be canceled in areas that
            # do not overlap
            self._initialize_avgmask_tensor(batch)
            self._initialize_hann_window()

            if self._output_tensor.dtype != torch.float32:
                self._output_tensor = self._output_tensor.float()

            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            if self._avgmask_tensor.dtype != torch.float32:
                self._avgmask_tensor = self._avgmask_tensor.float()

            for patch, location in zip(batch, locations, strict=True):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                raw_patch = patch.float()
                patch = patch * self._hann_window
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += patch
                assert self._hann_window is not None
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += self._hann_window
                if track_uncertainty and self._sq_tensor is not None:
                    # Track uncertainty on raw patch predictions per voxel.
                    assert self._sum_tensor is not None
                    self._sum_tensor[
                        :, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin
                    ] += raw_patch
                    self._sq_tensor[
                        :, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin
                    ] += raw_patch ** 2
                    self._count_tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] += 1

    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        assert isinstance(self._output_tensor, torch.Tensor)
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode in ['average', 'hann']:
            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/medtorch/medtorch/issues/526
            output = torch.true_divide(
                self._output_tensor,
                self._avgmask_tensor,
            )
        else:
            output = self._output_tensor
        if self.volume_padded:
            from ...transforms import Crop

            border = self.patch_overlap // 2
            cropping_values = [int(value) for value in border.repeat(2).tolist()]
            cropping = (
                cropping_values[0],
                cropping_values[1],
                cropping_values[2],
                cropping_values[3],
                cropping_values[4],
                cropping_values[5],
            )
            crop = Crop(cropping)
            cropped = crop(output)
            assert isinstance(cropped, torch.Tensor)
            return cropped
        return output

    def get_uncertainty_map(self) -> torch.Tensor:
        """Compute per-voxel predictive variance from overlapping patches.

        When ``track_uncertainty=True`` is passed to
        :meth:`add_batch`, this method returns the sample variance
        of predictions in each voxel across all overlapping patches
        that contributed to it:

        .. math::
            \\sigma^2(\\mathbf{r}) = \\frac{\\sum_i p_i^2}{n} -
            \\left(\\frac{\\sum_i p_i}{n}\\right)^2

        This provides a cheap, model-agnostic proxy for epistemic
        uncertainty in patch-based inference.

        Returns:
            Float tensor of shape ``(C, W, H, D)`` containing per-voxel
            variance values.

        Raises:
            RuntimeError: If :meth:`add_batch` was never called with
                ``track_uncertainty=True``.
        """
        if (
            self._sum_tensor is None
            or self._sq_tensor is None
            or self._count_tensor is None
        ):
            raise RuntimeError(
                'Uncertainty tracking is disabled. '
                'Call add_batch(..., track_uncertainty=True) first.'
            )
        count = self._count_tensor.clamp(min=1)
        mean = self._sum_tensor / count
        mean_sq = self._sq_tensor / count
        variance = (mean_sq - mean ** 2).clamp(min=0)
        return variance
