# Command-line tools

## `medtorch-transform`

A transform can be quickly applied to an image file using the command-line
tool `medtorch-transform`, which is automatically installed by `pip`:

```
$ medtorch-transform input.nii RandomAffine output.nii.gz --kwargs "degrees=(0,0,10) scales=0.1" --seed 42
```

For more information, run `medtorch-transform --help`.

## `medtorch-info`

To print some image metadata, `medtorch-info` can be used. Adding the `--plot`
argument will plot the image using Matplotlib:

```
$ medtorch-info ~/.cache/torchio/mni_colin27_1998_nifti/colin27_t1_tal_lin.nii
ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; dtype: torch.FloatTensor; memory: 27.1 MiB)
```

For more information, run `medtorch-info --help`.

Legacy aliases (`tiotr`, `tiohd`, and `torchio-transform`) are still available.
