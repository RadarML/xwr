"""Radar preprocessing toolkit for neural network training.

!!! warning

    This submodule requires `torch` and `torchvision` to be installed, e.g. via
    the `nn` extra.

!!! tip

    This submodule supports both `np.ndarray` and `torch.Tensor` inputs
    directly out-of-the box (with no additional overhead, since we already
    require `torch` to handle resizing).

When converting complex spectrum to real-valued representations, we can apply
a range of different data augmentations. The supported data augmentations
according to the
[`abstract_dataloader.ext.augment`][abstract_dataloader.ext.augment]
conventions are:

| Augmentation Key | Description |
| ---------------- | ----------- |
| `azimuth_flip`   | Flip along azimuth axis. |
| `doppler_flip`   | Flip along doppler axis. |
| `range_scale`    | Apply random range scale. |
| `speed_scale`    | Apply random speed scale. |
| `radar_scale`    | Radar magnitude scale factor. |
| `radar_phase`    | Phase shift across the frame. |

??? quote "Sample Hydra Configuration for `abstract_dataloader.ext.augment`"

    ```yaml
    _target_: abstract_dataloader.ext.augment.Augmentations
    azimuth_flip:
      _target_: abstract_dataloader.ext.augment.Bernoulli
      p: 0.5
    doppler_flip:
      _target_: abstract_dataloader.ext.augment.Bernoulli
      p: 0.5
    radar_scale:
      _target_: abstract_dataloader.ext.augment.TruncatedLogNormal
      std: 0.2
      clip: 2.0
    radar_phase:
      _target_: abstract_dataloader.ext.augment.Uniform
      lower: -3.14159265
      upper: 3.14159265
    range_scale:
      _target_: abstract_dataloader.ext.augment.Uniform
      lower: 1.0
      upper: 2.0
    speed_scale:
      _target_: abstract_dataloader.ext.augment.TruncatedLogNormal
      std: 0.2
      clip: 2.0
    ```
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.nn", "beartype.beartype"):
    from .representations import (
        Magnitude,
        PhaseAngle,
        PhaseVec,
        Representation,
    )
    from .utils import resize

__all__ = ["resize", "Magnitude", "PhaseAngle", "PhaseVec", "Representation"]
