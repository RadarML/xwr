"""Radar preprocessing toolkit for neural network training.

When converting complex spectrum to real-valued representations, we can apply
a range of different data augmentations. The supported data augmentations
according to the
[`abstract_dataloader.ext.augment`][abstract_dataloader.ext.augment]
conventions are:

- `azimuth_flip`: flip along azimuth axis.
- `doppler_flip`: flip along doppler axis.
- `range_scale`: apply random range scale.
- `speed_scale`: apply random speed scale.
- `radar_scale`: radar magnitude scale factor.
- `radar_phase`: phase shift across the frame.
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
