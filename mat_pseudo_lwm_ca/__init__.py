# =============================================================================
# mat_pseudo_lwm_ca/__init__.py  —  Package exports
#
# This package implements Coordinate Attention (CA) augmented pretraining
# for the MAT Pseudo-Grid LWM (Method 2: 8×16 Pseudo-Grid).
# =============================================================================

from mat_pseudo_lwm_ca.coordatt import CoordAtt, HSwish, HSigmoid
from mat_pseudo_lwm_ca.mat_pseudo_lwm_ca import (
    MATPseudoLWMWithCA,
    channels_to_patches,
    mask_patches,
)

__version__ = "1.0.0"

__all__ = [
    # Coordinate Attention
    "CoordAtt",
    "HSigmoid",
    "HSwish",
    # End-to-end pretraining model
    "MATPseudoLWMWithCA",
    # Pipeline helpers
    "channels_to_patches",
    "mask_patches",
]
