"""FireGuard forecast models M1-M4 (architectures.py) and their checkpoint format."""

from .architectures import (
    MODEL_REGISTRY,
    AttentionResUNet,
    ConvLSTMUNet,
    SpatialUNet,
    StackedDaysUNet,
)
from .unet import UNet

__all__ = ["MODEL_REGISTRY", "SpatialUNet", "AttentionResUNet", "StackedDaysUNet", "ConvLSTMUNet",
           "UNet"]
