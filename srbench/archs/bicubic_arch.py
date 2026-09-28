from torch import nn as nn
from torch.nn import functional as F

from srbench.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BicubicBaseline(nn.Module):
    """Interpolation baseline (the "Bicubic" row of the benchmark).

    A learned per-pixel linear map from the input bands to RGB (1x1 conv, i.e. a global radiometric
    calibration S2 reflectance -> Maxar 8-bit RGB) followed by bicubic upsampling. No spatial learning, so any
    gain of a deep model over this baseline is genuine super-resolution rather than colour mapping.

    Args:
        num_in_ch (int): Input bands. Default: 12.
        num_out_ch (int): Output channels. Default: 3.
        scale (int): Upsampling factor. Default: 4.
    """

    def __init__(self, num_in_ch=12, num_out_ch=3, scale=4):
        super().__init__()
        self.scale = scale
        self.color = nn.Conv2d(num_in_ch, num_out_ch, 1)

    def forward(self, x):
        return F.interpolate(self.color(x), scale_factor=self.scale, mode='bicubic', align_corners=False)
