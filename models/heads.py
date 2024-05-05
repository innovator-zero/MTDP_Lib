import torch.nn as nn
import torch.nn.functional as F


class BaseHead(nn.Module):
    """
    Head block for different tasks. Upsamples twice and applies a 1x1 convolution.
    :param int dim: Input dimension.
    :param int out_ch: Output channels.
    """

    def __init__(self, dim, out_ch):
        super().__init__()
        self.last_conv = nn.Conv2d(dim, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


class TransposeHead(nn.Module):
    """
    Head block for different tasks. Upsamples twice and applies a 1x1 convolution.
    :param int dim: Input dimension.
    :param int out_ch: Output channels.
    """

    def __init__(self, dim, out_ch):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)
        self.last_conv = nn.Conv2d(dim // 4, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.last_conv(x)

        return x
