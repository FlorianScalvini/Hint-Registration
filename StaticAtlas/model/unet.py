import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class UNet2(nn.Module):
    def __init__(self, num_layers, channels):
        super().__init__()

        self.BatchNorm = nn.BatchNorm3d
        self.Conv = nn.Conv3d
        self.ConvTranspose = nn.ConvTranspose3d
        self.avg_pool = F.avg_pool3d
        self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        self.lastConv = self.Conv(
            down_channels[0] + up_channels_out[0], dimension, kernel_size=3, padding=1
        )
        torch.nn.init.zeros_(self.lastConv.weight)
        torch.nn.init.zeros_(self.lastConv.bias)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size())

        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size()),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            # x = self.residues[depth](x)
            x = self.batchNorms[depth](x)
            x = x[
                :,
                :,
                : skips[depth].size()[2],
                : skips[depth].size()[3],
                : skips[depth].size()[4],
            ]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10

def pad_or_crop(x, shape):
    y = x[:, : shape[1]]
    if x.size()[1] < shape[1]:
        y = F.pad(y, (0, 0, 0, 0, 0, 0, shape[1] - x.size()[1], 0))
    assert y.size()[1] == shape[1]

    return y
