import torch
import torch.nn as nn

from network.conv2d_layer import Conv2dLayer

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, latent_channels: int = 48, padding_type: str = 'zero', activation: str = 'lrelu', norm: str = 'none'):
        super().__init__()

        self.model=nn.Sequential(
            Conv2dLayer(in_channels, latent_channels, 7, 1, 3, padding_type=padding_type, activation=activation, norm=norm, sn=True),         # out: [B, 64, 256, 256]
            Conv2dLayer(latent_channels, latent_channels*2, 4, 2, 1, padding_type=padding_type, activation=activation, norm=norm, sn=True),   # out: [B, 128, 128, 128]
            Conv2dLayer(latent_channels*2, latent_channels*4, 4, 2, 1, padding_type=padding_type, activation=activation, norm=norm, sn=True), # out: [B, 256, 64, 64]
            Conv2dLayer(latent_channels*4, latent_channels*4, 4, 2, 1, padding_type=padding_type, activation=activation, norm=norm, sn=True), # out: [B, 256, 32, 32]
            Conv2dLayer(latent_channels*4, latent_channels*4, 4, 2, 1, padding_type=padding_type, activation=activation, norm=norm, sn=True), # out: [B, 256, 16, 16]
            Conv2dLayer(latent_channels*4, 1, 4, 2, 1, padding_type=padding_type, activation='none', norm='none', sn=True),                   # out: [B, 256, 8, 8]
        )

    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        return self.model(x)
