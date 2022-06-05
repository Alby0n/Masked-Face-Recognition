import torch
import torch.nn as nn

from network.gated_conv import GatedConv2d, GatedDeConv2d

class CoarseNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, latent_channels: int = 48, padding_type: str = 'zero', activation: str = 'lrelu', norm: str = 'none'):
        super().__init__()

        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels*2, 3, 2, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 2, 1, padding_type = padding_type, activation = activation, norm = norm),
            # Bottleneck
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            ## dilated
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 2, dilation = 2, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 4, dilation = 4, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 8, dilation = 8, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 16, dilation = 16, padding_type = padding_type, activation = activation, norm = norm),
            ## end dilated
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            # decoder
            GatedDeConv2d(latent_channels*4, latent_channels*2, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedDeConv2d(latent_channels*2, latent_channels, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, padding_type = padding_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, padding_type = padding_type, activation = 'none', norm = norm),
            nn.Tanh()
        )

    def forward(self, img, mask):
        img_masked = img*(1 - mask) + mask
        x = torch.cat((img_masked, mask), dim=1)       # in: [B, 4, H, W]

        return self.coarse(x)
