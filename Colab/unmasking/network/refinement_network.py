import torch
import torch.nn as nn

from network.contextual_attention import ContextualAttention
from network.gated_conv import GatedConv2d, GatedDeConv2d

class RefinementNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, latent_channels: int = 48, padding_type: str = 'zero', activation: str = 'lrelu', norm: str = 'none'):
        super().__init__()

        # b1 has attention
        self.b1_1 = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 2, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation='relu', norm=norm)
        )
        self.b1_2 = nn.Sequential(
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm)
        )
        self.context_attention=ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)

        # b2 is conv only
        self.b2 = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 2, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 2, dilation=2, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 4, dilation=4, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 8, dilation=8, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 16, dilation=16, padding_type=padding_type, activation=activation, norm=norm)
        )

        self.combine = nn.Sequential(
            GatedConv2d(latent_channels*8, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedDeConv2d(latent_channels*4, latent_channels*2, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedDeConv2d(latent_channels*2, latent_channels, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, padding_type=padding_type, activation=activation, norm=norm),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, padding_type=padding_type, activation='none', norm=norm),
            nn.Tanh()
        )

    def forward(self, img, coarse_img, mask):
        img_masked = img * (1 - mask) + coarse_img * mask
        x = torch.cat([img_masked, mask], dim=1)

        x_1 = self.b2(x)

        x_2 = self.b1_1(x)
        mask_s = nn.functional.interpolate(mask, (x_2.shape[2], x_2.shape[3]))
        x_2 = self.context_attention(x_2, x_2, mask_s)
        x_2 = self.b1_2(x_2)

        y = torch.cat([x_1, x_2], dim=1)
        y = self.combine(y)
        y = nn.functional.interpolate(y, (img.shape[2], img.shape[3]))

        return y
