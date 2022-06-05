import torch
import torch.nn as nn

from network.coarse_network import CoarseNetwork
from network.refinement_network import RefinementNetwork

class Generator(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, latent_channels: int = 48, padding_type: str = 'zero', activation: str = 'lrelu', norm: str = 'none'):
        super().__init__()

        self.coarse = CoarseNetwork(in_channels=in_channels, out_channels=out_channels, latent_channels=latent_channels, padding_type=padding_type, activation=activation, norm=norm)
        self.refinement = RefinementNetwork(in_channels=in_channels, out_channels=out_channels, latent_channels=latent_channels, padding_type=padding_type, activation=activation, norm=norm)

    def forward(self, img, mask):
        coarse_img = self.coarse(img, mask)
        refined_img = self.refinement(img, coarse_img, mask)

        return coarse_img, refined_img
