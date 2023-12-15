import torch
import torch.nn as nn

import models.film_conditioning_layer as film

class InvConvFilmBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # film is applied after convolution
        self.film_layer = film.FilmConditioning(out_channels, conditioning_dim)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        """
        args:
            x: Tensor of shape [B, C, H, W].
            conditioning: Tensor of shape [B, D].
        """
        x = self.convt(x) # [B, out_channels, 2*H, 2*W]
        x = self.film_layer(x, conditioning) # [B, out_channels, 2*H, 2*W]
        return x