import torch
import torch.nn as nn

from models.inv_conv_film_block import InvConvFilmBlock

class FilmDecoderB0(nn.Module):
    """Decoder that uses FiLM conditioning.
    """

    def __init__(self, out_channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conditioning_dim = conditioning_dim
        # input is 1280x7x7 (padded to 1280x8x8), output is out_channelsx64x64 => we need 3 layers since 8*2^3 = 64
        self.blocks = nn.ModuleList([
            # 8x8 => 16x16
            InvConvFilmBlock(
                in_channels=1280,
                out_channels=512,
                conditioning_dim=conditioning_dim
            ),
            # 16x16 => 32x32
            InvConvFilmBlock(
                in_channels=512,
                out_channels=256,
                conditioning_dim=conditioning_dim
            ),
            # 32x32 => 64x64
            InvConvFilmBlock(
                in_channels=256,
                out_channels=out_channels,
                conditioning_dim=conditioning_dim
            ),
        ])

    def forward(self, image_features: torch.Tensor, conditioning: torch.Tensor):
        """
        args:
            image_features: Tensor of shape [B, 1280, 7, 7].
            conditioning: Tensor of shape [B, D].
        """
        assert image_features.shape[1] == 1280 and image_features.shape[2] == 7 and image_features.shape[3] == 7
        # pad to 8x8
        image_features = nn.functional.pad(image_features, (0, 1, 0, 1), mode='constant', value=0)
        for block in self.blocks:
            image_features = block(image_features, conditioning)
        return image_features

