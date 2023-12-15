"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import math
import os
from typing import List, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer

# TODO rename => name is confusing because there is swinv2 already
class SwinEncoder(nn.Module):
    r"""
    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        window_size: int,
        encoder_layer: List[int],
        patch_size: int,
        embed_dim: int,
        num_heads: List[int],
        pos_drop: float = 0.0,
        # TODO name_or_path ??
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert len(input_size) == 2, "input size must be (width, height)"
        assert len(encoder_layer) == len(num_heads), "encoder_layer and num_heads must have the same length"
        assert len(encoder_layer) == 4, "encoder_layer must have 4 layers corresponding to the 4 stages of swin"
        assert input_size[0] % (patch_size*2**3) == 0 and input_size[1] % (patch_size*2**3) == 0, "input size must be divisible by patch size*(2**3)"


        self.pos_drop = nn.Dropout(pos_drop) if pos_drop > 0 else None

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=0,
        )

        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        if self.pos_drop:
            x = self.pos_drop(x)
        # x = self.model.pos_drop(x)  # not defined in swin
        x = self.model.layers(x)
        return x