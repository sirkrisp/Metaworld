"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import math
import os
from typing import List, Union

import numpy as np
from PIL import Image
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.models.swin_transformer import SwinTransformer
from torchvision.transforms.functional import resize, rotate


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
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        patch_size: int,
        embed_dim: int,
        num_heads: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

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
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    # @staticmethod
    # def crop_margin(img: Image.Image) -> Image.Image:
    #     data = np.array(img.convert("L"))
    #     data = data.astype(np.uint8)
    #     max_val = data.max()
    #     min_val = data.min()
    #     if max_val == min_val:
    #         return img
    #     data = (data - min_val) / (max_val - min_val) * 255
    #     gray = 255 * (data < 200).astype(np.uint8)

    #     coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    #     a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    #     return img.crop((a, b, w + a, h + b))

    # @property
    # def to_tensor(self):
    #     # TODO
    #     # if self.training:
    #     #     return train_transform
    #     # else:
    #     #     return test_transform
    #     pass

    # def prepare_input(
    #     self, img: Image.Image, random_padding: bool = False
    # ) -> torch.Tensor:
    #     """
    #     Convert PIL Image to tensor according to specified input_size after following steps below:
    #         - resize
    #         - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
    #         - pad
    #     """
    #     if img is None:
    #         return
    #     # crop margins
    #     try:
    #         img = self.crop_margin(img.convert("RGB"))
    #     except OSError:
    #         # might throw an error for broken files
    #         return
    #     if img.height == 0 or img.width == 0:
    #         return
    #     if self.align_long_axis and (
    #         (self.input_size[0] > self.input_size[1] and img.width > img.height)
    #         or (self.input_size[0] < self.input_size[1] and img.width < img.height)
    #     ):
    #         img = rotate(img, angle=-90, expand=True)
    #     img = resize(img, min(self.input_size))
    #     img.thumbnail((self.input_size[1], self.input_size[0]))
    #     delta_width = self.input_size[1] - img.width
    #     delta_height = self.input_size[0] - img.height
    #     if random_padding:
    #         pad_width = np.random.randint(low=0, high=delta_width + 1)
    #         pad_height = np.random.randint(low=0, high=delta_height + 1)
    #     else:
    #         pad_width = delta_width // 2
    #         pad_height = delta_height // 2
    #     padding = (
    #         pad_width,
    #         pad_height,
    #         delta_width - pad_width,
    #         delta_height - pad_height,
    #     )
    #     return self.to_tensor(ImageOps.expand(img, padding))