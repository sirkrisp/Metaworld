import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, num_channels: int, conditioning_dim: int):
        """Constructs a FiLM conditioning layer.

        Args:
          num_channels: Number of filter channels to expect in the input.
        """
        super().__init__()
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.

        self._projection_add = nn.Linear(
            conditioning_dim,
            num_channels,
        )
        self._projection_mult = nn.Linear(
            conditioning_dim,
            num_channels,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        """
        args:
          conv_filters: Tensor of shape [B, C, H, W].
          conditioning: Tensor of shape [B, D].
        """
        B, C, _, _ = conv_filters.shape
        projected_cond_add = self._projection_add(conditioning)  # [B, C]
        projected_cond_mult = self._projection_mult(conditioning)  # [B, C]

        # [B, C] -> [B, C, 1, 1]
        projected_cond_add = projected_cond_add.view((B, C, 1, 1))
        projected_cond_mult = projected_cond_mult.view((B, C, 1, 1))

        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result
