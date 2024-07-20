from einops import rearrange
from torch import nn
import torch


class FtWeightModel(nn.Module):

    def __init__(
        self,
        dim=1280,
    ) -> None:
        super().__init__()
        # initialize with zeros
        self.weight = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, ft):
        """Transform the input feature using weighting.
        Args:
            ft (torch.Tensor): [B, C, n_kpts]
        """
        # apply sigmoid
        weight_softmax = torch.nn.functional.sigmoid(self.weight)
        return ft * weight_softmax

    def loss(self, kpt_embds_1, kpt_embds_2):
        """Compute the loss for the model.
        Args:
            kpt_embds_1 (torch.Tensor): [B, C, n_kpts]
            kpt_embds_2 (torch.Tensor): [B, C, n_kpts]
        """
        attns = torch.transpose(kpt_embds_1, dim0=-2, dim1=-1) @ kpt_embds_2  # (B, n_kpts, n_kpts)
        diag_attns = torch.diagonal(attns, dim1=-2, dim2=-1)  # (B, n_kpts)
        off_diag_attns = attns - torch.diag_embed(diag_attns)  # (B, n_kpts, n_kpts)
        # loss_pos = -torch.sum(diag_attns, dim=-1)  # (B,)
        loss_pos = -2*diag_attns[:,0]
        loss_neg = torch.sum(torch.sum(off_diag_attns, dim=-1), dim=-1)  # (B,)
        return torch.mean(loss_pos + loss_neg)

    # def loss(self, ft_1, ft_2, ft_3):
    #     """Compute the loss for the model.
    #     Args:
    #         ft_1 (torch.Tensor): [B, C]
    #         ft_2 (torch.Tensor): [B, C]
    #         is_pair_gt (torch.Tensor): [B,] 1 if ft_1 and ft_2 belong to the same keypoint index, 0 otherwise
    #     """
    #     attn_pos = torch.sum(ft_1 * ft_2, dim=-1)  # (B,)
    #     attn_neg = torch.sum(ft_1 * ft_3, dim=-1)  # (B,)
    #     loss = torch.mean(attn_neg - attn_pos)
    #     # loss = torch.mean(-torch.log(torch.sigmoid(attn_pos) + 1e-5) - torch.log(1 - torch.sigmoid(attn_neg) + 1e-5))
    #     # maximize attention for positive pairs and minimize attention for negative pairs
    #     # loss = loss_pos + loss_neg
    #     return loss


# class AlignFtModel(nn.Module):

#     def __init__(
#         self,
#         dim=1280,
#         hidden_depth=3,
#         activation="relu",
#         norm_type: Literal["batchnorm", "layernorm"] = "batchnorm",
#     ) -> None:
#         super().__init__()

#         self.mlp = mlp_factory.build_mlp(
#             input_dim=dim,
#             output_dim=dim,
#             hidden_dim=dim,
#             hidden_depth=hidden_depth,
#             add_input_activation=True,
#             activation=activation,
#             # weight_init="orthogonal",
#             # bias_init="zeros",
#             norm_type=norm_type,
#             # NOTE no output activation since we are using sigmoid in the loss
#         )

#     def forward(self, ft_1, ft_2):
#         """Transform the input features using an MLP.
#         Args:
#             ft_1 (torch.Tensor): [B, 1280]
#             ft_2 (torch.Tensor): [B, 1280]
#         """
#         # TODO maybe add selection mechanism
#         ft_1 = self.mlp(ft_1)
#         ft_2 = self.mlp(ft_2)
#         return ft_1, ft_2

#     def loss(self, ft_1, ft_2, is_pair: torch.Tensor):
#         """Compute the loss for the model.
#         Args:
#             ft_1 (torch.Tensor): [B, 1280]
#             ft_2 (torch.Tensor): [B, 1280]
#             is_pair (torch.Tensor): [B,] 1 if ft_1 and ft_2 belong to the same keypoint index, 0 otherwise
#         """
#         attn = torch.sum(ft_1 * ft_2, dim=-1)  # (B,)
#         attn = torch.sigmoid(attn)
#         # TODO add weight based on distance to true position
#         # false_negative_loss = torch.mean((1 - attn) * is_pair)
#         # false_positive_loss = torch.mean(attn * (1 - is_pair))
#         likelihood = attn * is_pair + (1 - attn) * (1 - is_pair)
#         # minimize negative log likelihood
#         loss = torch.mean(-torch.log(likelihood + 1e-5))
#         return loss
