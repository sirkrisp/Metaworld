import torch
import torch.nn as nn
import inspect
import models.model_blocks as blocks


class ModelBase(nn.Module):

    def __init__(self):
        super().__init__()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(
            self,
            weight_decay,
            learning_rate: float,
            beta1: float,
            beta2: float,
            device_type: str
    ):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        betas = (beta1, beta2)

        # # separate out all parameters to those that will and won't experience regularizing weight decay
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.parameter.Parameter)
        # blacklist_weight_modules = (torch.nn.LayerNorm, blocks.LayerNorm, torch.nn.Embedding)
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        #         # random note: because named_modules and named_parameters are recursive
        #         # we will see the same tensors p many times. but doing it this way
        #         # allows us to know which parent module any tensor p belongs to...
        #         if pn.endswith('bias'):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #             # print(f"DECAY: {m}, {fpn}, {pn}")
        #         elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)

        #         if "in_proj" in fpn:
        #             print(f"IN_PROJ: module {m}, {fpn}, {pn} not being decayed or not")

        # # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(
        #     inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # assert len(
        #     param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params),)

        # # create the pytorch optimizer object
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(
        #         list(decay))], "weight_decay": weight_decay},
        #     {"params": [param_dict[pn]
        #                 for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        # ]
        # # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        # use_fused = (device_type == 'cuda') and (
        #         'fused' in inspect.signature(torch.optim.AdamW).parameters)
        # print(f"using fused AdamW: {use_fused}")
        # extra_args = dict(fused=True) if use_fused else dict()
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        use_fused = (device_type == 'cuda') and (
                'fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, **extra_args)

        return optimizer
