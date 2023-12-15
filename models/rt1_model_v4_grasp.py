# imports
import math
import inspect
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.rt1_model import LayerNorm, AttentionEncoderBlock, AttentionEncoderBlockConfig, TokenLearnerAttentionMLP, TokenLearnerAttentionMLPConfig, ImageFeatureEncoder, ImageFeatureEncoderConfig


@dataclass
class TOSILConfig:
    n_head: int
    n_embd: int
    block_size_key: int # encoder block size
    block_size_obs: int # decoder block size
    n_tokens_per_frame: int = 8 # n_tokens per image feature
    feature_dim: int = 1280 # image feature dimension
    dropout: float = 0.0
    bias: bool = True
    n_layer: int = 12
    vocab_size: int = 100

class TOSIL(nn.Module):
    def __init__(self, config: TOSILConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size_key is not None
        assert config.block_size_obs is not None
        self.config = config

        self.vocab_size = config.vocab_size
        self.n_tokens_per_frame = config.n_tokens_per_frame
        self.block_size = (config.block_size_key + config.block_size_obs) * config.n_tokens_per_frame + 1

        self.transformer = nn.ModuleDict(dict(
            # image encoder => get embeddings for image features, return n_tokens per image feature
            img_en=ImageFeatureEncoder(config=ImageFeatureEncoderConfig(bias=config.bias, dropout=config.dropout, feature_dim=config.feature_dim, n_embd=config.n_embd, n_tokens=config.n_tokens_per_frame)),
            sep_emb=nn.Embedding(1, config.n_embd), # separator embedding
            # positional embeddings for encoder
            pos_emb=nn.Embedding(self.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            # encoder blocks
            h_en=nn.ModuleList([AttentionEncoderBlock(config=AttentionEncoderBlockConfig(
                n_head=config.n_head,
                n_embd=config.n_embd,
                dropout=config.dropout,
                bias=config.bias,
                block_size=self.block_size,
            )) for _ in range(config.n_layer)]),
            # layer norm at the end of the encoder and decoder transformer
            ln_en=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # last layer
        # predict action
        self.lm_head = nn.Linear(config.n_embd, 6, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the number of image encoder parameters is not counted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # TODO maybe subtract number of weights of image encoder
            pass
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # TODO add velocities
    def forward(self, key_frame_features: torch.Tensor, obs_frame_features: torch.Tensor, target_pos: torch.Tensor = None):
        """
        args:
            key_frame_features: shape (b, num_key_frames, c, h, w)
            obs_frame_features: shape (b, num_obs_frames, c, h, w)
            target_pos: shape (b, 6): target_pos[:, :3] = target grasp positions; target_pos[:, 3:] = goal positions
        """
        device = key_frame_features.device
        
        # encode key frames
        b, t_key, _, _, _ = key_frame_features.size()
        n_key_tokens = t_key * self.config.n_tokens_per_frame
        
        # get embeddings
        x = torch.concatenate([key_frame_features, obs_frame_features], dim=1) # shape (b, num_key_frames + num_obs_frames, c, h, w)
        x = self.transformer.img_en(x) # shape (b, (num_key_frames + num_obs_frames) * n_tokens, n_embd)
        x = torch.concatenate([
            x[:,:n_key_tokens,:], 
            self.transformer.sep_emb(torch.zeros(b, 1, dtype=torch.long, device=device)), 
            x[:,n_key_tokens:,:]
        ], dim=1) # shape (b, block_size, n_embd)

        # TODO add velocities to embeddings

        
        # add pos embeddings
        pidx = torch.arange(0, self.block_size, dtype=torch.long, device=device).unsqueeze(0) # shape (1, block_size)
        pos_emb = self.transformer.pos_emb(pidx) # shape (1, block_size, n_embd)
        x = self.transformer.drop(x + pos_emb) # shape (b, block_size, n_embd)

        # multi-head self attention
        for block in self.transformer.h_en:
            x = block(x)

        # layer norm
        x = self.transformer.ln_en(x) # shape (b, block_size, n_embd)

        # actions
        x = x[:,-1,:] # shape (b, n_embd), we only need the last token
        x = self.lm_head(x) # shape (b, 6)

        if target_pos is not None:
            # loss angle
            # x_pos = x[:,:3]
            # x_vel = torch.norm(x_pos, dim=1, keepdim=True) # shape (b, 1)
            # actions_pos = actions[:,-1,:3]
            # actions_vel = torch.norm(actions_pos, dim=1, keepdim=True) # shape (b, 1)
            # loss_angle = 1 - torch.sum(x_pos * actions_pos / (x_vel * actions_vel + 1e-7), dim=1) # shape (b,)
            # loss_angle = torch.mean(loss_angle) # shape (1,)
            
            # # loss speed
            # loss_speed = F.mse_loss(torch.clamp(x_vel, -1, 1), torch.clamp(actions_vel, -1, 1))
            
            # # loss grasp
            # loss_grasp = F.mse_loss(torch.clamp(x[:,3], -1, 1), torch.clamp(actions[:,-1,3], -1, 1))

            # # total loss
            # # NOTE max of loss_speed and loss_grasp is 4 while max of loss_angle is 2 so we multiply loss_angle by 2
            # loss = 2 * loss_angle + loss_speed + loss_grasp

            # # if we are given some desired targets also calculate the loss
            loss = F.mse_loss(100 * x, 100 * target_pos)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            loss = None

        return x, loss

    def configure_optimizers(self, weight_decay, learning_rate, beta1, beta2, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        betas = (beta1, beta2)

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,torch.nn.Conv2d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and (
            'fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    # TODO adapt this to the new model
    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    #     flops_per_token = 6 * N + 12 * L * H * Q * T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    #     flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu


    @torch.no_grad()
    def generate_next_action(self, key_frame_features, obs_frame_features):
        """
        Take a conditioning sequence of observed frames and key frames (LongTensor of shape (b,t)) and 
        generate a token.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert key_frame_features.shape[1] == self.config.block_size_key, "key_frames must be of length block_size_key"
        assert obs_frame_features.shape[1] == self.config.block_size_obs, "obs_frames must be of length block_size_obs"

        # forward the model to get the logits for the index in the sequence
        actions, _ = self(key_frame_features, obs_frame_features) # shape (b, 6)

        return actions