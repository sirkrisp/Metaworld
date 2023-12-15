# imports
import math
import inspect
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.rt1_model import LayerNorm, AttentionEncoderBlock, AttentionEncoderBlockConfig, TokenLearnerAttentionMLP, TokenLearnerAttentionMLPConfig, ImageFeatureEncoder, ImageFeatureEncoderConfig, AttentionMHAPointCloud, AttentionMHAPointCloudConfig
from models.film_decoder_b0 import FilmDecoderB0


@dataclass
class TOSILConfig:
    n_head: int
    n_embd: int
    n_point_clusters: int
    block_size_key: int # encoder block size
    block_size_obs: int # decoder block size
    n_tokens_per_frame: int = 8 # n_tokens per image feature
    num_depth_pts: int = 4096 # n_tokens per depth frame
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
        self.n_frames = config.block_size_key + config.block_size_obs
        self.block_size = self.n_frames * config.n_tokens_per_frame + 1 # +1 for separator token

        self.transformer = nn.ModuleDict(dict(
            # image encoder => get embeddings for image features, return n_tokens per image feature
            img_en=ImageFeatureEncoder(
                config=ImageFeatureEncoderConfig(
                    bias=config.bias, 
                    dropout=config.dropout, 
                    feature_dim=config.feature_dim, 
                    n_embd=config.n_embd, 
                    n_tokens=config.n_tokens_per_frame
                )
            ),
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

        # TODO pass in decoder config
        # TODO probably we want more than one decoder
        self.decoder = FilmDecoderB0(config.n_point_clusters, config.n_embd)
        # self.img_en_depth = ImageFeatureEncoder(config=ImageFeatureEncoderConfig(
        #     bias=config.bias,
        #     dropout=config.dropout,
        #     feature_dim=config.feature_dim,
        #     n_embd=config.n_embd // config.n_head,
        #     n_tokens=config.num_depth_pts
        # ))
        # self.mha_point_cloud = AttentionMHAPointCloud(config=AttentionMHAPointCloudConfig(
        #     n_head=config.n_head,
        #     n_embd=config.n_embd,
        #     bias=config.bias,
        #     dropout=config.dropout,
        # ))

        # last layer
        # predict action
        # self.lm_head = nn.Linear(config.n_embd, 4, bias=False)

        n_final = self.n_frames * 3 + self.n_frames * config.n_point_clusters * 3
        print("n_final", n_final)
        self.action_prediction_head = nn.Sequential(
            nn.Linear(n_final, n_final, bias=config.bias),
            nn.ReLU(),
            LayerNorm(n_final, bias=config.bias),
            nn.Linear(n_final, n_final, bias=config.bias),
            nn.ReLU(),
            LayerNorm(n_final, bias=config.bias),
            nn.Linear(n_final, 4)
            # TODO apply tanh because output values are between -1 and 1?
        )

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

    def forward(
            self, 
            key_frame_features: torch.Tensor, 
            key_point_clouds: torch.Tensor, 
            key_end_effector_positions: torch.Tensor, 
            obs_frame_features: torch.Tensor,
            obs_point_clouds: torch.Tensor,
            obs_end_effector_positions: torch.Tensor,
            actions: torch.Tensor = None
        ):
        """
        args:
            key_frame_features: shape (b, num_key_frames, c, h, w)
            key_point_clouds: shape (b, num_key_frames, 3, 4096)
            key_end_effector_positions: shape (b, num_key_frames, 3)
            obs_frame_features: shape (b, num_obs_frames, c, h, w)
            obs_point_clouds: shape (b, num_obs_frames, 3, 4096)
            obs_end_effector_positions: shape (b, num_obs_frames, 3)
            actions: shape (b, num_obs_frames, 4)
        """
        device = key_frame_features.device
        
        # encode key frames
        b, n_key_frames, c_en, h_en, w_en = key_frame_features.size()
        _, n_obs_frames, _, _, _ = obs_frame_features.size()
        n_frames = n_key_frames + n_obs_frames
        n_key_tokens = n_key_frames * self.config.n_tokens_per_frame
        _, _, _, n_points = key_point_clouds.size()
        
        # get embeddings
        x_img_features = torch.concatenate([key_frame_features, obs_frame_features], dim=1) # shape (b, num_key_frames + num_obs_frames, c, h, w)
        x = self.transformer.img_en(x_img_features) # shape (b, (num_key_frames + num_obs_frames) * n_tokens, n_embd)
        x = torch.concatenate([
            x[:,:n_key_tokens,:], 
            self.transformer.sep_emb(torch.zeros(b, 1, dtype=torch.long, device=device)), 
            x[:,n_key_tokens:,:]
        ], dim=1) # shape (b, block_size, n_embd)
        
        # add pos embeddings
        pidx = torch.arange(0, self.block_size, dtype=torch.long, device=device).unsqueeze(0) # shape (1, block_size)
        pos_emb = self.transformer.pos_emb(pidx) # shape (1, block_size, n_embd)
        x = self.transformer.drop(x + pos_emb) # shape (b, block_size, n_embd)

        # multi-head self attention
        for block in self.transformer.h_en:
            x = block(x)

        # layer norm
        x = self.transformer.ln_en(x) # shape (b, block_size, n_embd)

        # compute attention over depth map (points in point cloud)
        x_img_features = x_img_features.view((-1, *x_img_features.shape[2:])) # shape (b * n_frames, c, h, w)
        # conditioning is the same for all frames
        conditioning = x[:, 0, :] # shape (b, n_embd)
        conditioning = conditioning.repeat(n_frames, 1) # shape (b * n_frames, n_embd)
        point_cluster_attn = self.decoder(x_img_features, conditioning) # shape (b * n_frames, n_point_clusters, 64, 64)
        point_cluster_attn = point_cluster_attn.view((b, n_frames, self.config.n_point_clusters, 4096)) # shape (b, n_frames, n_point_clusters, 4096)
        point_cluster_attn = F.softmax(point_cluster_attn, dim=-1) # shape (b, n_frames, n_point_clusters, 4096)

        # compute point clusters
        point_clusters = torch.concatenate([key_point_clouds, obs_point_clouds], dim=1) # shape (b, n_frames, 3, 4096)
        point_clusters = point_clusters.transpose(2, 3) # shape (b, n_frames, 4096, 3)
        point_clusters = torch.matmul(point_cluster_attn, point_clusters) # shape (b, n_frames, n_point_clusters, 3)

        # # generate key,values for depth map
        # x = x[:, 0, :] # shape (b, n_embd)
        # # query is the same for all frame
        # d_q = x.repeat(n_frames, 1) # shape (b * n_frames, n_embd)
        # d_q = torch.unsqueeze(d_q, dim=1) # shape (b * n_frames, 1, n_embd)

        # x_img_features = x_img_features.view((-1, *x_img_features.shape[2:])) # shape (b * n_frames, c, h, w)
        # x_img_features = torch.unsqueeze(x_img_features, dim=1) # shape (b * n_frames, 1, c, h, w)
        # d_key = self.img_en_depth(x_img_features) # shape (b * n_frames, num_depth_pts, n_embd // n_head)

        # d_value = torch.concatenate([key_point_clouds, obs_point_clouds], dim=1) # shape (b, num_key_frames + num_obs_frames, 3, num_depth_pts)
        # d_value = d_value.view((-1, *d_value.shape[2:])) # shape (b * n_frames, 3, num_depth_pts)
        # d_value = torch.moveaxis(d_value, -1, -2) # shape (b * n_frames, num_depth_pts, 3)

        # point_clusters = self.mha_point_cloud(d_q, d_key, d_value) # shape (b * n_frames, n_head * 3)

        # generate final vector that goes into action prediction head
        # NOTE we do not really need the "query vector" from the transformer as input to the action prediction head, 
        # since the relationship between clusters should be enough to determine the next action
        x = torch.cat([
            # x, 
            key_end_effector_positions.view((b, -1)), 
            obs_end_effector_positions.view((b, -1)),
            point_clusters.view((b, -1)), 
        ], dim=1) # shape (b, n_embd + n_frames * 3 + n_frames * n_head * 3)
        # print("action_prediction_head input shape", x.shape)
        x = self.action_prediction_head(x) # shape (b, 4)

        # # actions
        # x = self.lm_head(x) # shape (b, block_size, 4)
        # x = x[:,-1,:] # shape (b, 4), we only need the last token

        if actions is not None:
            # # loss angle
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
            loss = F.mse_loss(torch.clamp(x, -1, 1), torch.clamp(actions[:,-1,:], -1, 1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            loss = None

        return x, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,torch.nn.Conv2d, torch.nn.ConvTranspose2d)
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
    def generate_next_action(
        self, 
        key_frame_features: torch.Tensor, 
        key_point_clouds: torch.Tensor, 
        key_end_effector_positions: torch.Tensor, 
        obs_frame_features: torch.Tensor,
        obs_point_clouds: torch.Tensor,
        obs_end_effector_positions: torch.Tensor,
    ):
        """
        Take a conditioning sequence of observed frames and key frames (LongTensor of shape (b,t)) and 
        generate a token.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert key_frame_features.shape[1] == self.config.block_size_key, "key_frames must be of length block_size_key"
        assert obs_frame_features.shape[1] == self.config.block_size_obs, "obs_frames must be of length block_size_obs"

        # forward the model to get the logits for the index in the sequence
        actions, _ = self(key_frame_features, key_point_clouds, key_end_effector_positions, obs_frame_features, obs_point_clouds, obs_end_effector_positions) # shape (b, 4)

        return actions