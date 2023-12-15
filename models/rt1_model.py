# imports
import math
import inspect
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F




@dataclass
class TOSILConfig:
    n_head: int
    n_embd: int
    block_size_en: int  # encoder block size
    block_size_de: int  # decoder block size
    n_tokens_per_frame: int = 8  # n_tokens per image feature
    feature_dim: int = 1280  # image feature dimension
    dropout: float = 0.0
    bias: bool = True
    n_layer: int = 12
    vocab_size: int = 100


# Transformer One-Shot Imitation Learning
class TOSIL(nn.Module):
    def __init__(self, config: TOSILConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size_en is not None
        assert config.block_size_de is not None
        self.config = config

        self.vocab_size = config.vocab_size
        self.n_tokens_per_frame = config.n_tokens_per_frame
        self.block_size_en = config.block_size_en * config.n_tokens_per_frame
        self.block_size_de = config.block_size_de * config.n_tokens_per_frame

        self.transformer = nn.ModuleDict(dict(
            # image encoder => get embeddings for image features, return n_tokens per image feature
            img_en=ImageFeatureEncoder(config=ImageFeatureEncoderConfig(bias=config.bias, dropout=config.dropout,
                                                                        feature_dim=config.feature_dim,
                                                                        n_embd=config.n_embd,
                                                                        n_tokens=config.n_tokens_per_frame)),
            # positional embeddings for encoder and decoder
            pos_emb_en=nn.Embedding(self.block_size_en, config.n_embd),
            pos_emb_de=nn.Embedding(self.block_size_de, config.n_embd),
            drop=nn.Dropout(config.dropout),
            # encoder and decoder blocks
            h_en=nn.ModuleList([AttentionEncoderBlock(config=AttentionEncoderBlockConfig(
                n_head=config.n_head,
                n_embd=config.n_embd,
                dropout=config.dropout,
                bias=config.bias,
                block_size=self.block_size_en,
            )) for _ in range(config.n_layer)]),
            h_de=nn.ModuleList([AttentionDecoderBlock(config=AttentionDecoderBlockConfig(
                n_head=config.n_head,
                n_embd=config.n_embd,
                dropout=config.dropout,
                bias=config.bias,
                block_size=self.block_size_de,
                non_causal_block_size=config.n_tokens_per_frame
            )) for _ in range(config.n_layer)]),
            # layer norm at the end of the encoder and decoder transformer
            ln_en=LayerNorm(config.n_embd, bias=config.bias),
            ln_de=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # last layer
        # multiply vocab size by 4 because we have 4 dimensions for each token
        self.lm_head = nn.Linear(config.n_embd, 4 * config.vocab_size, bias=False)

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

    def forward(self, key_frame_features, obs_frame_features, targets=None):
        device = key_frame_features.device

        # encode key frames
        tok_emb_en = self.transformer.img_en(key_frame_features)  # shape (b, num_key_frames * n_tokens, n_embd)
        b, t_en, c = tok_emb_en.size()
        pos_en = torch.arange(0, t_en, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        pos_emb_en = self.transformer.pos_emb_en(pos_en)  # shape (1, num_key_frames, n_embd)
        x = self.transformer.drop(tok_emb_en + pos_emb_en)
        for block in self.transformer.h_en:
            x = block(x)
        x = self.transformer.ln_en(x)  # shape (b, num_key_frames, n_embd)
        xkv = x

        # encode observed frames
        tok_emb_de = self.transformer.img_en(obs_frame_features)  # shape (b, num_obs_frames * n_tokens, n_embd)
        b, t_de, c = tok_emb_de.size()
        assert t_de <= self.block_size_de, f"Cannot forward sequence of length {t_de}, block size is only {self.block_size_de}"
        pos_de = torch.arange(0, t_de, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        pos_emb_de = self.transformer.pos_emb_de(pos_de)  # shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb_de + pos_emb_de)
        for block in self.transformer.h_de:
            x = block(x, xkv)
        x = self.transformer.ln_de(x)  # shape (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # TODO correctly, I need a loss per dimension => loss_pos_1, loss_pos_2, loss_pos_3, loss_torque and then sum them up
            logits = self.lm_head(x)[:, self.n_tokens_per_frame - 1::self.n_tokens_per_frame, :].reshape(-1,
                                                                                                         self.vocab_size)  # shape (b*config.block_size_en*4, vocab_size)
            loss = F.cross_entropy(logits, targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x)[:, [-1], :]  # shape (b, 1, 4*vocab_size)
            logits = logits.reshape(-1, 4, self.vocab_size)  # shape (b, 4, vocab_size)
            loss = None

        return logits, loss

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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
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
        # new PyTorch 2 has a new 'fused' option for AdamW that is much faster
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
    def generateNextIdx(self, key_frames, obs_frames, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of observed frames and key frames (LongTensor of shape (b,t)) and 
        generate a token.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        obs_frames = obs_frames if obs_frames.shape[1] <= self.config.block_size_de else obs_frames[:,
                                                                                         -self.config.block_size_de:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(key_frames, obs_frames)  # shape (b, 4, vocab_size)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits.view(-1, self.vocab_size) / temperature  # shape (b*4, vocab_size)
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, :, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # shape (b*4)
        return idx_next.view(-1, 4)
