import torch.nn as nn
import torch
from torch.nn import functional as F
from typing import Callable

from dataclasses import dataclass
import math
import models.model_utils as model_utils


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# https://github.com/mlfoundations/open_clip/blob/24ddefb37fc4892f6a0c975b732226fe8a9a8613/src/open_clip/transformer.py#L22
class LayerNormOpenClip(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

# Attention building blocks

@dataclass
class AttentionMLPConfig:
    n_embd: int
    bias: bool = True
    dropout: float = 0.0


class AttentionMLP(nn.Module):

    def __init__(self, config: AttentionMLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


@dataclass
class AttentionMHSAConfig:
    n_head: int
    n_embd: int
    dropout: float = 0.0
    causal: bool = False
    non_causal_block_size: int = 1
    bias: bool = True
    block_size: int = 8  # unfortunate naming, block_size is simply the max number of tokens over which we perform self attention


class AttentionMHSA(nn.Module):

    def __init__(self, config: AttentionMHSAConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal

        # construct layers

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # construct attention mask
        assert config.block_size % config.non_causal_block_size == 0
        attn_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        for i in range(config.block_size // config.non_causal_block_size):
            b1 = i * config.non_causal_block_size
            b2 = (i + 1) * config.non_causal_block_size
            attn_mask[b1:b2, b1:b2] = 1
        attn_mask = attn_mask.view(1, 1, config.block_size, config.block_size)
        attn_mask.requires_grad = False

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(
            torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if self.flash:
            self.register_buffer("attn_mask", attn_mask == 1)
        else:
            print(
                "WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # TODO why is this called bias?? => rename
            self.register_buffer("bias", attn_mask)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) d
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask[:, :, :T, :T],
                                                                 dropout_p=self.dropout)
        else:
            # manual implementation of attention
            #
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if (self.causal):
                # bias is defined in constructor: if not self.flash: ...
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            # normalize
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


@dataclass
class AttentionMHAPointCloudConfig:
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


class AttentionMHAPointCloud(nn.Module):
    """
    Attention over points in a point cloud. That is, V = (batch_size * n_frames, n_points, 3) and keys K = (batch_size * n_frames, n_points, n_embd // n_head) are the same for all heads
    """

    def __init__(self, config: AttentionMHAPointCloudConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # construct layers

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.ModuleDict(dict(
            q=nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        ))

        # output projection
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        # self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(
            torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self, xq, k, v):
        """
        args:
            xq: (B * n_frames, 1 = TQ, n_embd)
            xk: (B * n_frames, n_points = TK, n_embd // n_head = hs)
            xv: (B * n_frames, n_points = TV, 3)
        """
        B, TQ, C = xq.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        B, n_points, _ = k.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn.q(xq)
        q = q.view(B, TQ, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, TQ, hs)
        k = torch.unsqueeze(k, 1).repeat(1, self.n_head, 1, 1)  # (B, nh, n_points, hs)
        v = torch.unsqueeze(v, 1).repeat(1, self.n_head, 1, 1)  # (B, nh, n_points, 3)

        # attention: q x k.T = (B, nh, TQ, hs) x (B, nh, hs, TKV) -> (B, nh, TQ, TKV)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # normalize
            att = F.softmax(att, dim=-1)
            # att = self.attn_dropout(att)
            y = att @ v  # (B, nh, TQ, n_points) x (B, nh, n_points, 3) -> (B, nh, TQ, 3)

        # re-assemble all head outputs side by side
        y = y.view(B, self.n_head * 3)  # shape: (B, n_head * 3)

        return y


@dataclass
class AttentionMHAConfig:
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


class AttentionMHA(nn.Module):

    def __init__(self, config: AttentionMHAConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # construct layers

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.ModuleDict(dict(
            q=nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            k=nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            v=nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        ))

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(
            torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self, xq, xkv):
        B, TQ, C = xq.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        B, TKV, C = xkv.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn.q(xq), self.c_attn.k(xkv), self.c_attn.v(xkv)
        q = q.view(B, TQ, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, TQ, hs)
        k = k.view(B, TKV, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, TKV, hs)

        v = v.view(B, TKV, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, TKV, hs)

        # attention: q x k.T = (B, nh, TQ, hs) x (B, nh, hs, TKV) -> (B, nh, TQ, TKV)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # normalize
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, TQ, TKV) x (B, nh, TKV, hs) -> (B, nh, TQ, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, TQ, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


@dataclass
class AttentionEncoderBlockConfig:
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    block_size: int = 8


class AttentionEncoderBlock(nn.Module):

    def __init__(self, config: AttentionEncoderBlockConfig):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.attn = AttentionMHSA(
            config=AttentionMHSAConfig(bias=config.bias, block_size=config.block_size, causal=False,
                                       dropout=config.dropout, n_embd=config.n_embd, n_head=config.n_head))
        self.ln_2 = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.mlp = AttentionMLP(
            config=AttentionMLPConfig(bias=config.bias, dropout=config.dropout, n_embd=config.n_embd))

    def forward(self, x):
        # NOTE in the original paper: x = self.ln_1(x + self.attn(x))
        # However, according to Anrej Karpathy (see YouTube video about GPT), now it is more common to apply layer norm first
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class AttentionDecoderBlockConfig:
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    block_size: int = 8
    non_causal_block_size: int = 1
    # since decoder block consists of two MHA blocks, we might have different number of heads.
    # For now we keep everything the same between these two MHA blocks.


class AttentionDecoderBlock(nn.Module):
    def __init__(self, config: AttentionDecoderBlockConfig):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.attn_1 = AttentionMHSA(
            config=AttentionMHSAConfig(causal=True, non_causal_block_size=config.non_causal_block_size,
                                       bias=config.bias, block_size=config.block_size, dropout=config.dropout,
                                       n_embd=config.n_embd, n_head=config.n_head))
        self.ln_2 = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.attn_2 = AttentionMHA(
            config=AttentionMHAConfig(n_head=config.n_head, bias=config.bias, dropout=config.dropout,
                                      n_embd=config.n_embd))
        self.ln_3 = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.mlp = AttentionMLP(
            config=AttentionMLPConfig(bias=config.bias, dropout=config.dropout, n_embd=config.n_embd))

    def forward(self, x, xkv):
        """forward
        xkv: normalized input
        """
        # NOTE in the original paper: x = self.ln_1(x + self.attn(x))
        # However, according to Anrej Karpathy (see YouTube video about GPT), now it is more common to apply layer norm first
        x = x + self.attn_1(self.ln_1(x))
        x = x + self.attn_2(self.ln_2(x),
                            xkv)  # xkv should be normalized before calling forward() of AttentionDecoderBlock
        x = x + self.mlp(self.ln_3(x))
        return x


@dataclass
class TokenLearnerAttentionMLPConfig:
    n_embd: int
    n_tokens: int
    bias: bool = True
    dropout: float = 0.0


class TokenLearnerAttentionMLP(nn.Module):

    def __init__(self, config: TokenLearnerAttentionMLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 8 * config.n_tokens, bias=config.bias)
        self.c_proj = nn.Linear(8 * config.n_tokens, config.n_tokens, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


@dataclass
class ImageFeatureEncoderConfig:
    feature_dim: int  # = in_channels
    n_embd: int  # = out_channels
    n_tokens: int
    dropout: float = 0.0
    bias: bool = True


class ImageFeatureEncoder(nn.Module):
    """ImageFeatureEncoder encodes image features into a sequence of tokens.
    """

    def __init__(self, config: ImageFeatureEncoderConfig) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_tokens = config.n_tokens

        self.ln = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.conv1x1 = nn.Conv2d(config.feature_dim, self.n_embd, 1)
        self.mlp = TokenLearnerAttentionMLP(config=TokenLearnerAttentionMLPConfig(
            n_embd=self.n_embd, n_tokens=self.n_tokens, bias=config.bias, dropout=config.dropout))

    def forward(self, x):
        b, t, c, h, w = x.shape
        # TODO maybe normalize x before passing it to conv1x1
        # TODO apply layernorm
        x = x.view(b * t, c, h, w)  # shape: (b*t, c, h, w)
        x = self.conv1x1(x)  # shape: (b*t, n_embd, h, w)
        x = x.view(b * t, self.n_embd, h * w).transpose(1, 2)  # shape: (b*t, h*w, n_embd)
        att = self.ln(x)
        att = self.mlp(att)  # shape: (b*t, h*w, n_tokens)
        att = att.transpose(1, 2)  # shape: (b*t, n_tokens, h*w)
        att = F.softmax(att, dim=-1)
        x = att @ x  # shape: (b*t, n_tokens, n_embd)
        x = x.view(b, t * self.n_tokens, self.n_embd)  # shape: (b, t*n_tokens, n_embd)
        return x


@dataclass
class ImageFeatureDecoderConfig:
    feature_dim: int  # = in_channels
    n_embd: int  # = out_channels
    n_tokens: int
    dropout: float = 0.0
    bias: bool = True


class ImageFeatureDecoder(nn.Module):
    """ImageFeatureDecoder decodes image features into a larger dimensional vector (B, ).
    """


class Conv2dBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size: tuple[int, int] = (3, 3), stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)
    

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def geom_dist_attention(q, k, v, mask=None):
    d_k = q.size()[-1]
    pass


class MultiheadAttention(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        

class MultiheadAttentionGeomDist(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o


class LinearTokenizer(nn.Module):

    def __init__(self, input_dim, token_dim, n_tokens):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self.proj = nn.Linear(input_dim, token_dim*n_tokens)

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.fill_(0)

    def forward(self, x):
        b, s, c = x.shape
        x = self.proj(x)  # shape: [Batch, SeqLen, n_tokens*token_dim]
        x = x.reshape(b, s*self.n_tokens, self.token_dim)  # shape: [Batch, SeqLen*n_tokens, token_dim]
        return x

        

def choose_topk(A, V, k):
    b, h, s, h_n = V.shape
    A_topk, topk_indices = torch.topk(A, k, dim=-1)
    V_idx = topk_indices[:, :, :, :, None].repeat(1, 1, 1, 1, h_n)  # shape (b, h, s, k, h_n)
    V_topk = V[:, :, None, :, :].repeat(1, 1, s, 1, 1)  # shape (b, h, s, s, h_n)
    V_topk = torch.gather(input=V_topk, dim=-2, index=V_idx)  # shape (b, h, s, k, h_n)
    return A_topk, V_topk


def deep_attention(A_topk, V_topk, model):
    """ No longer needed
    """
    b, h, s, k, h_n = V_topk.shape
    A_topk = A_topk[:, :, :, :, None].repeat(1, 1, 1, 1, h_n)  # shape (b, h, s, k, h_n)
    x = A_topk * V_topk  # shape (b, h, s, k, h_n)
    x = x.view(b, h, s, k*h_n)
    x = model(x)
    return x


class DeepMultiheadAttention(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    
    def __init__(self, input_dim, embed_dim, num_heads, k=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.dropout = dropout
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Sequential(
            # No attention needed 
            nn.Linear(self.embed_dim*k, self.embed_dim*(k // 2)),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim*(k // 2), self.embed_dim)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)  # shape: [Batch, SeqLen, 3*embed_dim]
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # shape: [Batch, Head, SeqLen, head_dim]
        
        # Determine value outputs
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # shape: [Batch, Head, SeqLen, SeqLen]
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1) # shape: [Batch, Head, SeqLen, SeqLen]

        # Deep attention
        # TODO if gradient is an issue, we can combine with traditional attention to give hints
        A_topk, V_topk = choose_topk(attention, v, self.k)
        b, h, s, k, h_n = V_topk.shape
        A_topk = A_topk[:, :, :, :, None].repeat(1, 1, 1, 1, h_n)  # shape (b, h, s, k, h_n)
        values = A_topk * V_topk  # shape (b, h, s, k, h_n)
        values = values.view(b, h, s, k*h_n)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, k*h_n]
        values = values.reshape(batch_size, seq_length, self.k * self.embed_dim)
        o = self.o_proj(values)  # shape: [Batch, SeqLen, embed_dim]
        
        if return_attention:
            return o, attention
        else:
            return o

        # # Deep attention
        # # NOTE reshape is row-major ([[1,2], [3, 4]] => [1, 2, 3, 4])
        # v = v.reshape(batch_size, self.num_heads, seq_length * self.head_dim)
        # attention = attention.repeat_interleave(self.head_dim, dim=-1) # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # # NOTE element-wise multiplication instead of matrix multiplication
        # # apply dense NN to each head

        # v = attention * v  # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # attnention_top_k, top_k_indices = torch.topk(attention, self.top_k * seq_length, dim=-1)

        
        # # compute statistics
        # values_weighted_mean = attention * v  # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # values_weighted_mean = values_weighted_mean.reshape(batch_size, self.num_heads, seq_length, seq_length, self.head_dim) # shape: [Batch, Head, SeqLen, SeqLen, head_dim]
        # values_weighted_mean = torch.sum(values_weighted_mean, dim=-2) # shape: [Batch, Head, SeqLen, 1, head_dim]
        # values_weighted_mean = values_weighted_mean.reshape(batch_size, self.num_heads, seq_length * self.head_dim)  # shape: [Batch, Head, SeqLen*head_dim]
        
        # values_var = (v - values_weighted_mean)  # shape: [Batch, Head, SeqLen*head_dim]
        # values_var = attention * values_var
        # values_var = values_var.reshape(batch_size, self.num_heads, seq_length, seq_length, self.head_dim)
        # values_var = torch.sum(values_var, dim=-2, keepdim=True)  # shape: [Batch, Head, SeqLen, 1, head_dim]
        # values_var = values_var.reshape(batch_size, self.num_heads, seq_length * self.head_dim)  # shape: [Batch, Head, SeqLen*head_dim]

        # # kernel
        # x1 = attention * torch.ones_like(v)  # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # x2 = attention * v  # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # x3 = attention * v**2  # shape: [Batch, Head, SeqLen, SeqLen*head_dim]
        # x1 = x1.reshape(batch_size, self.num_heads, seq_length, seq_length, self.head_dim) # shape: [Batch, Head, SeqLen, SeqLen, head_dim]
        # x2 = x2.reshape(batch_size, self.num_heads, seq_length, seq_length, self.head_dim) # shape: [Batch, Head, SeqLen, SeqLen, head_dim]
        # x3 = x3.reshape(batch_size, self.num_heads, seq_length, seq_length, self.head_dim) # shape: [Batch, Head, SeqLen, SeqLen, head_dim]
        # x = torch.concat([x1, x2, x3], dim=-1) # shape: [Batch, Head, Seqlen, Seqlen, head_dim*3]
        # x_t = torch.transpose(x, -1, -2) # shape: [Batch, Head, Seqlen, head_dim*3, Seqlen]
        # x = torch.matmul(x, x_t) # shape: [Batch, Head, Seqlen, head_dim*3, head_dim*3]


class DeepEncoderBlock(nn.Module):
    """
    This class implements a single encoder block of the Transformer model.
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, input_dim, num_heads, dim_feedforward, k=8, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = DeepMultiheadAttention(input_dim, input_dim, num_heads, k=k, dropout=dropout)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class DeepTransformerEncoder(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DeepEncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    # def get_attention_maps(self, x, mask=None):
    #     attention_maps = []
    #     for l in self.layers:
    #         _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
    #         attention_maps.append(attn_map)
    #         x = l(x)
    #     return attention_maps


class EncoderBlock(nn.Module):
    """
    This class implements a single encoder block of the Transformer model.
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # # Attention layer
        # # self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        # self.self_attn = nn.MultiheadAttention(
        #     input_dim, 
        #     num_heads, 
        #     dropout=0.0, 
        #     bias=True, 
        #     add_bias_kv=False, 
        #     add_zero_attn=False, 
        #     kdim=None, 
        #     vdim=None, 
        #     batch_first=True, 
        #     device=None, 
        #     dtype=None
        # )

        # # Two-layer MLP
        # self.linear_net = nn.Sequential(
        #     nn.Linear(input_dim, dim_feedforward),
        #     nn.Dropout(dropout),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_feedforward, input_dim)
        # )

        # # Layers to apply in between the main layers
        # self.norm1 = nn.LayerNorm(input_dim)
        # self.norm2 = nn.LayerNorm(input_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        return self.encoder_layer(x)
        # # Attention part
        # # attn_out = self.self_attn(x, mask=mask)
        # attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        # x = x + self.dropout(attn_out)
        # x = self.norm1(x)

        # # MLP part
        # linear_out = self.linear_net(x)
        # x = x + self.dropout(linear_out)
        # x = self.norm2(x)

        # return x

class TransformerEncoder(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    # def get_attention_maps(self, x, mask=None):
    #     attention_maps = []
    #     for l in self.layers:
    #         _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
    #         attention_maps.append(attn_map)
    #         x = l(x)
    #     return attention_maps


class PositionalEncoding(nn.Module):
    """
    Copied from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    

class AttentionBlockPreLN(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        self.patch_size = patch_size
        
        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlockPreLN(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))
    
    
    def forward(self, x):
        # Preprocess input
        x = model_utils.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        
        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out





# https://github.com/mlfoundations/open_clip/blob/24ddefb37fc4892f6a0c975b732226fe8a9a8613/src/open_clip/transformer.py#L163
class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNormOpenClip
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, L, D) tensor where N is the batch size, L is the sequence length, and D is the embedding dimension.
        """
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)