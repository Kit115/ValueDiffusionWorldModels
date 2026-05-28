from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import math

from functools import partial


class LayerNorm(nn.Module):
    def __init__(self, config): 
        super().__init__() 
        self.weight = nn.Parameter(torch.ones(config.transformer_dim)) 
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5) 

class SelfAttention(nn.Module): 
    def __init__(self, config, is_causal=None, attention_mask=None): 
        assert config.transformer_dim % config.num_heads == 0
        assert is_causal is not None or attention_mask is not None
        super().__init__()

        self.is_causal      = is_causal
        if attention_mask is not None:
            self.register_buffer("attention_mask", attention_mask)
        else:
            self.attention_mask = None

        self.num_heads = config.num_heads
        self.transformer_dim = config.transformer_dim

        self.qkv    = nn.Linear(config.transformer_dim, 3*config.transformer_dim, bias=config.bias)
        self.proj   = nn.Linear(config.transformer_dim, config.transformer_dim, bias=config.bias)
    
    def forward(self, X): 
        B, T, C = X.shape
        q, k, v = self.qkv(X).split(self.transformer_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        if self.attention_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attention_mask[:, :, :T, :T])
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)


        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class FeedForward(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        ffwd_dim = int(config.expansion_ratio * config.transformer_dim)
        self.c_fc       = nn.Linear(config.transformer_dim, ffwd_dim, bias=config.bias) 
        self.c_proj     = nn.Linear(ffwd_dim, config.transformer_dim, bias=config.bias) 

    def forward(self, X): 
        X = self.c_fc(X) 
        X = F.gelu(X, approximate="tanh") 
        X = self.c_proj(X) 
        return X 

 


class TransformerBlock(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__() 
        self.ln_1   = LayerNorm(config) 
        self.attn   = SelfAttention(config, is_causal=is_causal) 
        self.ln_2   = LayerNorm(config) 
        self.mlp    = FeedForward(config) 

    def forward(self, X): 
        X = X + self.attn(self.ln_1(X)) 
        X = X + self.mlp(self.ln_2(X)) 
        return X 


class MLPBlock(nn.Module):
    def __init__(self, residual_dim, block_dim, activation="gelu", use_norm=True):
        super().__init__()

        activation_cls = {
            "relu": nn.ReLU,
            "gelu": partial(nn.GELU, approximate="tanh")
        }[activation]

        self.layers = nn.Sequential(
            nn.LayerNorm(residual_dim) if use_norm else nn.Identity(),
            nn.Linear(residual_dim, block_dim),
            activation_cls(),
            nn.Linear(block_dim, residual_dim)
        )
    def forward(self, x):
        return x + self.layers(x)
