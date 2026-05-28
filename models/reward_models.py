import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass

from .common import MLPBlock


@dataclass
class RewardModelConfig:
    state_embed_dim:        int     = 39
    act_dim:                int     = 9

    num_blocks:             int     = 2

    residual_dim:           int     = 256
    block_dim:              int     = 512

    use_norm:               bool    = True
    activation:             str     = "gelu"


class RewardModel(nn.Module):
    def __init__(self, config: RewardModelConfig):
        super().__init__()

        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.state_embed_dim + config.act_dim, config.residual_dim),
            *[MLPBlock(config.residual_dim, config.block_dim, activation=config.activation, use_norm=config.use_norm) for _ in range(config.num_blocks)],
            nn.Linear(config.residual_dim, 1)
        )

    def forward(self, s0, a):
        # s0:   (B, state_embed_dim)
        # a:    (B, act_dim)
        x = torch.cat((s0, a), -1)
        return self.layers(x).squeeze(-1)


