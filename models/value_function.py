import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass, asdict
from models.common import MLPBlock
#from common import MLPBlock
from functools import partial

@dataclass
class ValueModelConfig:
    ensemble_n:     int     = 2

    state_dim:      int     = 39

    hidden_dim:      int     = 512


class ValueModel(nn.Module):

    config_cls = ValueModelConfig 

    def __init__(self, config: ValueModelConfig):
        super().__init__()

        self.config = config
        
        self.models = nn.ModuleList([
            self._build_model(config) for _ in range(config.ensemble_n)
        ])

    def _build_model(self, config):
        
        activation_cls = partial(nn.GELU, approximate="tanh")

        return nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            activation_cls(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            activation_cls(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state):
        qs  = [model(state) for model in self.models]
        return qs



if __name__ == "__main__":
    q = ValueModel(ValueModelConfig())

    states = torch.randn(64, 39)

    v1, v2 = q(states)
    print(v1.shape)

