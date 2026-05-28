from dataclasses import dataclass, asdict

import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import TransformerBlock, LayerNorm


@dataclass
class RectifiedFlowDynamicsModelConfig:
    state_dim:              int     = 39
    act_dim:                int     = 9

    transformer_dim:        int     = 256
    num_heads:              int     = 8
    num_blocks:             int     = 6
    expansion_ratio:        float   = 2.0

    bias:                   bool    = False

    num_target_states:      int     = 5

    time_embed_dim:         int     = 32
    time_fourier_dim:       int     = 32

class ContinuousTimeEmbedding(nn.Module):
    """
    Sinusoidal/Fourier embedding for continuous t in [0, 1].
    """
    def __init__(self, out_dim: int, fourier_dim: int = 32):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.fourier_dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), half, device=t.device, dtype=t.dtype
            )
        )  # (half,)
        args = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.fourier_dim:
            emb = F.pad(emb, (0, self.fourier_dim - emb.shape[-1]))
        return self.proj(emb)


class RectifiedFlowDynamicsModel(nn.Module):
    def __init__(self, config: RectifiedFlowDynamicsModelConfig):
        super().__init__()
        self.config = config

        csm_dim = config.transformer_dim - config.state_dim
        self.condition_state_marker = nn.Parameter(0.02 * torch.randn(1, csm_dim))

        tsm_dim = config.transformer_dim - (
            config.state_dim + config.act_dim + config.time_embed_dim
        )
        self.target_state_marker = nn.Parameter(
            0.02 * torch.randn(1, config.num_target_states, tsm_dim)
        )

        self.time_emb = ContinuousTimeEmbedding(
            out_dim=config.time_embed_dim,
            fourier_dim=config.time_fourier_dim,
        )

        self.layers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_blocks)],
            LayerNorm(config),
            nn.Linear(config.transformer_dim, config.state_dim, bias=config.bias),
        )

    def forward(self, current_state, actions, candidate_predictions, flow_time):
        """
        current_state:          (B, state_dim)
        actions:                (B, T, act_dim)
        candidate_predictions:  (B, T, state_dim)   # x_t
        flow_time:              (B,) in [0, 1]
        """
        B, T, _ = candidate_predictions.shape

        cond_toks = torch.cat(
            (current_state, self.condition_state_marker.repeat(B, 1)), dim=-1
        ).unsqueeze(1)

        t_emb = self.time_emb(flow_time)[:, None, :].repeat(1, T, 1)

        targ_toks = torch.cat(
            (
                t_emb,
                self.target_state_marker.repeat(B, 1, 1),
                actions,
                candidate_predictions,
            ),
            dim=-1,
        )

        x = torch.cat((cond_toks, targ_toks), dim=1)

        # Predict flow velocity u_theta(x_t, t | cond)
        return self.layers(x)[:, 1:, :]

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        config = RectifiedFlowDynamicsModelConfig(**checkpoint["config"])
        model = cls(config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @torch.no_grad()
    def sample_trajectory(model, current_state, actions, num_steps=2):
        """
        Euler sampler for rectified flow.

        We use the convention:
            x_t = (1 - t) z + t x0
        so:
            t = 0 -> pure noise
            t = 1 -> clean sample

        The model predicts dx/dt = u_theta(x_t, t | cond).
        """
        device = actions.device
        B, T, _ = actions.shape

        # start from noise at t=0
        x = torch.randn(B, T, model.config.state_dim, device=device)

        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = ts[i]
            dt = ts[i + 1] - ts[i]

            t_batch = torch.full((B,), t.item(), device=device)
            u = model(
                current_state=current_state,
                actions=actions,
                candidate_predictions=x,
                flow_time=t_batch,
            )

            x = x + dt * u

        return x
