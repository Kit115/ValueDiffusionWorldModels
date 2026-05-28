from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .common import TransformerBlock, LayerNorm

@dataclass
class DiffusionDynamicsModelConfig:
    state_dim:              int     = 39 # states include encoded observations and proprioceptive information
    act_dim:                int     = 9

    transformer_dim:        int     = 256 
    num_heads:              int     = 8
    num_blocks:             int     = 6
    expansion_ratio:        float   = 2.0

    bias:                   bool    = False

    num_target_states:      int     = 5

    num_diff_steps:         int     = 1000

    diff_step_embed_dim:    int     = 32


class DiffusionDynamicsModel(nn.Module):
    def __init__(self, config: DiffusionDynamicsModelConfig):
        super().__init__()

        self.config = config

        csm_dim = config.transformer_dim - (config.state_dim)
        self.condition_state_marker = nn.Parameter(0.02 * torch.randn(1, csm_dim))

        tsm_dim = config.transformer_dim - (config.state_dim + config.act_dim + config.diff_step_embed_dim)
        self.target_state_marker    = nn.Parameter(0.02 * torch.randn(1, config.num_target_states, tsm_dim))

        self.diffusion_step_emb     = nn.Embedding(config.num_diff_steps, config.diff_step_embed_dim)

        self.layers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_blocks)],
            LayerNorm(config),
            nn.Linear(config.transformer_dim, config.state_dim, bias=config.bias),
        )


    def forward(self, current_state, actions, candidate_predictions, diffusion_step):
        # current_state:            (B, l)
        # actions:                  (B, t, a)
        # candidate_predictions:    (B, t, l)
        # diffusion_step:           (B,)
        
        B, T, _ = candidate_predictions.shape

        cond_toks = torch.cat((current_state, self.condition_state_marker.repeat(B, 1)), -1).unsqueeze(1)

        targ_toks = torch.cat((
            self.diffusion_step_emb(diffusion_step).unsqueeze(1).repeat(1, T, 1),
            self.target_state_marker.repeat(B, 1, 1),
            actions, 
            candidate_predictions
        ), -1)
 
        x = torch.cat((
            cond_toks,
            targ_toks
        ), 1)

        return self.layers(x)[:, 1:, :] # exclude the conditional token prediction

    def save_checkpoint(self, path: str):
        torch.save({
            "config":       asdict(self.config),
            "state_dict":   self.state_dict()
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device)

        config  = DiffusionWMConfig(**checkpoint["config"])
        model   = cls(config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @torch.no_grad()
    def sample_trajectory(
        model,
        current_state,
        actions,
        alphas_cumprod,
        num_steps=None,
        initial_x=None,
        initial_timestep=None,
        return_velocity_predictions=False,
    ):
        device = actions.device
        B, T, _ = actions.shape
        state_dim = model.config.state_dim

        N = alphas_cumprod.shape[0]

        if initial_timestep is None:
            initial_timestep = N - 1

        if not (0 <= initial_timestep < N):
            raise ValueError(
                f"initial_timestep must be in [0, {N - 1}], got {initial_timestep}"
            )

        start_t = int(initial_timestep)
        available_steps = start_t + 1  # e.g. 999 -> 1000 possible steps: 999..0

        # Build timestep schedule
        if num_steps is None or num_steps >= available_steps:
            # Full schedule from start_t down to 0
            timesteps = torch.arange(start_t, -1, -1, device=device, dtype=torch.long)
        else:
            # Reduced schedule from start_t downward, excluding 0 when num_steps > 1
            # Examples with start_t=999:
            #   num_steps=1 -> [999]
            #   num_steps=2 -> [999, 499]
            #   num_steps=3 -> [999, 666, 333]
            steps = torch.arange(num_steps, 0, -1, device=device, dtype=torch.long)
            timesteps = ((steps * available_steps + num_steps - 1) // num_steps) - 1

        # Initialize x
        if initial_x is None:
            x = torch.randn(B, T, state_dim, device=device)
        else:
            if initial_x.shape != (B, T, state_dim):
                raise ValueError(
                    f"initial_x must have shape {(B, T, state_dim)}, got {tuple(initial_x.shape)}"
                )
            x = initial_x.to(device)

        velocity_predictions = [] if return_velocity_predictions else None

        for idx, t in enumerate(timesteps):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

            v = model(
                current_state=current_state,
                actions=actions,
                candidate_predictions=x,
                diffusion_step=t_batch,
            )

            if return_velocity_predictions:
                velocity_predictions.append(v)

            a_t = alphas_cumprod[t]
            s = torch.sqrt(a_t)
            r = torch.sqrt(1.0 - a_t)

            s_ = s.view(1, 1, 1)
            r_ = r.view(1, 1, 1)

            x0 = s_ * x - r_ * v
            eps = r_ * x + s_ * v

            # Final inference step: return x0 directly
            if idx == len(timesteps) - 1:
                x = x0
            else:
                t_prev = timesteps[idx + 1]
                a_prev = alphas_cumprod[t_prev]
                s_prev = torch.sqrt(a_prev).view(1, 1, 1)
                r_prev = torch.sqrt(1.0 - a_prev).view(1, 1, 1)
                x = s_prev * x0 + r_prev * eps

        if return_velocity_predictions:
            velocity_predictions = torch.stack(velocity_predictions, dim=0)
            return x, velocity_predictions, timesteps

        return x




"""
    @torch.no_grad()
    def sample_trajectory(model, current_state, actions, alphas_cumprod, num_steps=None):
#        DDIM (eta=0) sampler for v-prediction diffusion.
        device = actions.device
        B, T, _ = actions.shape

        # allow fewer sampling steps by skipping
        N = alphas_cumprod.shape[0]
        if num_steps is None or num_steps >= N:
            timesteps = torch.arange(N-1, -1, -1, device=device)   # N-1 ... 0
        else:
            # uniform stride
            timesteps = torch.linspace(N-1, 0, num_steps, device=device).long()

        # start from noise at "max t"
        x = torch.randn(B, T, model.config.state_dim, device=device)
        
        ####### NEW SECTION #######
        if num_steps == 1:
            t = timesteps[0]
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

            v = model(
                current_state=current_state,
                actions=actions,
                candidate_predictions=x,
                diffusion_step=t_batch
            )

            a_t = alphas_cumprod[t]
            s = torch.sqrt(a_t).view(1, 1, 1)
            r = torch.sqrt(1.0 - a_t).view(1, 1, 1)

            x0 = s * x - r * v
            return x0
        ####### NEW SECTION #######

        for idx, t in enumerate(timesteps):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

            v = model(
                current_state=current_state,
                actions=actions,
                candidate_predictions=x,
                diffusion_step=t_batch
            )

            a_t = alphas_cumprod[t]                 # scalar tensor
            s = torch.sqrt(a_t)
            r = torch.sqrt(1.0 - a_t)

            # reshape for broadcasting over (B,T,state_dim)
            s_ = s.view(1, 1, 1)
            r_ = r.view(1, 1, 1)

            x0 = s_ * x - r_ * v                    # x0_pred
            eps = r_ * x + s_ * v                   # eps_pred

            if t.item() == 0:
#                print(torch.allclose(x, x0))
#                print(x - x0)
#                exit()
                x = x0
            else:
                # next step target alpha_bar
                t_prev = timesteps[idx + 1]
                a_prev = alphas_cumprod[t_prev]
                s_prev = torch.sqrt(a_prev).view(1, 1, 1)
                r_prev = torch.sqrt(1.0 - a_prev).view(1, 1, 1)
                x = s_prev * x0 + r_prev * eps      # DDIM eta=0 update

        return x



"""
