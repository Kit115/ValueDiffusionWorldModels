import torch
from torch import nn

from models.encoders            import StateEncoder
from models.diffusion_dynamics  import DiffusionDynamicsModel, DiffusionDynamicsModelConfig
from models.reward_models       import RewardModel, RewardModelConfig
from models.value_function      import ValueModel, ValueModelConfig


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)  # [T]
    alphas = 1.0 - betas                             # [T]
    alphas_cumprod = torch.cumprod(alphas, dim=0)    # [T]
    return alphas_cumprod


def add_noise(x0, t, alphas_cumprod):
    B = x0.shape[0]

    alpha_bar_t = alphas_cumprod[t]  # [B]
    alpha_bar_t = alpha_bar_t.view(B, *([1] * (x0.ndim - 1)))

    eps = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
    return x_t, eps, alpha_bar_t


# ---------------------------------------------------------------------------
# Task-Oriented Latent Dynamics Models
# ---------------------------------------------------------------------------

class DiffusionTOLD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder    = StateEncoder(
            latent_dim=cfg.state_representation_dim
        )

        self.dynamics   = DiffusionDynamicsModel(DiffusionDynamicsModelConfig(
            state_dim=cfg.state_representation_dim,
            num_target_states=cfg.horizon
        ))

        self.reward     = RewardModel(RewardModelConfig(
            state_embed_dim=cfg.state_representation_dim
        ))

        self.vs         = ValueModel(ValueModelConfig(
            state_dim=cfg.state_representation_dim
        ))

        # registered as a buffer so it moves with .to(device) and is saved in
        # the model's state_dict.
        self.register_buffer(
            "alphas_cumprod",
            make_beta_schedule(cfg.diff_steps)
        )

    def dynamics_loss(self, s_cur, s_next, actions):
        """
        Compute the diffusion v-prediction loss and produce a one-step trajectory
        estimate of the latent states.

        s_cur:    (B, state_dim)
        s_next:   (B, horizon, state_dim)            ground-truth encoded next states
        actions:  (B, horizon, act_dim * act_chunks) horizon action chunks linking
                                                     s_cur to s_next

        Returns:
            loss:        scalar tensor
            trajectory:  (B, horizon+1, state_dim) with trajectory[:, 0, :] == s_cur
        """
        cfg = self.cfg
        B   = s_cur.shape[0]

        diffusion_step = torch.randint(cfg.diff_steps, (B,), device=s_cur.device)
        s_next_noisy, noise_gt, alpha_bar_t = add_noise(
            s_next,
            diffusion_step,
            alphas_cumprod=self.alphas_cumprod
        )

        sqrt_alpha_bar_t      = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha  = torch.sqrt(1.0 - alpha_bar_t)
        v_gt = sqrt_alpha_bar_t * noise_gt - sqrt_one_minus_alpha * s_next

        v_pred = self.dynamics(
            current_state           = s_cur,
            actions                 = actions,
            candidate_predictions   = s_next_noisy,
            diffusion_step          = diffusion_step
        )

        loss = ((v_pred - v_gt) ** 2).mean()

        x0_pred    = sqrt_alpha_bar_t * s_next_noisy - sqrt_one_minus_alpha * v_pred
        trajectory = torch.cat((s_cur.unsqueeze(1), x0_pred), 1)
        return loss, trajectory

    @torch.no_grad()
    def sample_trajectory(self, start_state, actions):
        """
        start_state: (B, state_dim)
        actions:     (B, horizon, act_dim * act_chunks)

        Returns trajectory of shape (B, horizon+1, state_dim).
        """
        cfg = self.cfg
        imagined_states = self.dynamics.sample_trajectory(
            start_state, actions,
            alphas_cumprod = self.alphas_cumprod,
            num_steps      = cfg.num_diff_steps
        )
        return torch.cat((start_state.unsqueeze(1), imagined_states), 1)


