import torch
from torch import nn
from torch.nn import functional as F

from models.encoders       import StateEncoder
from models.flow_dynamics  import RectifiedFlowDynamicsModel, RectifiedFlowDynamicsModelConfig
from models.reward_models  import RewardModel, RewardModelConfig
from models.value_function import ValueModel, ValueModelConfig


def sample_rf_times(batch_size, device):
    # Basic rectified flow: uniform t in [0, 1]
    return torch.rand(batch_size, device=device)


def make_rectified_flow_training_pair(x0, t):
    """
    x0: (B, T, D) clean target trajectory
    t:  (B,) sampled in [0, 1]

    Returns:
        x_t:        interpolated point on straight path from noise -> data
        z:          sampled Gaussian noise
        u_target:   target velocity = x0 - z
    """
    B = x0.shape[0]
    z = torch.randn_like(x0)

    t_view = t.view(B, *([1] * (x0.ndim - 1)))  # (B,1,1) for broadcasting
    x_t = (1.0 - t_view) * z + t_view * x0
    u_target = x0 - z

    return x_t, z, u_target


def rf_predict_x0(x_t, u_pred, t):
    B = x_t.shape[0]
    t_view = t.view(B, *([1] * (x_t.ndim - 1)))
    return x_t + (1.0 - t_view) * u_pred


class FlowTOLD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder    = StateEncoder(
            latent_dim=cfg.state_representation_dim
        )

        self.dynamics   = RectifiedFlowDynamicsModel(RectifiedFlowDynamicsModelConfig(
            state_dim=cfg.state_representation_dim,
            num_target_states=cfg.horizon
        ))

        self.reward     = RewardModel(RewardModelConfig(
            state_embed_dim=cfg.state_representation_dim
        ))

        self.vs         = ValueModel(ValueModelConfig(
            state_dim=cfg.state_representation_dim
        ))

    def dynamics_loss(self, s_cur, s_next, actions):
        """
        Compute the rectified-flow matching loss and produce a one-step trajectory
        estimate of the latent states.

        s_cur:    (B, state_dim)
        s_next:   (B, horizon, state_dim)            ground-truth encoded next states
        actions:  (B, horizon, act_dim * act_chunks) horizon action chunks linking
                                                     s_cur to s_next

        Returns:
            loss:        scalar tensor
            trajectory:  (B, horizon+1, state_dim) with trajectory[:, 0, :] == s_cur
        """
        B = s_cur.shape[0]

        flow_t = sample_rf_times(B, device=s_cur.device)            # (B,)

        # straight-line interpolation between noise and target trajectory
        s_next_xt, _z, u_gt = make_rectified_flow_training_pair(s_next, flow_t)

        # predict velocity field u_theta(x_t, t | cond)
        u_pred = self.dynamics(
            current_state         = s_cur,
            actions               = actions,
            candidate_predictions = s_next_xt,
            flow_time             = flow_t,
        )

        loss = F.mse_loss(u_pred, u_gt)

        x0_pred    = rf_predict_x0(s_next_xt, u_pred, flow_t)
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
            num_steps = cfg.num_diff_steps
        )
        return torch.cat((start_state.unsqueeze(1), imagined_states), 1)



