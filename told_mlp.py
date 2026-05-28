import torch
from torch import nn

from models.encoders            import StateEncoder
from models.reward_models       import RewardModel, RewardModelConfig
from models.value_function      import ValueModel, ValueModelConfig
from models.common              import MLPBlock


class MLPDynamics(nn.Module):
    def __init__(self, state_dim=64, act_dim=9):
        super().__init__()
        self.state_dim  = state_dim
        self.act_dim    = act_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim + act_dim, 384),
            *[MLPBlock(384, 768) for _ in range(6)],
            nn.Linear(384, state_dim)
        )

    def forward(self, state, action):
        # state:    (B, ..., state_dim)
        # action:   (B, ..., act_dim)
        x = torch.cat((state, action), -1)
        return state + self.layers(x)


class MLPTOLD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder    = StateEncoder(latent_dim=cfg.state_representation_dim)
        self.dynamics   = MLPDynamics(state_dim=cfg.state_representation_dim)
        self.reward     = RewardModel(RewardModelConfig(
            state_embed_dim=cfg.state_representation_dim
        ))
        self.vs         = ValueModel(ValueModelConfig(
            state_dim=cfg.state_representation_dim
        ))

    def _rollout(self, start_state, actions):
        """
        Autoregressively roll out the deterministic dynamics.

        start_state: (B, state_dim)
        actions:     (B, T, act_dim * act_chunks)

        Returns trajectory of shape (B, T+1, state_dim) where trajectory[:, 0, :]
        is start_state.
        """
        predicted_states = [start_state.unsqueeze(1)]
        T = actions.shape[1]
        for i in range(T):
            current_state = predicted_states[-1]
            action = actions[:, i:i+1, :]
            predicted_states.append(self.dynamics(current_state, action))
        return torch.cat(predicted_states, 1)

    def dynamics_loss(self, s_cur, s_next, actions):
        """
        Compute a one-step MSE prediction loss against the encoded next states.

        s_cur:    (B, state_dim)
        s_next:   (B, horizon, state_dim)
        actions:  (B, horizon, act_dim * act_chunks)

        Returns:
            loss:        scalar tensor
            trajectory:  (B, horizon+1, state_dim) with trajectory[:, 0, :] == s_cur
        """
        trajectory = self._rollout(s_cur, actions)
        x0_pred    = trajectory[:, 1:, :]
        loss       = ((x0_pred - s_next) ** 2).mean()
        return loss, trajectory

    @torch.no_grad()
    def sample_trajectory(self, start_state, actions):
        """
        start_state: (B, state_dim)
        actions:     (B, horizon, act_dim * act_chunks)

        Returns trajectory of shape (B, horizon+1, state_dim).
        """
        return self._rollout(start_state, actions)


