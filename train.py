from buffer import Buffer
from environment import construct_environment

from models.encoders            import StateEncoder
from models.diffusion_dynamics  import DiffusionDynamicsModel, DiffusionDynamicsModelConfig
from models.reward_models       import RewardModel, RewardModelConfig
from models.value_function      import ValueModel, ValueModelConfig
from models.common              import MLPBlock

from told_flow import FlowTOLD
from told_diffusion import DiffusionTOLD
from told_mlp import MLPTOLD

import torch
from torch import nn
import lejepa
from tensordict import TensorDict

from dataclasses import dataclass, asdict
from copy import deepcopy


@dataclass
class TDMPCConfig:
    device: str = "cuda:0"

    # which dynamics backbone to use
    dynamics_type: str = "diffusion"   # "flow", "diffusion" or "mlp"
    diff_steps: int = 1000             # only used by DiffusionTOLD

    # state
    state_representation_dim: int = 64

    # general
    act_chunks: int  = 3
    act_dim: int     = 3
    gamma: float     = 0.99

    # planning
    pop_size: int        = 512
    n_elite: int         = 64
    n_iters: int         = 10
    horizon: int         = 5
    cem_std_floor: float = 1e-6
    num_diff_steps: int  = 1           # only used by FlowTOLD and DiffusionTOLD

    # training
    lr: float             = 3e-4
    polyak: float         = 0.995
    grad_clip_norm: float = 10.0

    # loss calculation
    lam: float                 = 0.05
    detach_reward_inputs: bool = False
    rew_coef: float            = 0.01
    detach_v_inputs: bool      = False
    v_coef: float              = 0.01

    # signature-regularisation (lejepa)
    sigreg_n_points: int   = 17
    sigreg_num_slices: int = 1024


@torch.no_grad()
def ema_update(model, target, polyak=0.995):
    for p, p_targ in zip(model.parameters(), target.parameters()):
        p_targ.data.mul_(polyak)
        p_targ.data.add_((1 - polyak) * p.data)


# ---------------------------------------------------------------------------
# Diffusion helpers
# ---------------------------------------------------------------------------


TOLD_CLASSES = {
    "diffusion": DiffusionTOLD,
    "mlp":       MLPTOLD,
    "flow":     FlowTOLD
}


# ---------------------------------------------------------------------------
# TDMPC (agnostic to the chosen TOLD backbone)
# ---------------------------------------------------------------------------

class TDMPC():
    def __init__(self, cfg: TDMPCConfig):
        self.cfg = cfg
        self.device = cfg.device

        if cfg.dynamics_type not in TOLD_CLASSES:
            raise ValueError(
                f"Unknown dynamics_type '{cfg.dynamics_type}'. "
                f"Expected one of {list(TOLD_CLASSES)}."
            )
        told_cls    = TOLD_CLASSES[cfg.dynamics_type]
        self.model  = told_cls(cfg).to(cfg.device)
        self.target = deepcopy(self.model).to(cfg.device)

        self.optim  = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.univariate_test     = lejepa.univariate.EppsPulley(n_points=cfg.sigreg_n_points)
        self.sigreg_loss_fn      = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test = self.univariate_test,
            num_slices      = cfg.sigreg_num_slices
        )

        gamma_wm        = cfg.gamma ** cfg.act_chunks
        self.discounts  = gamma_wm ** torch.arange(cfg.horizon+1, device=self.device, dtype=torch.float).unsqueeze(0)

    @torch.no_grad()
    def evaluate_trajectories(self, start_state, action_sequences):
        cfg = self.cfg
        action_sequences = torch.tanh(action_sequences)
        action_sequences = action_sequences.reshape(cfg.pop_size, cfg.horizon, cfg.act_dim * cfg.act_chunks)

        trajectories       = self.model.sample_trajectory(start_state, action_sequences)
        trajectory_rewards = self.model.reward(trajectories[:, :-1, :], action_sequences)

        bootstrap_state     = trajectories[:, -1, :]
        bootstrap_val_1, bootstrap_val_2 = self.model.vs(bootstrap_state)
        bootstrap_val, _ = torch.min(torch.cat([bootstrap_val_1, bootstrap_val_2], -1), dim=-1, keepdim=True)

        trajectory_returns = (self.discounts * torch.cat((trajectory_rewards, bootstrap_val), -1)).sum(-1)
        return -trajectory_returns, trajectories

    @torch.no_grad()
    def plan(self, state, noise_std=0.0, return_everything=False):
        cfg = self.cfg
        start_state = state.repeat(cfg.pop_size, 1)

        mean = torch.zeros(cfg.horizon * cfg.act_chunks, cfg.act_dim, device=self.device)
        std  = torch.ones_like(mean, device=self.device)

        best_action_sequence = None
        best_cost = torch.tensor(torch.inf, device=self.device)

        if return_everything:
            everything = []

        for _ in range(cfg.n_iters):

            sample_mean = mean.expand(cfg.pop_size, cfg.horizon*cfg.act_chunks, cfg.act_dim)
            sample_std  = std.expand_as(sample_mean)

            samples = torch.normal(sample_mean, sample_std)

            costs, trajectories = self.evaluate_trajectories(start_state, samples)

            if return_everything:
                everything.append({
                    "actions":      torch.tanh(samples),
                    "trajectories": trajectories,
                    "costs":        costs
                })

            min_cost, min_idx = costs.min(dim=0)

            if min_cost < best_cost:
                best_cost               = min_cost
                best_action_sequence    = samples[min_idx].clone()

            elite_idx = torch.topk(costs, k=cfg.n_elite, largest=False).indices
            elites = samples[elite_idx]

            mean = elites.mean(dim=0)
            std  = elites.std(dim=0) + cfg.cem_std_floor


        acts = torch.tanh(noise_std*torch.randn_like(best_action_sequence) + best_action_sequence)

        if return_everything:
            return acts, everything

        return acts

    @torch.no_grad()
    def get_state_from_observation(self, obs):
        # obs: dict from the env with
        #   "rgb":      (96, 96, 3) uint8
        #   "proprios": (7,)        float
        obs_td = TensorDict({
            "rgb":      torch.as_tensor(obs["rgb"]).unsqueeze(0).to(self.device),
            "proprios": torch.as_tensor(obs["proprios"]).float().unsqueeze(0).to(self.device),
        }, batch_size=[1])
        return self.model.encoder(obs_td)

    def act(self, obs, noise_std=0.0, return_everything=False):
        state = self.get_state_from_observation(obs)
        if not return_everything:
            return self.plan(state, noise_std=noise_std)
        return self.plan(state, noise_std=noise_std, return_everything=True)


    def save_checkpoint(self, path, **extra):
        torch.save({
            "cfg":      asdict(self.cfg),
            "model":    self.model.state_dict(),
            "target":   self.target.state_dict(),
            "optim":    self.optim.state_dict(),
            **extra,
        }, path)

    @staticmethod
    def from_checkpoint(path, device=None):
        ckpt = torch.load(path, map_location=device)

        cfg = TDMPCConfig(**ckpt["cfg"])
        if device is not None:
            cfg.device = device

        tdmpc = TDMPC(cfg)
        tdmpc.model.load_state_dict(ckpt["model"])
        tdmpc.target.load_state_dict(ckpt["target"])
        tdmpc.optim.load_state_dict(ckpt["optim"])

        return tdmpc, ckpt

    def reward_loss(self, s, a, r):
        cfg = self.cfg
        discounts = cfg.gamma ** torch.arange(r.shape[-1], device=r.device).reshape(1, 1, r.shape[-1])
        targets   = (discounts * r).sum(-1)

        if cfg.detach_reward_inputs:
            s, a = s.detach(), a.detach()

        pred = self.model.reward(s, a)
        loss = ((targets - pred) ** 2).mean()
        return loss

    def v_loss(self, s0, obs, r):
        # s0:   (B, horizon+1, state_dim)
        # obs:  TensorDict of batch_size (B, num_frames) with "rgb" and "proprios"
        # r:    (B, horizon+1, act_chunks)
        cfg = self.cfg

        if cfg.detach_v_inputs:
            s0 = s0.detach()

        # create TD Target
        with torch.no_grad():
            obs_next = obs[:, 1:]                                       # batch_size (B, T)
            s1 = self.model.encoder(obs_next)                          # (B, T, state_dim)

            v_est_1, _ = torch.min(torch.cat(self.target.vs(s1), -1), -1)

            discounts = cfg.gamma ** torch.arange(r.shape[-1], device=r.device).reshape(1, 1, r.shape[-1])
            r_targ    = (discounts * r).sum(-1)

            v_target = r_targ + (cfg.gamma ** cfg.act_chunks) * v_est_1

        v1, v2 = self.model.vs(s0)
        v1, v2 = v1.squeeze(-1), v2.squeeze(-1)

        v1_loss = ((v1 - v_target) ** 2).mean()
        v2_loss = ((v2 - v_target) ** 2).mean()
        v_loss  = 0.5 * (v1_loss + v2_loss)

        return v_loss

    def update(self, batch):
        # batch: TensorDict of batch_size (B,) with
        #   "obs":     TensorDict (B, num_frames) of "rgb" and "proprios"
        #   "actions": (B, num_frames-1, act_dim * act_chunks)
        #   "rewards": (B, num_frames-1, act_chunks)
        #   "dones":   (B, num_frames-1)
        cfg = self.cfg
        batch = batch.to(cfg.device)

        obs = batch["obs"]
        act = batch["actions"]
        rew = batch["rewards"]

        # Split observations temporally: first frame is "current", middle frames
        # are the targets for the dynamics model, last frame is consumed by v_loss.
        obs_cur  = obs[:, 0]            # batch_size (B,)
        obs_next = obs[:, 1:-1]         # batch_size (B, num_frames-2)

        s_cur = self.model.encoder(obs_cur)                             # (B, state_dim)

        with torch.no_grad():
            # the dynamics target is always produced by the target encoder
            s_next = self.target.encoder(obs_next)                      # (B, num_frames-2, state_dim)

        dyn_loss, trajectory = self.model.dynamics_loss(s_cur, s_next, act[:, :-1, :])

        sigreg_loss = self.sigreg_loss_fn(s_cur)

        reward_loss = self.reward_loss(trajectory, act, rew)
        value_loss  = self.v_loss(trajectory, obs, rew)

        loss    = ((1 - cfg.lam) * (dyn_loss + (cfg.rew_coef * reward_loss) + (cfg.v_coef * value_loss))) + (cfg.lam * sigreg_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip_norm)

        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        ema_update(self.model, self.target, cfg.polyak)

        return {
            "total_loss":    loss.item(),
            "dynamics_loss": dyn_loss.item(),
            "sigreg_loss":   sigreg_loss.item(),
            "reward_loss":   reward_loss.item()
        }


@torch.no_grad()
def collect_trajectory(tdmpc, noise_std=0.0, num_steps=600):
    cfg = tdmpc.cfg
    replan_every = cfg.act_chunks
    assert (num_steps % replan_every) == 0

    env = construct_environment()
    obs, _ = env.reset()
    done, trunc = False, False

    rgbs        = [obs["rgb"]]
    proprios    = [obs["proprios"]]
    actions     = []
    rewards     = []
    dones       = []

    for _ in range(num_steps//replan_every):
        best_actions = tdmpc.act(obs, noise_std=noise_std)[:replan_every, :].cpu().numpy()

        for a in best_actions:
            assert a.shape == (cfg.act_dim,)

            obs, r, done, trunc, _ = env.step(a)   # typically a should be converted to env action boundaries via "to_env_actions"

            rgbs.append(obs["rgb"])
            proprios.append(obs["proprios"])
            actions.append(a)
            rewards.append(r)
            dones.append(done)

            if done or trunc:
                break
        if done or trunc:
            break

    return rgbs, proprios, actions, rewards, dones


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from experiment import load_config, save_config, setup_run_dir, write_log

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str,
                        help="path to a YAML config (new run) or to a run directory (with --resume)")
    parser.add_argument("--resume", action="store_true",
                        help="treat `config` as an existing run directory and continue it")
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.config)
        train_cfg, algo_cfg = load_config(run_dir / "config.yaml", TDMPCConfig)

        print("Loading Algorithm..")
        tdmpc, ckpt = TDMPC.from_checkpoint(run_dir / "latest.pth", device=algo_cfg.device)
        print("Loading Buffer..")
        buffer = Buffer(
            num_trajectories  = train_cfg.buffer_capacity,
            trajectory_length = train_cfg.trajectory_length,
            frame_skip        = algo_cfg.act_chunks,
            num_frames        = algo_cfg.horizon + 2,
        )
        buffer.load(run_dir / "buffer.npz")

        trajectory_returns = list(ckpt.get("trajectory_returns", []))
        start_epoch        = int(ckpt.get("epoch", -1)) + 1
    else:
        train_cfg, algo_cfg = load_config(args.config, TDMPCConfig)
        run_dir = setup_run_dir(args.config)
        save_config(run_dir / "config.yaml", train_cfg, algo_cfg)

        tdmpc  = TDMPC(algo_cfg)
        buffer = Buffer(
            num_trajectories  = train_cfg.buffer_capacity,
            trajectory_length = train_cfg.trajectory_length,
            frame_skip        = algo_cfg.act_chunks,
            num_frames        = algo_cfg.horizon + 2,
        )

        trajectory_returns = []
        start_epoch        = -train_cfg.seed_trajectories

    print(f"Run directory: {run_dir}")
    print(algo_cfg)
    print(f"Starting Training at Epoch #{start_epoch}")

    def save(epoch):
        extra = {"epoch": epoch, "trajectory_returns": trajectory_returns}
        tdmpc.save_checkpoint(run_dir / "latest.pth",       **extra)
        tdmpc.save_checkpoint(run_dir / f"{epoch:04d}.pth", **extra)
        buffer.save(run_dir / "buffer.npz")
        write_log(run_dir / "log.csv", trajectory_returns, train_cfg.log_window_size)

    for epoch in range(start_epoch, train_cfg.num_epochs):
        # collect a trajectory and add it to the buffer
        noise_std = train_cfg.exploration_std(epoch)
        o, p, a, r, d = collect_trajectory(tdmpc, noise_std=noise_std, num_steps=train_cfg.trajectory_length)
        buffer.append(o, p, a, r, d)
        trajectory_returns.append(float(sum(r)))

        # live progress (the durable record is log.csv, rewritten at each checkpoint)
        window = trajectory_returns[-train_cfg.log_window_size:]
        avg    = sum(window) / len(window)
        print(f"\rEpoch {epoch}: avg_return={avg:.2f}; last_return={trajectory_returns[-1]:.2f}", end="        ")

        if epoch < 0:
            continue

        # update the world model
        for _ in range(train_cfg.updates_per_epoch):
            batch = buffer.get_batch(batch_size=train_cfg.batch_size)
            tdmpc.update(batch)

        if epoch % train_cfg.checkpoint_interval == 0:
            save(epoch)

    tdmpc.save_checkpoint(run_dir / "final.pth") 

