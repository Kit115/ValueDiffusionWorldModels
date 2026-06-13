import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TDMPC
from environment import construct_environment
import numpy as np

import random

import torch

from tqdm import tqdm

@torch.no_grad()
def eval_obs(tdmpc, current_frame, action_sequence, n_diff_steps=1):
    # returns the value function's opinion of the current state and its opinion
    # of every predicted state along the imagined trajectory (shape [H+1])
    cfg = tdmpc.cfg
    z = tdmpc.get_state_from_observation(current_frame)
    action_sequence = action_sequence.reshape(1, cfg.horizon, cfg.act_dim * cfg.act_chunks)
    trajectory = tdmpc.model.sample_trajectory(z, action_sequence)
    v_all = torch.cat(tdmpc.model.vs(trajectory[:, :, :]), -1).min(-1).values.squeeze(0)  # [H+1]
    v_0 = v_all[0]
    return v_0, v_all


@torch.no_grad()
def rollout(tdmpc, seed):
    env     = construct_environment()
    obs, _  = env.reset(seed=seed)

    H = tdmpc.cfg.horizon       # horizon in WM steps, not env steps
    C = tdmpc.cfg.act_chunks    # WM steps * Act Chunks = Env Steps

    v_0s, v_alls = [], []

    for step in range(200):
        actions = tdmpc.act(obs, noise_std=0.0)

        v_0, v_all = eval_obs(tdmpc, obs, actions)
        v_0s.append(v_0.cpu())
        v_alls.append(v_all.cpu())

        a_exec  = actions[:C, ...].cpu().numpy()
        for action in a_exec:
            obs, _, _, _, _ = env.step(action)

    v_0, v_all = eval_obs(tdmpc, obs, actions)
    v_0s.append(v_0.cpu())
    v_alls.append(v_all.cpu())

    v_0s   = torch.tensor(v_0s)              # [T]
    v_alls = torch.stack(v_alls, 0)          # [T, H+1]

    # for each offset k in 1..H: compare V(real state at t+k) with V(imagined state at offset k, predicted from t)
    diffs_per_k = []
    for k in range(1, H + 1):
        diffs_per_k.append(v_0s[k:] - v_alls[:-k, k])   # length T+1-k
    return diffs_per_k   # list of length H, each entry a 1-D tensor

def main():
    num_eval_seeds = 100
    seed_gen = np.random.default_rng(seed=12345)
    eval_seeds = seed_gen.integers(low=0, high=2**31 - 1, size=num_eval_seeds, dtype=np.int64).tolist()
    base_path   = "experiments/"
    device      = "cuda:0"
 
    def evaluate(tdmpc):
        H = tdmpc.cfg.horizon
        per_k_values = [[] for _ in range(H)]
        for seed in tqdm(eval_seeds):
            diffs_per_k = rollout(tdmpc, seed)
            for k_idx, d in enumerate(diffs_per_k):
                per_k_values[k_idx].append(d)
        per_k_flat = [torch.cat(vs, 0) for vs in per_k_values]
        return per_k_flat
 
    def print_results(results):
        col_w = 12
        print("+" + "-" * (col_w * 5 + 4) + "+")
        print(f"| {'k':>{col_w}} | {'mean(signed)':>{col_w}} | {'mean(|.|)':>{col_w}} | {'std':>{col_w}} | {'n':>{col_w}} |")
        print("+" + "-" * (col_w * 5 + 4) + "+")
        for k_idx, d in enumerate(results, start=1):
            print(f"| {k_idx:>{col_w}d} | {d.mean().item():>{col_w}.4f} | {d.abs().mean().item():>{col_w}.4f} | {d.std().item():>{col_w}.4f} | {d.numel():>{col_w}d} |")
        print("+" + "-" * (col_w * 5 + 4) + "+")
 
    diff_tdmpc, _  = TDMPC.from_checkpoint(f"{base_path}/default_diffusion/run_1/latest.pth")
    print("Evaluating Diffusion TDMPC...")
    diff_results = evaluate(diff_tdmpc)
    print_results(diff_results)

    mlp_tdmpc, _  = TDMPC.from_checkpoint(f"{base_path}/default_mlp/run_1/latest.pth")
    print("Evaluating MLP TDMPC...")
    mlp_results = evaluate(mlp_tdmpc)
    print_results(mlp_results)
 
 
if __name__ == "__main__":
    main()

