import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TDMPC
import numpy as np

import json
import random

import torch

from tqdm import tqdm

@torch.no_grad()
def calculate_td_errors(tdmpc, states, actions):
    cfg = tdmpc.cfg
    actions = actions.reshape(1, cfg.horizon, cfg.act_dim * cfg.act_chunks)
    gamma_wm = 0.99 ** cfg.act_chunks
    values  = torch.cat(tdmpc.model.vs(states), -1).min(-1).values.squeeze(0)
    rewards = tdmpc.model.reward(states[:, :-1, :], actions).squeeze(0)

    # signed one-step TD residuals: positive means V is optimistic relative to bootstrap target
    one_step_target = rewards + gamma_wm * values[1:]
    one_step_errors = one_step_target - values[:-1]

    # signed multi-step residual from t=0 to t=H
    return one_step_errors

def main():
    num_eval_samples = 10000
    rng = np.random.default_rng(seed=759)
    base_path   = "experiments"
    device      = "cuda:0"

    def evaluate(tdmpc, rgb_dataset, prop_dataset):
        ids = rng.choice(len(rgb_dataset), size=num_eval_samples, replace=True).tolist()
        one_step_errors = []

        for idx in tqdm(ids):
            obs = {
                "rgb":  rgb_dataset[idx, ...],
                "proprios": prop_dataset[idx, ...]
            }

            action_sequence = tdmpc.act(obs, noise_std=0.0).reshape(1, 5, 9)
            z = tdmpc.get_state_from_observation(obs)
            trajectory = tdmpc.model.sample_trajectory(z, action_sequence)

            os_err = calculate_td_errors(tdmpc, trajectory, action_sequence)
            one_step_errors.append(os_err)

        one_step_errors = torch.stack(one_step_errors, 0)  # [N, H]
        return one_step_errors

    def print_results(results):
        one_step_errors = results
        H = one_step_errors.shape[1]

        col_w = 12
        width = col_w * 5 + 4
        print("+" + "-" * width + "+")
        print(f"| {'t':>{col_w}} | {'mean(signed)':>{col_w}} | {'mean(|.|)':>{col_w}} | {'std':>{col_w}} | {'n':>{col_w}} |")
        print("+" + "-" * width + "+")
        for t in range(H):
            d = one_step_errors[:, t]
            print(f"| {t:>{col_w}d} | {d.mean().item():>{col_w}.4f} | {d.abs().mean().item():>{col_w}.4f} | {d.std().item():>{col_w}.4f} | {d.numel():>{col_w}d} |")
        print("+" + "-" * width + "+")

    diff_tdmpc, _   = TDMPC.from_checkpoint(f"{base_path}/default_diffusion/run_1/latest.pth")
    diff_rgb_dataset    = np.load(f"{base_path}/default_diffusion/run_1/datasets/evaluation_rgb.npy").reshape(-1, 96, 96, 3)
    diff_prop_dataset    = np.load(f"{base_path}/default_diffusion/run_1/datasets/evaluation_prop.npy").reshape(-1, 7)

    print("Evaluating Diffusion TDMPC...")
    diff_results = evaluate(diff_tdmpc, diff_rgb_dataset, diff_prop_dataset)
    print_results(diff_results)


if __name__ == "__main__":
    main()
