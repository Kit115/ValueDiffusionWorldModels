from train import TDMPC

import gymnasium as gym
import numpy as np

from environment import construct_environment

tdmpc, _ = TDMPC.from_checkpoint("experiments/default_diffusion/run_1/latest.pth")

env = construct_environment(render_mode="human")

replan_every = 3

seed = 1234
step = 0
action_history = []
if True:
    obs, _ = env.reset(seed=seed)

    for step in range(600//replan_every):
        best_sequence = tdmpc.act(obs, noise_std=0.0)

        actions = best_sequence[0:replan_every, ...].cpu().numpy()

        for action in actions:
            action_history.append(action)
            obs, reward, _, _, _ = env.step(action)
        print(step)

action_history = np.stack(action_history, 0)
#print(action_history.shape)
#
#np.save(f"rollouts/track_{seed}.npy", action_history)
 


