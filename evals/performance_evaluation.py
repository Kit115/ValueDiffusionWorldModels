import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TDMPC
from environment import construct_environment

import numpy as np

import json

def main(args):
    num_eval_seeds = 100
    seed_gen = np.random.default_rng(seed=12345)
    eval_seeds = seed_gen.integers(low=0, high=2**31 - 1, size=num_eval_seeds, dtype=np.int64).tolist()
    
    tdmpc, _ = TDMPC.from_checkpoint(f"experiments/{args.config_name}/{args.run_name}/latest.pth")
    
    # this is such jank omg
    for setting, value in args.modifications.items():
        exec(f"tdmpc.cfg.{setting} = {value}")
    print(tdmpc.cfg)

    env = construct_environment(render_mode="human")
    replan_every = 3

    results = {
        "save_path": f"experiments/{args.config_name}/{args.run_name}/{args.eval_name}.json",
        "modifications": args.modifications,
        "episode_returns": []
    }

    with open(results["save_path"], "w+") as f:
        f.write("Hello!")


    for idx, eval_seed in enumerate(eval_seeds):
        obs, _ = env.reset(seed=eval_seed)

        episode_return = 0.

        for step in range(600//replan_every):
            best_sequence = tdmpc.act(obs)

            actions = best_sequence[0:replan_every, ...].cpu().numpy()

            for action in actions:
                obs, reward, _, _, _ = env.step(action) 
                episode_return += reward

        results["episode_returns"].append(episode_return)
        print(f"{idx:03d}: Avg: {sum(results['episode_returns'])/len(results['episode_returns']):.2f}; Latest: {episode_return:.2f}")

    with open(results["save_path"], "w+") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-name", default="default_performance_evaluation")
    parser.add_argument("--modifications", type=json.loads, default="{}")

    args = parser.parse_args()
    main(args)


