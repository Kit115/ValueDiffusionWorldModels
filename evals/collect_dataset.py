import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TDMPC
from environment import construct_environment

import json
import numpy as np

from tqdm import tqdm
import os


def main(args):
    print(args)

    tdmpc, _ = TDMPC.from_checkpoint(f"experiments/{args.config_name}/{args.run_name}/latest.pth")

    env = construct_environment()

    save_dir    =f"experiments/{args.config_name}/{args.run_name}/datasets/"
    os.makedirs(save_dir, exist_ok=True)

    for dataset, dataset_size in zip(["training", "evaluation"], [args.num_train_trajs, args.num_val_trajs]):
        print(dataset, dataset_size)

        rgb_buffer = np.zeros((dataset_size, 601, 96, 96, 3), dtype=np.uint8)
        prop_buffer = np.zeros((dataset_size, 601, 7), dtype=np.float32)

        rgb_save_path   = f"{save_dir}/{dataset}_rgb.npy"
        prop_save_path   = f"{save_dir}/{dataset}_prop.npy"

        for sample_idx in tqdm(range(dataset_size)):
            obs, _ = env.reset()
            frame_counter = 0

            rgb_buffer[sample_idx, frame_counter, ...]  = obs["rgb"]
            prop_buffer[sample_idx, frame_counter, ...] = obs["proprios"]

            for _ in range(200):
                actions = tdmpc.act(obs)[0:3, ...].cpu().numpy()
                for action in actions:
                    obs, _, _, _, _ = env.step(action) 

                    frame_counter += 1
                    rgb_buffer[sample_idx, frame_counter, ...]  = obs["rgb"]
                    prop_buffer[sample_idx, frame_counter, ...] = obs["proprios"]

        np.save(rgb_save_path, rgb_buffer)
        np.save(prop_save_path, prop_buffer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-train-trajs",    type=int,   default=2500)
    parser.add_argument("--num-val-trajs",      type=int,   default=250)
    parser.add_argument("--dataset-name",       type=str,   default="default_dataset")
    parser.add_argument("--config-name",        type=str,   default="default_diffusion")
    parser.add_argument("--run-name",           type=str,   default="run_1")

    args = parser.parse_args()
    main(args)

