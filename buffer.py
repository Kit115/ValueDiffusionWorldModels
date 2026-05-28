import numpy as np
import torch
from tensordict import TensorDict


class Buffer():
    def __init__(self, num_trajectories=1000, trajectory_length=600, frame_skip=3, num_frames=6):
        self.num_trajectories   = num_trajectories
        self.trajectory_length  = trajectory_length
        self.frame_skip         = frame_skip
        self.num_frames         = num_frames

        self.observations   = np.zeros((num_trajectories, trajectory_length+1, 96, 96, 3),  dtype=np.uint8)
        self.proprios       = np.zeros((num_trajectories, trajectory_length+1, 7),          dtype=np.float32)
        self.actions        = np.zeros((num_trajectories, trajectory_length, 3),            dtype=np.float32)
        self.rewards        = np.zeros((num_trajectories, trajectory_length),               dtype=np.float32)
        self.dones          = np.zeros((num_trajectories, trajectory_length),               dtype=np.float32)
        self.traj_lengths   = np.zeros((num_trajectories,), dtype=np.int32)

        self.ptr    = 0
        self.size   = 0


    def append(self, observations, proprios, actions, rewards, dones):
        assert all(isinstance(arg, list) for arg in (observations, proprios, actions, rewards, dones)), "Expected type 'list' for arguments 'observations', 'proprios', 'actions' and 'rewards'."

        traj_length = len(observations) - 1

        observations    = np.array(observations)
        proprios        = np.array(proprios)
        actions         = np.array(actions)
        rewards         = np.array(rewards)
        dones           = np.array(dones).astype(np.float32)

        assert observations.dtype   == np.uint8, "List elements of observations must be of type np.uint8"

        self.observations[self.ptr, :traj_length+1, ...] = observations
        self.proprios[self.ptr,     :traj_length+1, ...] = proprios
        self.actions[self.ptr,      :traj_length,   ...] = actions
        self.rewards[self.ptr,      :traj_length,   ...] = rewards
        self.dones[self.ptr,        :traj_length,   ...] = dones
        self.traj_lengths[self.ptr]         = traj_length


        self.ptr    = (self.ptr + 1) % self.num_trajectories
        self.size   = min(self.size + 1, self.num_trajectories)


    def get_item(self, trajectory_idx, start, end):
        observations    = self.observations[trajectory_idx, start:end+1:self.frame_skip, ...]
        proprios        = self.proprios[trajectory_idx, start:end+1:self.frame_skip, ...]
        actions         = self.actions[trajectory_idx, start:end, ...].reshape(self.num_frames - 1, -1)
        rewards         = self.rewards[trajectory_idx, start:end].reshape(self.num_frames - 1, -1)
        dones           = self.dones[trajectory_idx, start:end].reshape(self.num_frames - 1, -1).sum(-1).clip(min=0.0, max=1.0)
        return observations, proprios, actions, rewards, dones


    def get_batch(self, batch_size=256):
        trajectory_ids = np.random.randint(0, self.size, size=(batch_size,))

        max_start = self.traj_lengths[trajectory_ids] - self.frame_skip * (self.num_frames - 1)
        start_ids = np.array([np.random.randint(0, m + 1) for m in max_start])

        end_ids         = start_ids + (self.frame_skip * (self.num_frames - 1))

        observations    = np.zeros((batch_size, self.num_frames, 96, 96, 3),                dtype=np.uint8)
        proprios        = np.zeros((batch_size, self.num_frames, 7),                        dtype=np.float32)
        actions         = np.zeros((batch_size, self.num_frames-1, 3 * self.frame_skip),    dtype=np.float32)
        rewards         = np.zeros((batch_size, self.num_frames-1, self.frame_skip),        dtype=np.float32)
        dones           = np.zeros((batch_size, self.num_frames-1),                         dtype=np.float32)

        for i, (trajectory_idx, start, end) in enumerate(zip(trajectory_ids, start_ids, end_ids)):
            o, p, a, r, d = self.get_item(trajectory_idx, start, end)

            observations[i, ...] = o
            proprios[i, ...] = p
            actions[i, ...] = a
            rewards[i, ...] = r
            dones[i, ...]   = d

        return TensorDict({
            "obs": TensorDict({
                "rgb":      torch.from_numpy(observations),
                "proprios": torch.from_numpy(proprios),
            }, batch_size=[batch_size, self.num_frames]),
            "actions":  torch.from_numpy(actions),
            "rewards":  torch.from_numpy(rewards),
            "dones":    torch.from_numpy(dones),
        }, batch_size=[batch_size])

    def save(self, path):
        np.savez_compressed(
            path,
            num_trajectories=self.num_trajectories,
            trajectory_length=self.trajectory_length,
            frame_skip=self.frame_skip,
            num_frames=self.num_frames,
            observations=self.observations,
            proprios=self.proprios,
            actions=self.actions,
            rewards=self.rewards,
            traj_lengths=self.traj_lengths,
            ptr=self.ptr,
            size=self.size,
        )

    def load(self, path, strict=True):
        data = np.load(path, allow_pickle=False)

        # restore configuration
        self.num_trajectories = int(data["num_trajectories"])
        self.trajectory_length = int(data["trajectory_length"])
        self.frame_skip = int(data["frame_skip"])
        self.num_frames = int(data["num_frames"])

        if strict:
            # validate shapes against current allocations
            for k, arr in (
                ("observations", self.observations),
                ("proprios", self.proprios),
                ("actions", self.actions),
                ("rewards", self.rewards),
                ("traj_lengths", self.traj_lengths),
            ):
                if data[k].shape != arr.shape:
                    raise ValueError(
                        f"Shape mismatch for '{k}': checkpoint {data[k].shape} vs current {arr.shape}. "
                        f"Set strict=False to reallocate."
                    )

            # load into existing arrays
            self.observations[...] = data["observations"]
            self.proprios[...] = data["proprios"]
            self.actions[...] = data["actions"]
            self.rewards[...] = data["rewards"]
            self.traj_lengths[...] = data["traj_lengths"]
        else:
            # reallocate to checkpoint shapes
            self.observations = data["observations"]
            self.proprios = data["proprios"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.traj_lengths = data["traj_lengths"]

        # restore pointers
        self.ptr = int(data["ptr"])
        self.size = int(data["size"])
