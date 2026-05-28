import numpy as np
import gymnasium as gym
from gymnasium import spaces


def extract_sensor_values(observation):
    """Extract [speed, abs_0..abs_3, steering, gyroscope] from a (96, 96, 3) frame."""
    speed = observation[84:94, 12, 0].sum() / 255 / 5

    abs_sensors = observation[84:94, 18:25:2, 2].sum(axis=0) / 255 / 5  # (4,)

    steer_crop = observation[88, 38:58, 1] / 255 / 10
    steer_crop[:10] *= -1
    steering = steer_crop.sum()

    gyro_crop = observation[88, 58:86, 0] / 255 / 5
    gyro_crop[:14] *= -1
    gyroscope = gyro_crop.sum()

    return np.array([speed, *abs_sensors, steering, gyroscope])


def block_proprioceptives(observation):
    observation[84:, :, :] = 0


class SDCObservationWrapper(gym.ObservationWrapper):
    """Splits the CarRacing RGB observation into a blocked image + proprio vector."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "rgb": env.observation_space,
            "proprios": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        })

    def observation(self, observation):
        proprios = extract_sensor_values(observation)

        rgb = observation.copy()
        block_proprioceptives(rgb)

        return {"rgb": rgb, "proprios": proprios}


def construct_environment(render_mode=None):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = SDCObservationWrapper(env)
    return env


if __name__ == "__main__":
    env = construct_environment()
    obs, info = env.reset()
    print(obs)


