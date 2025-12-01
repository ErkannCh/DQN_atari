import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))

def make_atari_env(
    env_id: str,
    frame_skip: int = 4,
    num_stack: int = 4,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=num_stack)
    env = ClipRewardEnv(env)
    return env
