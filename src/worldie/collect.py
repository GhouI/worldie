from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from worldie.config import CollectConfig
from worldie.utils import ensure_dir, set_seed


def _resize_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(frame)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def collect_random_episodes(config: CollectConfig) -> list[Path]:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    env = gym.make(config.env_id, render_mode="rgb_array")
    saved_paths: list[Path] = []

    for episode_idx in range(config.episodes):
        _, _ = env.reset(seed=config.seed + episode_idx)
        frames: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        frame = env.render()
        frames.append(_resize_frame(frame, config.image_size))

        for _ in range(config.max_steps):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)

            frames.append(_resize_frame(env.render(), config.image_size))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))

            if terminated or truncated:
                break

        path = config.output_dir / f"episode_{episode_idx:05d}.npz"
        np.savez_compressed(
            path,
            observations=np.stack(frames, axis=0),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
        )
        saved_paths.append(path)

    env.close()
    return saved_paths

