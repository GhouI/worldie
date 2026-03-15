from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    env_id: str | None = None
    action_dim: int | None = None
    image_size: int | None = None

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])


def load_episodes(data_dir: Path) -> list[Episode]:
    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode files found in {data_dir}")

    episodes: list[Episode] = []
    for path in paths:
        data = np.load(path)
        episodes.append(
            Episode(
                observations=data["observations"],
                actions=data["actions"],
                rewards=data["rewards"],
                dones=data["dones"],
                env_id=str(data["env_id"]) if "env_id" in data.files else None,
                action_dim=int(data["action_dim"]) if "action_dim" in data.files else None,
                image_size=int(data["image_size"]) if "image_size" in data.files else None,
            )
        )
    return episodes


def infer_action_dim(episodes: list[Episode]) -> int:
    metadata_values = {episode.action_dim for episode in episodes if episode.action_dim is not None}
    if len(metadata_values) > 1:
        raise ValueError("Episode files disagree on action_dim. Keep each training run on one environment.")
    if metadata_values:
        return metadata_values.pop()

    max_action = max(int(np.max(episode.actions)) for episode in episodes if episode.actions.size > 0)
    return max_action + 1


def split_episodes(
    episodes: list[Episode],
    validation_ratio: float,
    seed: int,
) -> tuple[list[Episode], list[Episode]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in the range [0.0, 1.0).")
    if validation_ratio == 0.0 or len(episodes) < 2:
        return episodes, []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    validation_count = max(1, int(round(len(episodes) * validation_ratio)))
    validation_count = min(validation_count, len(episodes) - 1)
    validation_indices = set(indices[:validation_count].tolist())

    train_episodes = [episode for idx, episode in enumerate(episodes) if idx not in validation_indices]
    validation_episodes = [episode for idx, episode in enumerate(episodes) if idx in validation_indices]
    return train_episodes, validation_episodes


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, episodes: list[Episode], sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.items: list[tuple[int, int]] = []
        self.episodes = episodes

        for episode_idx, episode in enumerate(episodes):
            if episode.transitions < sequence_length:
                continue
            max_start = episode.transitions - sequence_length
            for start in range(max_start + 1):
                self.items.append((episode_idx, start))

        if not self.items:
            raise ValueError("No training sequences available. Collect more data or reduce sequence_length.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self.items[index]
        episode = self.episodes[episode_idx]
        end = start + self.sequence_length

        observations = episode.observations[start : end + 1]
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        dones = episode.dones[start:end]

        observation_tensor = torch.from_numpy(observations).float().permute(0, 3, 1, 2) / 255.0

        return {
            "observations": observation_tensor,
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones.astype(np.float32)).float(),
        }
