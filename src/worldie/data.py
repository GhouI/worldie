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
            )
        )
    return episodes


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

