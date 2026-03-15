from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldie.data import SequenceDataset, load_episodes
from worldie.models.world_model import WorldModel


def test_dataset_shapes(tmp_path: Path) -> None:
    observations = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros((5,), dtype=np.int64)
    rewards = np.zeros((5,), dtype=np.float32)
    dones = np.zeros((5,), dtype=np.bool_)
    np.savez_compressed(
        tmp_path / "episode_00000.npz",
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    episodes = load_episodes(tmp_path)
    dataset = SequenceDataset(episodes, sequence_length=4)
    batch = dataset[0]

    assert batch["observations"].shape == (5, 3, 64, 64)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["dones"].shape == (4,)


def test_world_model_output_shapes() -> None:
    model = WorldModel(action_dim=2, latent_dim=32, hidden_dim=128)
    batch = {
        "observations": torch.rand(2, 5, 3, 64, 64),
        "actions": torch.zeros(2, 4, dtype=torch.long),
        "rewards": torch.zeros(2, 4),
        "dones": torch.zeros(2, 4),
    }

    outputs = model(batch)

    assert outputs["reconstruction"].shape == (2, 4, 3, 64, 64)
    assert outputs["reward"].shape == (2, 4)
    assert outputs["done_logits"].shape == (2, 4)
    assert outputs["kl"].shape == (2, 4)
