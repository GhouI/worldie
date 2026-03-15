from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldie.config import TrainConfig
from worldie.data import Episode, SequenceDataset, infer_action_dim, load_episodes, split_episodes
from worldie.models.world_model import WorldModel
from worldie.train import train_world_model


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
        env_id=np.asarray("CartPole-v1"),
        action_dim=np.asarray(2, dtype=np.int64),
        image_size=np.asarray(64, dtype=np.int64),
    )

    episodes = load_episodes(tmp_path)
    dataset = SequenceDataset(episodes, sequence_length=4)
    batch = dataset[0]

    assert batch["observations"].shape == (5, 3, 64, 64)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["dones"].shape == (4,)
    assert episodes[0].env_id == "CartPole-v1"
    assert episodes[0].action_dim == 2
    assert episodes[0].image_size == 64


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


def test_split_and_action_dim_inference() -> None:
    episodes = [
        build_episode(action_value=0, env_id="CartPole-v1", action_dim=2),
        build_episode(action_value=1, env_id="CartPole-v1", action_dim=2),
        build_episode(action_value=0, env_id="CartPole-v1", action_dim=2),
    ]

    train_episodes, validation_episodes = split_episodes(episodes, validation_ratio=0.34, seed=7)

    assert len(train_episodes) == 2
    assert len(validation_episodes) == 1
    assert infer_action_dim(episodes) == 2


def test_training_writes_run_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for episode_idx in range(4):
        observations = np.zeros((7, 64, 64, 3), dtype=np.uint8)
        actions = np.full((6,), episode_idx % 2, dtype=np.int64)
        rewards = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        dones = np.zeros((6,), dtype=np.bool_)
        dones[-1] = True
        np.savez_compressed(
            data_dir / f"episode_{episode_idx:05d}.npz",
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            env_id=np.asarray("CartPole-v1"),
            action_dim=np.asarray(2, dtype=np.int64),
            image_size=np.asarray(64, dtype=np.int64),
        )

    checkpoint = train_world_model(
        TrainConfig(
            data_dir=data_dir,
            artifact_dir=tmp_path / "artifacts",
            sequence_length=4,
            batch_size=2,
            epochs=1,
            log_every=100,
            validation_ratio=0.25,
            device="cpu",
        )
    )

    assert checkpoint.exists()
    assert (tmp_path / "artifacts" / "best.pt").exists()
    assert (tmp_path / "artifacts" / "last.pt").exists()
    assert (tmp_path / "artifacts" / "history.json").exists()


def build_episode(action_value: int, env_id: str, action_dim: int) -> Episode:
    return Episode(
        observations=np.zeros((6, 64, 64, 3), dtype=np.uint8),
        actions=np.full((5,), action_value, dtype=np.int64),
        rewards=np.zeros((5,), dtype=np.float32),
        dones=np.zeros((5,), dtype=np.bool_),
        env_id=env_id,
        action_dim=action_dim,
        image_size=64,
    )
