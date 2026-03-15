from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CollectConfig:
    env_id: str = "CartPole-v1"
    episodes: int = 50
    max_steps: int = 500
    image_size: int = 64
    output_dir: Path = Path("data/cartpole")
    seed: int = 7


@dataclass(slots=True)
class TrainConfig:
    data_dir: Path = Path("data/cartpole")
    artifact_dir: Path = Path("artifacts/world_model")
    image_size: int = 64
    sequence_length: int = 16
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 3e-4
    kl_scale: float = 0.1
    latent_dim: int = 32
    hidden_dim: int = 128
    action_dim: int | None = None
    log_every: int = 10
    validation_ratio: float = 0.1
    num_workers: int = 0
    save_every: int = 1
    device: str = "auto"
    resume_from: Path | None = None
    seed: int = 7
