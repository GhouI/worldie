from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from worldie.config import TrainConfig
from worldie.data import SequenceDataset, load_episodes
from worldie.models.world_model import WorldModel
from worldie.utils import default_device, ensure_dir, set_seed


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_world_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    ensure_dir(config.artifact_dir)

    device = default_device()
    episodes = load_episodes(config.data_dir)
    dataset = SequenceDataset(episodes=episodes, sequence_length=config.sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model = WorldModel(
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics: list[dict[str, float]] = []
    step_count = 0

    for epoch in range(config.epochs):
        model.train()
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)

            target_observations = batch["observations"][:, 1:]
            reconstruction_loss = F.mse_loss(outputs["reconstruction"], target_observations)
            reward_loss = F.mse_loss(outputs["reward"], batch["rewards"])
            done_loss = F.binary_cross_entropy_with_logits(outputs["done_logits"], batch["dones"])
            kl_loss = outputs["kl"].mean()

            loss = reconstruction_loss + reward_loss + done_loss + (config.kl_scale * kl_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % config.log_every == 0:
                metric = {
                    "epoch": float(epoch),
                    "step": float(step_count),
                    "loss": float(loss.item()),
                    "reconstruction_loss": float(reconstruction_loss.item()),
                    "reward_loss": float(reward_loss.item()),
                    "done_loss": float(done_loss.item()),
                    "kl_loss": float(kl_loss.item()),
                }
                metrics.append(metric)
                print(json.dumps(metric))

    checkpoint_path = config.artifact_dir / "world_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "device": str(device),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path
