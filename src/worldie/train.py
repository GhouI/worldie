from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from worldie.config import TrainConfig
from worldie.data import SequenceDataset, infer_action_dim, load_episodes, split_episodes
from worldie.models.world_model import WorldModel
from worldie.utils import ensure_dir, resolve_device, set_seed


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _serialize_config(config: TrainConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _validate_image_size(episodes: list, expected_size: int) -> None:
    shapes = {(episode.observations.shape[1], episode.observations.shape[2]) for episode in episodes}
    if shapes != {(expected_size, expected_size)}:
        raise ValueError(
            f"Expected all observations to be {expected_size}x{expected_size}, but found {sorted(shapes)}."
        )


def _build_loader(
    episodes: list,
    config: TrainConfig,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader | None:
    if not episodes:
        return None

    dataset = SequenceDataset(episodes=episodes, sequence_length=config.sequence_length)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def _compute_losses(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    kl_scale: float,
) -> dict[str, torch.Tensor]:
    target_observations = batch["observations"][:, 1:]
    reconstruction_loss = F.mse_loss(outputs["reconstruction"], target_observations)
    reward_loss = F.mse_loss(outputs["reward"], batch["rewards"])
    done_loss = F.binary_cross_entropy_with_logits(outputs["done_logits"], batch["dones"])
    kl_loss = outputs["kl"].mean()
    total_loss = reconstruction_loss + reward_loss + done_loss + (kl_scale * kl_loss)

    return {
        "loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "reward_loss": reward_loss,
        "done_loss": done_loss,
        "kl_loss": kl_loss,
    }


def _average_metrics(accumulator: dict[str, float], batches: int) -> dict[str, float]:
    if batches == 0:
        return {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "reward_loss": 0.0,
            "done_loss": 0.0,
            "kl_loss": 0.0,
        }
    return {key: value / batches for key, value in accumulator.items()}


def _run_epoch(
    model: WorldModel,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    step_count: int,
) -> tuple[dict[str, float], int]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    metric_sums = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "reward_loss": 0.0,
        "done_loss": 0.0,
        "kl_loss": 0.0,
    }
    batch_count = 0

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            losses = _compute_losses(batch, outputs, config.kl_scale)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                step_count += 1
                if step_count % config.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "train_step",
                                "epoch": epoch,
                                "step": step_count,
                                **{key: float(value.item()) for key, value in losses.items()},
                            }
                        )
                    )

            for key, value in losses.items():
                metric_sums[key] += float(value.item())
            batch_count += 1

    return _average_metrics(metric_sums, batch_count), step_count


def _checkpoint_payload(
    model: WorldModel,
    optimizer: optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    step_count: int,
    action_dim: int,
    best_validation_loss: float,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _serialize_config(config),
        "device": str(device),
        "epoch": epoch,
        "step_count": step_count,
        "action_dim": action_dim,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def _write_history(
    path: Path,
    *,
    config: TrainConfig,
    device: torch.device,
    action_dim: int,
    train_episode_count: int,
    validation_episode_count: int,
    history: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "config": _serialize_config(config),
                "device": str(device),
                "action_dim": action_dim,
                "train_episode_count": train_episode_count,
                "validation_episode_count": validation_episode_count,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def train_world_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    ensure_dir(config.artifact_dir)

    device = resolve_device(config.device)
    episodes = load_episodes(config.data_dir)
    _validate_image_size(episodes, config.image_size)
    train_episodes, validation_episodes = split_episodes(
        episodes=episodes,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )

    action_dim = config.action_dim or infer_action_dim(episodes)
    train_loader = _build_loader(train_episodes, config, shuffle=True, device=device)
    if train_loader is None:
        raise ValueError("Training split is empty. Collect more data or reduce validation_ratio.")
    validation_loader = _build_loader(validation_episodes, config, shuffle=False, device=device)

    model = WorldModel(
        action_dim=action_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, object]] = []
    step_count = 0
    start_epoch = 0
    best_validation_loss = float("inf")

    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        step_count = int(checkpoint.get("step_count", 0))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))

    for epoch in range(start_epoch, start_epoch + config.epochs):
        train_metrics, step_count = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
            step_count=step_count,
        )

        validation_metrics = None
        if validation_loader is not None:
            validation_metrics, step_count = _run_epoch(
                model=model,
                loader=validation_loader,
                optimizer=None,
                device=device,
                config=config,
                epoch=epoch,
                step_count=step_count,
            )

        epoch_record: dict[str, object] = {
            "event": "epoch_summary",
            "epoch": epoch,
            "step_count": step_count,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        comparison_loss = (
            validation_metrics["loss"] if validation_metrics is not None else train_metrics["loss"]
        )
        if comparison_loss <= best_validation_loss:
            best_validation_loss = comparison_loss

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            device=device,
            epoch=epoch,
            step_count=step_count,
            action_dim=action_dim,
            best_validation_loss=best_validation_loss,
            history=history,
        )

        last_path = config.artifact_dir / "last.pt"
        torch.save(payload, last_path)

        if comparison_loss <= best_validation_loss:
            torch.save(payload, config.artifact_dir / "best.pt")

        if config.save_every > 0 and (epoch + 1) % config.save_every == 0:
            torch.save(payload, config.artifact_dir / f"epoch_{epoch + 1:04d}.pt")

    _write_history(
        config.artifact_dir / "history.json",
        config=config,
        device=device,
        action_dim=action_dim,
        train_episode_count=len(train_episodes),
        validation_episode_count=len(validation_episodes),
        history=history,
    )

    best_path = config.artifact_dir / "best.pt"
    final_checkpoint = config.artifact_dir / "world_model.pt"
    if best_path.exists():
        shutil.copyfile(best_path, final_checkpoint)
    else:
        shutil.copyfile(config.artifact_dir / "last.pt", final_checkpoint)
    return final_checkpoint
