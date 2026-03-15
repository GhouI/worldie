from __future__ import annotations

import argparse
from pathlib import Path

from worldie.collect import collect_random_episodes
from worldie.config import CollectConfig, TrainConfig
from worldie.train import train_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Worldie CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect random trajectories")
    collect_parser.add_argument("--env-id", type=str, default="CartPole-v1")
    collect_parser.add_argument("--episodes", type=int, default=50)
    collect_parser.add_argument("--max-steps", type=int, default=500)
    collect_parser.add_argument("--image-size", type=int, default=64)
    collect_parser.add_argument("--output-dir", type=Path, default=Path("data/cartpole"))
    collect_parser.add_argument("--seed", type=int, default=7)

    train_parser = subparsers.add_parser("train", help="Train the world model")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/cartpole"))
    train_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/world_model"))
    train_parser.add_argument("--sequence-length", type=int, default=16)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--kl-scale", type=float, default=0.1)
    train_parser.add_argument("--latent-dim", type=int, default=32)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--action-dim", type=int, default=None)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--validation-ratio", type=float, default=0.1)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--save-every", type=int, default=1)
    train_parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--resume-from", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=7)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        config = CollectConfig(
            env_id=args.env_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            image_size=args.image_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        paths = collect_random_episodes(config)
        print(f"Saved {len(paths)} episodes to {config.output_dir}")
        return

    if args.command == "train":
        config = TrainConfig(
            data_dir=args.data_dir,
            artifact_dir=args.artifact_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            kl_scale=args.kl_scale,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            log_every=args.log_every,
            validation_ratio=args.validation_ratio,
            num_workers=args.num_workers,
            save_every=args.save_every,
            device=args.device,
            resume_from=args.resume_from,
            seed=args.seed,
        )
        checkpoint = train_world_model(config)
        print(f"Saved checkpoint to {checkpoint}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
