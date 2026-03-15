# Implementation Log

## 2026-03-15: Bootstrap

### What was added

- a Python package with a small CLI
- data collection for rendered `CartPole-v1`
- a recurrent latent world model in PyTorch
- a training loop with explicit losses
- basic tests for shapes and data loading

### Why we started here

You said you do not yet know how to build AI systems, so the project needs to teach while it grows.

That means:

- small enough to understand end to end
- real enough to contain the important ideas
- documented enough that you can read the "why" beside the code

### Important decisions

- `CartPole-v1` instead of a harder environment:
  it keeps iteration fast and failure cases understandable
- recurrent state-space model instead of plain next-frame prediction:
  this is much closer to how modern world models are structured
- CLI scripts instead of notebooks:
  it makes the workflow reproducible and easier to document

### What to study from this version

- why latent states are useful
- what KL regularization is doing
- why we reconstruct reward and done, not only pixels
- how sequence models differ from i.i.d. supervised learning

## 2026-03-15: First baseline run

### What was done

- collected `100` episodes from `CartPole-v1`
- trained for `5` epochs on the collected dataset
- saved a baseline checkpoint to `artifacts/baseline_2026_03_15/world_model.pt`

### Why this run was important

The goal was not to chase performance yet. The goal was to confirm that the project works as a real training pipeline outside of a tiny smoke test.

### What changed in our understanding

- we now have a reproducible baseline artifact
- future model changes can be compared against a known starting point
- the next bottleneck is no longer "can it run?" but "how should we improve it?"

## 2026-03-15: Training pipeline upgrade

### What was added

- validation split support
- resumable checkpoints
- `best.pt`, `last.pt`, and periodic epoch snapshots
- `history.json` per-run metadata
- CLI device selection
- environment metadata in collected episodes

### Why this mattered

The original trainer could produce a model, but it was weak as a long-running workflow. Larger experiments need safer checkpoints, explicit validation, and enough metadata to compare runs without guessing.

### Hardware reality

This machine exposes AMD graphics hardware, not NVIDIA CUDA hardware. That means the correct future GPU path is ROCm, while the currently verified training path remains CPU PyTorch.

### Richer environment check

The collection and training path was also validated on `Acrobot-v1`, which proves the code is no longer narrowly tied to `CartPole-v1`.
