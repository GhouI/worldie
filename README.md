# Worldie

`Worldie` is a starter project for learning how to build a world model from scratch.

The first version is intentionally small and explicit:

- collect trajectories from a simple environment as image sequences
- train a latent dynamics model that predicts what happens next
- document the code and the reasoning in `docs/`

## Why this project starts small

World models can become complex quickly. Starting with a rendered `CartPole-v1` environment keeps the core ideas visible:

- observations are images
- actions influence future states
- the model compresses observations into latent state
- a recurrent dynamics model predicts future latent state
- decoders reconstruct observations, rewards, and episode termination

This is a learning foundation, not a benchmark-chasing implementation.

## Project layout

- `src/worldie/`: source code
- `docs/`: explanations, architecture notes, and implementation log
- `tests/`: small verification tests

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m worldie.cli collect --episodes 50
python -m worldie.cli train --epochs 5
```

## Main commands

Collect random-policy data:

```bash
python -m worldie.cli collect --episodes 50 --output-dir data/cartpole
```

Train the first world model:

```bash
python -m worldie.cli train --data-dir data/cartpole --epochs 5
```

Run tests:

```bash
pytest
```

Read [docs/README.md](/run/media/abdul/Work/worldie/docs/README.md) first if you want the guided explanation.

