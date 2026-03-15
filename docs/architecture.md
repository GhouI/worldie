# Architecture

## Goal

Build a model that learns an internal simulation of an environment from sequences of:

- images
- actions
- rewards
- episode termination flags

## First version

The first version uses a simplified recurrent state-space model:

1. an image encoder compresses each frame into features
2. a posterior network turns encoded features and recurrent state into a stochastic latent state
3. a prior network predicts the next latent state from the previous latent state, action, and recurrent state
4. decoders reconstruct the observation, reward, and termination flag

This is the core world-model loop used in systems like PlaNet and Dreamer, but stripped down to be teachable.

## Why this architecture

- It is expressive enough to be real machine learning, not toy linear algebra.
- It separates representation learning from dynamics learning.
- It gives you the right mental model for stronger future systems.
- It fits on a single machine and does not need a massive dataset.

## Directory map

- `src/worldie/collect.py`: gathers trajectories from `CartPole-v1`
- `src/worldie/data.py`: loads saved episodes and builds training sequences
- `src/worldie/models/`: encoder, latent dynamics, decoders, full model
- `src/worldie/train.py`: training loop and loss calculation
- `src/worldie/cli.py`: command line entry point

## Data flow

1. collect episodes with rendered RGB frames
2. resize frames to `64x64`
3. store each episode as a compressed `.npz`
4. sample short sequences during training
5. encode each frame and roll the latent state through time
6. reconstruct frame, reward, and done
7. train with reconstruction + reward + done + KL losses

## Current training workflow

The training pipeline now also supports:

- train/validation episode splits
- resumable checkpoints
- `last`, `best`, and periodic epoch checkpoints
- `history.json` run summaries
- device selection through the CLI

## Environment scope

Right now the project supports discrete-action environments because the latent dynamics model embeds actions as integer tokens. That is enough for `CartPole-v1` and `Acrobot-v1`, but not yet for continuous-control tasks.

## What this version does not do yet

- imagination rollouts for policy learning
- transformers
- large offline datasets
- latent overshooting
- discrete latents
- video prediction benchmarks

That is intentional. Those are phase-two topics after the baseline works.
