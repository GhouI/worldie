# Datasets

## Do we need a dataset?

Yes. A world model learns from sequences of experience, so it always needs data.

The question is not whether we need a dataset. The question is where the data should come from.

## Phase 1 dataset: collect it ourselves

For the first version, the project creates its own dataset by interacting with `CartPole-v1`.

Each saved episode contains:

- RGB frames
- actions
- rewards
- done flags

This is enough to train a first world model and learn the full workflow:

1. collect trajectories
2. save them to disk
3. load sequences for training
4. fit a latent dynamics model

## Why not start with a giant external dataset?

- It would make the project harder to understand.
- It adds storage and preprocessing complexity before the baseline works.
- It hides the important relationship between action, state, and future outcome.

For learning, self-collected environment data is the right first step.

## When we should move to external datasets

After the baseline is working, external datasets become useful if we want:

- richer visuals
- longer horizons
- more complex action sequences
- broader generalization

At that point we can move to one of these:

- offline RL benchmark datasets
- gameplay recordings with action labels
- robot logs
- larger multimodal video datasets if we redesign the model for them

## Practical rule

- use self-collected trajectories for the first version
- use benchmark or domain data once the baseline is stable and understandable
