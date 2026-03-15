# Research Roadmap

## Phase 1: Understand the baseline

Read and learn:

- autoencoders
- variational autoencoders
- recurrent neural networks and GRUs
- KL divergence
- model-based reinforcement learning

Project upgrades:

- better logging
- validation split
- reconstruction image saving
- GPU device reporting

## Phase 2: Make the world model stronger

Study:

- Dreamer
- PlaNet
- RSSM
- latent imagination

Project upgrades:

- train a policy from imagined rollouts
- add configurable latent sizes and model sizes
- support continuous-control environments
- add checkpoint resume support

## Phase 3: Move toward richer worlds

Study:

- video prediction
- transformers for sequence modeling
- tokenized observations
- offline RL datasets

Project upgrades:

- use richer environments with moving backgrounds
- add action repeat and longer horizons
- explore transformer dynamics or discrete latents
- train on larger stored datasets using the available disk space

