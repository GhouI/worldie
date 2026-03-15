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

