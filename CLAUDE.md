# new_transformer

Language-model fundamentals implemented by hand, in dependency order: scalar autograd →
count-based bigram → neural bigram → Bengio MLP → initialization and BatchNorm diagnostics →
manual backprop → GPT. Nothing here imports a finished implementation of the thing it is
demonstrating.

## Layout

| Path | What |
|---|---|
| `engine.py` | scalar reverse-mode autodiff: `Value`, topological `backward()`, `tanh`, `Neuron`/`Layer`/`MLP`, a finite-difference check |
| `bigram/makemore.py` | count-based bigram baseline: smoothed counts, normalized rows, average NLL, sampling |
| `bigram/neural_bigram.py` | the same model as one-hot × weight matrix, NumPy forward pass |
| `bigram/neural_bigram_torch.py` | the PyTorch version of it |
| `bigram/bengio_mlp.py` | context-window MLP with an embedding table, minibatch loop, LR decay, `no_grad` eval, `multinomial` sampling |
| `bigram/lr_sweep.py`, `bigram/l05_diagnostics.py` | reproducible experiment scripts; they write the `.png` next to them |
| `bigram/names.txt` | dataset (32k names) |
| `notes/` | **untracked.** Progress ledger, deferred curriculum, market research — local only |

## Environment

Windows, global Python 3.14, no project venv. `numpy`, `torch`, `matplotlib`, `graphviz`.
If a venv becomes necessary, `uv venv`.

## Working rules for an agent in this repo

- The point of the repo is that the owner writes the core mechanism by hand. Do not
  replace a hand-written file with a finished implementation, and do not "fix" a file
  into completeness — ask first.
- Commented-out blocks in `engine.py` and the `if step == 0` diagnostic block in
  `bengio_mlp.py` are deliberate scaffolding, not dead code. Leave them.
- Experiment scripts (`lr_sweep.py`, `l05_diagnostics.py`) are agent-written and safe to
  modify; they must stay seeded and reproducible.
- Every claim about a run belongs in `notes/ledger.md` with the number that produced it.
  A number without its derivation next to it gets cut, not kept.
