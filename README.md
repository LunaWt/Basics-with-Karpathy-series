# Basics-with-Karpathy-series


Language-model fundamentals written by hand, in dependency order, following Karpathy's
"Neural Networks: Zero to Hero". Nothing here imports a finished implementation of the
thing it is demonstrating: the autograd engine, the bigram models, the MLP, the
initialization fixes and BatchNorm are all typed out and then verified against PyTorch.

Karpathy's videos are the most useful thing I have found for learning this properly. It
matters to me to understand it from the bottom, and this is a good place to start.

## What's here

| Path | What |
|---|---|
| `engine.py` | scalar reverse-mode autodiff: `Value`, topological `backward()`, `tanh`, `Neuron`/`Layer`/`MLP` |
| `bigram/makemore.py` | count-based bigram baseline: smoothed counts, normalized rows, average NLL, sampling |
| `bigram/neural_bigram.py` | the same model as one-hot × weight matrix, NumPy forward pass |
| `bigram/neural_bigram_torch.py` | the PyTorch version, trained by gradient descent |
| `bigram/bengio_mlp.py` | context-window MLP: embedding table, minibatch loop, LR decay, `no_grad` eval, `multinomial` sampling |
| `bigram/lr_sweep.py` | learning-rate sweep + independent fixed-LR runs → `lr_sweep.png` |
| `bigram/l05_diagnostics.py` | 8-layer `tanh` stack under three init regimes → `l05_diagnostics.png` |
| `bigram/manual_backprop.py` | manual gradients through cross-entropy, `tanh`, matmul, embeddings, BatchNorm (in progress) |

Dataset: `bigram/names.txt`, 32k names. Everything runs on CPU in seconds to minutes.

## Results that are actually measured

- **Autograd.** Central finite differences agree with the engine's gradients to `4.2e-11`
  on a DAG with a shared path; an `MLP([2, 4, 4, 1])` drops loss `4.191788 → 0.006832`.
- **Bigram.** Neural bigram converges to `2.4608` mean NLL, against the smoothed count
  model's `2.4541` — the same optimum, reached by gradient descent instead of counting.
- **Bengio MLP.** `train 2.0432 / val 2.1117` at 200k steps
  (`block_size=3`, `embedding_dim=10`, `hidden_dim=200`, `batch_size=32`).
- **A split bug worth keeping.** `names.txt` is ordered by popularity, so a sequential
  80/20 split is a distribution shift, not a split: train/val gap `+0.2679`. After
  `random.Random(42).shuffle` the same run gives `+0.0091`.
- **Initialization.** With `W1 ~ U(-1,1)`, `43.7%` of `tanh` units sit past `±0.99`
  (`atanh(0.99) = 2.6467` converts the threshold into pre-activation space). Scaling by
  `1/√fan_in` predicts `std(pre) = 1.095` and measures `1.0856`, saturation `→ 1.62%`.
  BatchNorm on top: `→ 0.30%`.
- **The output layer wants the opposite.** Fan-in scaling on `W2` made the *initial* loss
  worse than a hand-tuned small init (`3.5703` vs `3.3718`, ideal `log(27) = 3.2958`) —
  hidden layers want variance preserved, the output layer wants logits near zero.
- **Backward, not just forward.** Bad init makes gradients *explode* toward layer 1
  (~95× over 8 layers), because the matmul's amplification `√fan · std(W) = 8.164` beats
  the throttling by a saturated `tanh'`. Both failures happen at once.

## Run

```sh
python bigram/makemore.py        # count baseline
python bigram/bengio_mlp.py      # MLP, ~200k steps
python bigram/lr_sweep.py        # writes bigram/lr_sweep.png
python bigram/l05_diagnostics.py # writes bigram/l05_diagnostics.png
```

Python 3.14, no venv: `numpy`, `torch`, `matplotlib`, `graphviz`.

## Status

Milestones L00–L05 are closed (autograd, count bigram, neural bigram, Bengio MLP,
initialization/BatchNorm diagnostics). L06 — manual backprop validated against autograd —
is in progress. After that: WaveNet-style hierarchy, then a decoder-only GPT and a
byte-level BPE tokenizer.

The progress ledger with per-milestone evidence lives outside the repo; every number in
this file comes from a run that is reproducible with the scripts above.
