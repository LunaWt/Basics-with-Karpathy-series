"""Learning-rate search for the Bengio MLP (L04, step 7).

Two panels:
  left  - Karpathy-style rising sweep: lr grows exponentially during one run.
  right - independent fixed-lr runs from the same init (slower, no confound).
"""

import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

BLOCK_SIZE = 3
EMBEDDING_DIM = 10
HIDDEN_DIM = 200
BATCH_SIZE = 32
SEED = 42

words = open('bigram/names.txt', encoding='utf-8').read().splitlines()
vocab = ['.'] + sorted(set(''.join(words)))
vocab_size = len(vocab)
stoi = {letter: idx for idx, letter in enumerate(vocab)}

random.Random(SEED).shuffle(words)
cut = int(len(words) * 0.8)


def build(word_list):
    xs, ys = [], []
    for word in word_list:
        word = '.' * BLOCK_SIZE + word + '.'
        for i in range(BLOCK_SIZE, len(word)):
            xs.append([stoi[c] for c in word[i - BLOCK_SIZE:i]])
            ys.append(stoi[word[i]])
    return torch.tensor(xs), torch.tensor(ys)


X_train, Y_train = build(words[:cut])


def init_params():
    torch.manual_seed(SEED)
    C = torch.randn(vocab_size, EMBEDDING_DIM, requires_grad=True)
    W1 = torch.empty(BLOCK_SIZE * EMBEDDING_DIM, HIDDEN_DIM).uniform_(-1, 1).requires_grad_()
    b1 = torch.zeros(HIDDEN_DIM, requires_grad=True)
    W2 = (torch.empty(HIDDEN_DIM, vocab_size).uniform_(-1, 1) * 0.05).requires_grad_()
    b2 = torch.zeros(vocab_size, requires_grad=True)
    return [C, W1, b1, W2, b2]


def forward(params, X):
    C, W1, b1, W2, b2 = params
    emb = C[X].view(X.shape[0], -1)
    return torch.tanh(emb @ W1 + b1) @ W2 + b2


def train_step(params, lr):
    ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    loss = F.cross_entropy(forward(params, X_train[ix]), Y_train[ix])
    for p in params:
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
    return loss.item()


def rising_sweep(steps=20000, lo=-3.0, hi=0.0):
    params = init_params()
    exponents = torch.linspace(lo, hi, steps)
    lrs, losses = [], []
    for s in range(steps):
        lr = float(10 ** exponents[s])
        loss = train_step(params, lr)
        lrs.append(lr)
        losses.append(loss if loss == loss and loss < 20 else 20.0)
    return lrs, losses


def fixed_runs(lr_grid, steps=15000):
    out = []
    for lr in lr_grid:
        params = init_params()
        for _ in range(steps):
            train_step(params, lr)
        with torch.no_grad():
            full = F.cross_entropy(forward(params, X_train), Y_train).item()
        full = full if full == full and full < 20 else 20.0
        out.append(full)
        print(f'  lr={lr:<8.4g} -> train loss {full:.4f}')
    return out


def smooth(values, window=200):
    kernel = torch.ones(window) / window
    padded = torch.tensor(values).view(1, 1, -1)
    return torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).view(-1).tolist()


if __name__ == '__main__':
    print('rising sweep...')
    lrs, losses = rising_sweep()
    sm = smooth(losses)
    lrs_sm = lrs[len(lrs) - len(sm):]

    lr_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    print('fixed runs...')
    finals = fixed_runs(lr_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lrs_sm, sm)
    ax1.set_xscale('log')
    ax1.set_xlabel('learning rate (log)')
    ax1.set_ylabel('minibatch loss (smoothed, window=200)')
    ax1.set_title('rising sweep: lr grows during one run')
    ax1.grid(alpha=0.3)

    ax2.plot(lr_grid, finals, marker='o')
    ax2.set_xscale('log')
    ax2.set_xlabel('learning rate (log)')
    ax2.set_ylabel('full train loss after 15000 steps')
    ax2.set_title('independent runs, same init')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('bigram/lr_sweep.png', dpi=120)
    print('saved bigram/lr_sweep.png')
