"""L06 — ручной backprop.

Тот же forward, что в bengio_mlp.py, но разобранный на мелкие шаги: каждый
промежуточный тензор лежит в своей переменной, чтобы его можно было
продифференцировать отдельно.

План: идём от loss назад, по одному тензору за раз, и каждый свой градиент
сверяем с тем, что посчитал PyTorch.

Запуск:  python bigram/manual_backprop.py
"""

import random

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Данные — один в один как в bengio_mlp.py, чтобы числа были сопоставимы
# --------------------------------------------------------------------------

with open('bigram/names.txt', encoding='utf-8') as file:
    words = file.read().splitlines()

random.Random(42).shuffle(words)

block_size = 3
words = ['.' * block_size + word + '.' for word in words]

vocab = sorted({letter for word in words for letter in word})
vocab.remove('.')
vocab.insert(0, '.')
vocab_size = len(vocab)

stoi = {letter: idx for idx, letter in enumerate(vocab)}
itos = {idx: letter for idx, letter in enumerate(vocab)}

xs, ys = [], []
for word in words[: int(len(words) * 0.8)]:
    for i in range(block_size, len(word)):
        xs.append([stoi[c] for c in word[i - block_size : i]])
        ys.append(stoi[word[i]])

X_train = torch.tensor(xs, dtype=torch.long)
Y_train = torch.tensor(ys, dtype=torch.long)

# --------------------------------------------------------------------------
# Параметры
# --------------------------------------------------------------------------

embedding_dim = 10
hidden_dim = 64  # меньше, чем 200 в bengio_mlp.py — тензоры считаются быстрее
batch_size = 32
fan_in = block_size * embedding_dim

g = torch.Generator().manual_seed(42)

C = torch.randn((vocab_size, embedding_dim), generator=g)
W1 = torch.randn((fan_in, hidden_dim), generator=g) / fan_in**0.5
W2 = torch.randn((hidden_dim, vocab_size), generator=g) / hidden_dim**0.5

# Заводим мелким шумом, а не точными нулями/единицами специально: на ровных
# 0 и 1 часть ошибок в градиентах взаимно сокращается и баг не видно.
b2 = torch.randn(vocab_size, generator=g) * 0.1
bn_gain = torch.randn((1, hidden_dim), generator=g) * 0.1 + 1.0
bn_bias = torch.randn((1, hidden_dim), generator=g) * 0.1

# b1 нет: BatchNorm вычитает среднее и сдвиг всё равно исчезает

params = [C, W1, W2, b2, bn_gain, bn_bias]
print(f'Параметров: {sum(p.nelement() for p in params)}')
for p in params:
    p.requires_grad_(True)

# --------------------------------------------------------------------------
# Один батч
# --------------------------------------------------------------------------

n = batch_size
ix = torch.randint(0, X_train.shape[0], (n,), generator=g)
Xb, Yb = X_train[ix], Y_train[ix]

# --------------------------------------------------------------------------
# Forward, разобранный на шаги
# --------------------------------------------------------------------------

# эмбеддинги
emb = C[Xb]                                              # (n, block_size, embedding_dim)
embcat = emb.view(emb.shape[0], -1)                      # (n, fan_in)

# линейный слой 1
hprebn = embcat @ W1                                     # (n, hidden_dim)

# BatchNorm, тоже по шагам
bn_mean = 1 / n * hprebn.sum(0, keepdim=True)            # (1, hidden_dim)
bn_diff = hprebn - bn_mean                               # (n, hidden_dim)
bn_diff2 = bn_diff**2                                    # (n, hidden_dim)
bn_var = 1 / (n - 1) * bn_diff2.sum(0, keepdim=True)     # (1, hidden_dim)  поправка Бесселя
bn_var_inv = (bn_var + 1e-5) ** -0.5                     # (1, hidden_dim)
bn_raw = bn_diff * bn_var_inv                            # (n, hidden_dim)
hpreact = bn_gain * bn_raw + bn_bias                     # (n, hidden_dim)

# нелинейность
hidden = torch.tanh(hpreact)                             # (n, hidden_dim)

# линейный слой 2
logits = hidden @ W2 + b2                                # (n, vocab_size)

# cross-entropy, тоже по шагам (то же самое, что F.cross_entropy)
logit_maxes = logits.max(1, keepdim=True).values         # (n, 1)
norm_logits = logits - logit_maxes                       # (n, vocab_size)  защита от overflow
counts = norm_logits.exp()                               # (n, vocab_size)
counts_sum = counts.sum(1, keepdim=True)                 # (n, 1)
counts_sum_inv = counts_sum**-1                          # (n, 1)
probs = counts * counts_sum_inv                          # (n, vocab_size)
logprobs = probs.log()                                   # (n, vocab_size)
loss = -logprobs[range(n), Yb].mean()                    # скаляр

# --------------------------------------------------------------------------
# Эталонные градиенты от PyTorch
# --------------------------------------------------------------------------

intermediates = [
    logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits,
    logit_maxes, logits, hidden, hpreact, bn_raw, bn_var_inv, bn_var,
    bn_diff2, bn_diff, bn_mean, hprebn, embcat, emb,
]
for t in intermediates:
    t.retain_grad()

for p in params:
    p.grad = None
loss.backward()

# проверка, что разбор на шаги ничего не сломал
assert torch.allclose(loss, F.cross_entropy(logits, Yb)), 'разбор cross-entropy разошёлся'
print(f'loss = {loss.item():.6f}   (F.cross_entropy сходится)')

# --------------------------------------------------------------------------
# Сверка
# --------------------------------------------------------------------------


def cmp(name: str, dt: torch.Tensor, t: torch.Tensor) -> None:
    """Сравнить ручной градиент dt с эталонным t.grad."""
    exact = torch.all(dt == t.grad).item()
    close = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{name:16s} | точно: {str(exact):5s} | близко: {str(close):5s} | макс. разница: {maxdiff}')


# --------------------------------------------------------------------------
# ЗДЕСЬ ТВОИ ГРАДИЕНТЫ
# --------------------------------------------------------------------------

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1/n
dprobs = dlogprobs * (1 / probs)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts_sum = dcounts_sum_inv * (-1 * 1 / counts_sum**2)
dcounts = dprobs * counts_sum_inv
dcounts += dcounts_sum * torch.ones_like(counts)
dnorm_logits = dcounts * counts
dlogits = dnorm_logits * 1
dlogit_maxes = (dnorm_logits * -1).sum(1, keepdim=True)
dlogits += dlogit_maxes * F.one_hot(logits.max(1).indices, num_classes=27)
dhidden = dlogits @ W2.T
dW2 = hidden.T @ dlogits
db2 = 
# dbn_gain = (dhpreact * bn_raw).sum(0, keepdim=True)
# dbn_raw = dhpreact * bn_gain
cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logits', dlogits, logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('hidden', dhidden, hidden)
cmp('W2', dW2, W2)
# cmp('bn_gain', dbn_gain, bn_gain)
# cmp('bn_raw', dbn_raw, bn_raw)

# emb = C[Xb]                                              # (n, block_size, embedding_dim)
# embcat = emb.view(emb.shape[0], -1)                      # (n, fan_in)

# # линейный слой 1
# hprebn = embcat @ W1                                     # (n, hidden_dim)

# # BatchNorm, тоже по шагам
# bn_mean = 1 / n * hprebn.sum(0, keepdim=True)            # (1, hidden_dim)
# bn_diff = hprebn - bn_mean                               # (n, hidden_dim)
# bn_diff2 = bn_diff**2                                    # (n, hidden_dim)
# bn_var = 1 / (n - 1) * bn_diff2.sum(0, keepdim=True)     # (1, hidden_dim)  поправка Бесселя
# bn_var_inv = (bn_var + 1e-5) ** -0.5                     # (1, hidden_dim)
# bn_raw = bn_diff * bn_var_inv                            # (n, hidden_dim)
# hpreact = bn_gain * bn_raw + bn_bias                     # (n, hidden_dim)

# # нелинейность
# hidden = torch.tanh(hpreact)                             # (n, hidden_dim)

# # линейный слой 2
# logits = hidden @ W2 + b2                                # (n, vocab_size)

# # cross-entropy, тоже по шагам (то же самое, что F.cross_entropy)
# logit_maxes = logits.max(1, keepdim=True).values         # (n, 1)
# norm_logits = logits - logit_maxes                       # (n, vocab_size)  защита от overflow
# counts = norm_logits.exp()                               # (n, vocab_size)
# counts_sum = counts.sum(1, keepdim=True)                 # (n, 1)
# counts_sum_inv = counts_sum**-1                          # (n, 1)
