import torch 
import torch.nn.functional as F
import random

from torch.types import Number

torch.manual_seed(42)

file = open('bigram/names.txt', encoding='utf-8')
words = file.read().splitlines()
file.close()
words_len = len(words)

random.Random(42).shuffle(words)

for i in range(words_len):
    words[i] = '...' + words[i] + '.'
    
vocab = set()
for idx in range(words_len):
    for item in words[idx]:
        if item not in vocab:
            vocab.add(item)

vocab = sorted(list(vocab))
vocab.remove('.') 
vocab.insert(0, '.')

vocab_size = len(vocab)

stoi = {letter: idx for idx, letter in enumerate(vocab)}
itos = {idx: letter for idx, letter in enumerate(vocab)}

train_words = words[:int(words_len * 0.8)]
val_words = words[int(words_len * 0.8):]

# flat_train_words = ''.join(train_words)
# flat_val_words = ''.join(val_words)

xs_train = []
ys_train = []

xs_val = []
ys_val = []

for word in train_words:
    for i in range(3, len(word)):
        xs_train.append([stoi[word[i - 3]], stoi[word[i - 2]], stoi[word[i - 1]]])
        ys_train.append(stoi[word[i]])

for word in val_words:
    for i in range(3, len(word)):
        xs_val.append([stoi[word[i - 3]], stoi[word[i - 2]], stoi[word[i - 1]]])
        ys_val.append(stoi[word[i]])


X_train = torch.tensor(xs_train, dtype=torch.long)
Y_train = torch.tensor(ys_train, dtype=torch.long)

X_val = torch.tensor(xs_val, dtype=torch.long)
Y_val = torch.tensor(ys_val, dtype=torch.long)


lr = 0.1
embedding_dim = 10
hidden_dim = 200
block_size = 3
batch_size = 32
steps = 200001
C = torch.randn((vocab_size, embedding_dim), requires_grad=True)

# emb_train = C[X_train]
# emb = emb_train.view(-1, block_size * embedding_dim)

W1 = torch.randn(size=(block_size * embedding_dim, hidden_dim), requires_grad=True)
W2 = torch.randn(size=(hidden_dim, vocab_size), requires_grad=True)
b2 = torch.zeros(vocab_size, requires_grad=True)
bn_gain = torch.ones(size=(1, hidden_dim), requires_grad=True)
bn_bias = torch.zeros(size=(1, hidden_dim), requires_grad=True)
bn_mean_running = torch.zeros(size=(1, hidden_dim))
bn_std_running = torch.ones(size=(1, hidden_dim))
with torch.no_grad():
    W1 /= torch.sqrt(torch.tensor(block_size * embedding_dim)) 
    # W1 *= 5/3
    W2 /= torch.sqrt(torch.tensor(hidden_dim))


@torch.no_grad()
def get_loss_full(X, Y):
    emb_train = C[X]
    emb_train = emb_train.view(-1, block_size * embedding_dim)

    hpreact = emb_train @ W1
    hpreact  = bn_gain * (hpreact - bn_mean_running) / (bn_std_running + 1e-5) + bn_bias
    hidden = torch.tanh(hpreact)
    out = hidden @ W2 + b2

    return (F.cross_entropy(out, Y)).item()

params = [C, W1, W2, b2, bn_gain, bn_bias]

for step in range(steps):
    ix = torch.randint(0, X_train.shape[0], (batch_size,))
    X_train_batch = X_train[ix]

    emb_train = C[X_train_batch]
    emb_train = emb_train.view(-1, block_size * embedding_dim)
    if step == 0:
        with torch.no_grad():
            hpreact = emb_train @ W1
            mean = hpreact.mean(dim=0, keepdim=True)
            std = hpreact.std(dim=0, keepdim=True)
            hid  = bn_gain * (hpreact - mean) / (std + 1e-5) + bn_bias
            print(f'Mean: {hid.mean()} | Std: {hid.std()}')
            print(torch.atanh(torch.tensor(0.99)))
            print((hid.tanh().abs() > 0.99).float().mean())
            print(f'Mean: {hid.tanh().mean()} | Std: {hid.tanh().std()}')
    hpreact = emb_train @ W1
    mean = hpreact.mean(dim=0, keepdim=True)
    std = hpreact.std(dim=0, keepdim=True)
    with torch.no_grad():
        bn_mean_running = 0.999 * bn_mean_running + 0.001 * mean
        bn_std_running  = 0.999 * bn_std_running  + 0.001 * std
    hpreact  = bn_gain * (hpreact - mean) / (std + 1e-5) + bn_bias
    hidden = torch.tanh(hpreact)
    # hidden = torch.tanh(emb_train @ W1)
    out = hidden @ W2 + b2

    loss = F.cross_entropy(out, Y_train[ix])
    loss.backward()

    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad = None

    if step % 20000 == 0:
        print(f"""
        Step: {step}
        Train loss: {get_loss_full(X_train, Y_train)}
        Val loss: {get_loss_full(X_val, Y_val)}""")

    if step >= steps * 0.7:
        lr = 0.01
        
@torch.no_grad()
def generate(len_generation: int):
    context = [0, 0, 0]
    name = ''

    for gen in range(len_generation):

        emb_train = C[context]
        emb_train = emb_train.view(-1, block_size * embedding_dim)

        hpreact = emb_train @ W1
        hpreact = bn_gain * (hpreact - bn_mean_running) / (bn_std_running + 1e-5) + bn_bias
        hidden = torch.tanh(hpreact)
        out = hidden @ W2 + b2

        out = F.softmax(out, dim=1)

        ix = torch.multinomial(out, 1).item()

        context = context[1:] + [ix]
        name += itos[ix]

        if ix == 0:
            return name

    return name


for _ in range(20):
    print(f'Name: {generate(20)}')


# print(X_train.shape, X_train.dtype)
# print(emb.shape)
# print(h.shape)

