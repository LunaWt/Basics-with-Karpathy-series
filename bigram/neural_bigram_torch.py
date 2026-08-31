import torch 
import torch.nn.functional as F

torch.manual_seed(42)

file = open('bigram/names.txt', encoding='utf-8')
words = file.read().splitlines()
file.close()
words_len = len(words)

for i in range(words_len):
    words[i] = '.' + words[i] + '.'
    
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



xs = torch.tensor([stoi[x] for word in words for x in word[:-1]]).long()
ys = torch.tensor([stoi[y] for word in words for y in word[1:]]).long()
N = len(xs)
lr = 50

X = F.one_hot(xs, vocab_size).float()
weights = torch.empty(size=(vocab_size, vocab_size), requires_grad=True)
torch.nn.init.uniform_(weights, a=-1.0, b=1.0)


for _ in range(250):
    weights.grad = None
    
    forward = X @ weights

    softmax_logits = torch.exp(forward)
    row_sums = softmax_logits.sum(dim=1, keepdim=True)
    probs = softmax_logits / row_sums

    loss = -torch.log(probs[torch.arange(N), ys]).mean()

    loss.backward()

    with torch.no_grad():
        weights -= lr * weights.grad

    if _ == 0:
        print(loss.item())
    if _ % 25 == 0:
        print(loss.item())


