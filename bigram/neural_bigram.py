import numpy as np

np.random.seed(42)

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

weights = np.random.uniform(-1.0, 1.0, size=(vocab_size, vocab_size))

xs = np.array([stoi[x] for word in words for x in word[:-1]])
ys = np.array([stoi[y] for word in words for y in word[1:]])
N = len(xs)

X = np.zeros((N, vocab_size))
X[np.arange(N), xs] = 1

logits = X @ weights

softmax_logits = np.exp(logits)
softmax_logits /= np.sum(softmax_logits, axis=1, keepdims=True)

loss = -np.log(softmax_logits[np.arange(N), ys]).mean()

print(loss)
print(vocab)




for i, item in enumerate(xs):
    if xs[i] == 24:
        print(logits[i] == weights[24])
        break

print(xs[:5])