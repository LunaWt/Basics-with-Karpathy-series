import numpy as np
import random

random.seed(42)

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

xs = np.array([x for word in words for x in word[:-1]])
ys = np.array([x for word in words for x in word[1:]])

x = [0.0 for _ in range(vocab_size)]
target_idx = stoi[ys[-1]]
x[stoi[xs[-1]]] = 1.0

logits = x @ weights

softmax_logits = np.exp(logits)
softmax_logits /= sum(softmax_logits)

print(target_idx)
print(stoi[xs[-1]])
loss = -np.log(softmax_logits[target_idx])
print(loss)
print(vocab)