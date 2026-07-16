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

probability_matrix = [[0.1 for _ in range(vocab_size)] for __ in range(vocab_size)]

for word in words:
    for x, y in zip(word[:-1], word[1:]):
        probability_matrix[stoi[x]][stoi[y]] += 1.0

loss = 0.0
n = 0
# -log(P(correct))


for idx, itm in enumerate(probability_matrix):
    
    itm_sum = sum(probability_matrix[idx])
    probability_matrix[idx] = [num / itm_sum for num in itm]

print(sum(probability_matrix[0]))

for word in words:
    for x, y in zip(word[:-1], word[1:]): 
        loss += -np.log(probability_matrix[stoi[x]][stoi[y]])
        n += 1

print(loss / n)


dot = '.'
generated_words = []
generations = 5

for generation in range(generations):
    word = [dot]
    while True:
        index = stoi[word[-1]]
        row = probability_matrix[index]
        indices = [idx for idx, _ in enumerate(row)]
        prob = row
        choice = random.choices(indices, prob, k=1)[0]

        word.append(itos[choice])

        if word[-1] == '.':
            generated_words.append(''.join(word))
            break

print(generated_words)
        
