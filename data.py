# This code is only used for testing the data
import numpy as np
import torch

DataSet = np.load('tang.npz', allow_pickle=True)
data = DataSet['data']
print(data.shape)
print(type(data))
ix2word = DataSet['ix2word'].item()
print(type(ix2word))
word2ix = DataSet['word2ix'].item()
print(type(word2ix))

input_ = data[23333][:-1]
target = data[23333][1:]

poem = ''
for i in input_:
    poem += ix2word[i]
print(poem)

poem = ''
for i in target:
    poem += ix2word[i]
print(poem)

poem = ''
for i in data[23333]:
    word = ix2word[i]
    if word != '</s>':
        poem += word
print(poem)
