# This code is only used for testing the data
import numpy as np

DataSet = np.load('tang.npz', allow_pickle=True)
data = DataSet['data']
print(data.shape)
print(type(data))
ix2word = DataSet['ix2word'].item()
print(type(ix2word))
word2ix = DataSet['word2ix'].item()
print(type(word2ix))
