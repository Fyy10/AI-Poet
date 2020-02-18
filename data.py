import numpy as np
import torch
from config import *

DataSet = np.load('tang.npz', allow_pickle=True)
data = DataSet['data']
ix2word = DataSet['ix2word'].item()
word2ix = DataSet['word2ix'].item()


def poem2input(in_poem):
    start_ix = 0
    while ix2word[in_poem[start_ix]] != '<START>':
        start_ix += 1
    end = start_ix
    while end < len(in_poem) and ix2word[in_poem[end]] != '，':
        end += 1
    in_arr = in_poem[start_ix + 1: end]
    in_arr = in_arr.tolist()
    in_arr.append(word2ix['<EOP>'])
    in_arr.reverse()
    in_tensor = torch.tensor(in_arr)
    in_tensor = in_tensor.view(-1, 1)
    return in_tensor.long().to(Config.device)


def poem2target(in_poem):
    start_ix = 0
    while ix2word[in_poem[start_ix]] != '，':
        start_ix += 1
    end = start_ix
    while end < len(in_poem) and ix2word[in_poem[end]] != '。':
        end += 1
    target_arr = in_poem[start_ix + 1: end]
    target_arr = target_arr.tolist()
    target_arr.append(word2ix['<EOP>'])
    target_tensor = torch.tensor(target_arr)
    # target_tensor = torch.from_numpy(target_arr)
    target_tensor = target_tensor.view(-1, 1)
    return target_tensor.long().to(Config.device)


def get_pair(in_poem):
    return poem2input(in_poem), poem2target(in_poem)


if __name__ == '__main__':
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

    print(poem2input(data[23333]))
    print([ix2word[ix.item()] for ix in poem2input(data[23333])])

    print(poem2target(data[23333]))
    print([ix2word[ix.item()] for ix in poem2target(data[23333])])
