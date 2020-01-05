import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from models import *
from config import *


# given the first sentence
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    if type(model) == CharRNN:
        poetry = generate_char_rnn(model, start_words, ix2word, word2ix, prefix_words)
    elif type(model) == Seq2Seq:
        poetry = generate_seq2seq(model, start_words, ix2word, word2ix, prefix_words)
    else:
        raise TypeError('Wrong model type!')
    return poetry


def generate_char_rnn(model, start_words, ix2word, word2ix, prefix_words=None):
    poetry = list(start_words)
    len_start_words = len(start_words)
    # the first word should be <START>
    in_seq = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    in_seq = in_seq.to(Config.device)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(in_seq, hidden)
            in_seq = in_seq.data.new([word2ix[word]]).view(1, 1)

    for i in range(Config.max_gen_len):
        output, hidden = model(in_seq, hidden)
        if i < len_start_words:
            w = poetry[i]
            in_seq = in_seq.data.new([word2ix[w]]).view(1, 1)
        else:
            # get top index of word
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            poetry.append(w)
            in_seq = in_seq.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del poetry[-1]
            break
        if w == '。' or w == '？':
            poetry.append('\n')
    if poetry[-1] == '\n':
        del poetry[-1]
    return poetry


def generate_seq2seq(model, start_words, ix2word, word2ix, prefix_words=None):
    poetry = list(start_words)
    len_start_words = len(start_words)
    # the first word should be <START>
    in_seq = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    in_seq = in_seq.to(Config.device)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(in_seq, hidden)
            in_seq = in_seq.data.new([word2ix[word]]).view(1, 1)

    for i in range(Config.max_gen_len):
        pre_poetry = ['<START>'] + poetry
        poetry_tensor = torch.Tensor([word2ix[word] for word in pre_poetry]).view(len(pre_poetry), -1).long()
        output, hidden = model(poetry_tensor, in_seq, hidden)
        if i < len_start_words:
            w = poetry[i]
            in_seq = in_seq.data.new([word2ix[w]]).view(1, 1)
        else:
            # get top index of word
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            poetry.append(w)
            in_seq = in_seq.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del poetry[-1]
            break
        # if w == '。' or w == '？':
        #     poetry.append('\n')
    if poetry[-1] == '\n':
        del poetry[-1]
    return poetry
