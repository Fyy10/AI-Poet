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


def generate_seq2seq(encoder, decoder, sentence, ix2word, word2ix, max_length=Config.max_gen_len):
    with torch.no_grad():
        indexes = [word2ix[w] for w in sentence]
        indexes.append(word2ix['<EOP>'])
        indexes.reverse()
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=Config.device).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=Config.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[word2ix['<START>']]], device=Config.device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == word2ix['<EOP>']:
                decoded_words.append('<EOP>')
                break
            else:
                decoded_words.append(ix2word[topi.item()])
            decoder_input = topi.detach()
        return decoded_words
