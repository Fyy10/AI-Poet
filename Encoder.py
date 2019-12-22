import torch
import torch.nn as nn
from config import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim=Config.embedding_dim)
        self.lstm = nn.LSTM(Config.embedding_dim, hidden_size, num_layers=Config.num_layers)

    def forward(self, input_seq, hidden=None):
        seq_len, batch_size = input_seq.size()
        # input_seq: [seq_len, batch_size]
        if hidden is None:
            h_0 = input_seq.data.new(Config.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = input_seq.data.new(Config.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embedding(input_seq)      # embeds: [seq_len, batch_size, embed_dim]
        output, (h_n, c_n) = self.lstm(embeds, (h_0, c_0))
        # output: [seq_len, batch_size, hidden_dim], h_n/c_n: [num_layers, batch_size, hidden_dim]
        return output, (h_n, c_n)
