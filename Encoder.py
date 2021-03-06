import torch
import torch.nn as nn
from config import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim=Config.embedding_dim)
        self.lstm = nn.LSTM(Config.embedding_dim, hidden_size)

    def forward(self, input_seq, hidden=None):
        # input_seq: [1]
        embedded = self.embedding(input_seq).view(1, 1, -1)
        # embedded: [1, 1, embedding_dim]
        output = embedded
        output, hidden = self.lstm(output, hidden)
        # output: [1, 1, hidden_size]
        # hidden: (h_n: [1, 1, hidden_size], c_n: [1, 1, hidden_size])
        return output, hidden

    def init_hidden(self):
        h_0 = torch.randn(1, 1, self.hidden_size, device=Config.device)
        c_0 = torch.zeros(1, 1, self.hidden_size, device=Config.device)
        return h_0, c_0
