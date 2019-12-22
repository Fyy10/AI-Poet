import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, Config.embedding_dim)
        self.lstm = nn.LSTM(Config.embedding_dim, hidden_size, num_layers=Config.num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden=None):
        seq_len, batch_size = input_seq.size()
        # input_seq: [seq_len, batch_size]
        if hidden is None:
            h_0 = input_seq.data.new(Config.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = input_seq.data.new(Config.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # hidden/cell: [num_layers, batch_size, hidden_dim]
        embedded = self.embedding(input_seq)
        # embedded: [seq_len, batch_size, embedding_dim]

        output, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
        # output: [seq_len, batch_size, hidden_dim]
        # hidden/cell: [num_layers, batch_size, hidden_dim]
        prediction = self.fc(output.view(seq_len * batch_size, -1))
        # prediction: [seq_len * batch_size, output_dim (voc_size)]
        return prediction, (h_n, c_n)
