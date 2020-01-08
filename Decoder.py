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
        self.lstm = nn.LSTM(Config.embedding_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        h_0 = torch.zeros(1, 1, self.hidden_size, device=Config.device)
        c_0 = torch.zeros(1, 1, self.hidden_size, device=Config.device)
