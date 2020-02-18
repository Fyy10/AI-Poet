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
        # input_seq: [1, 1]
        embedded = self.embedding(input_seq).view(1, 1, -1)
        # embedded: [1, 1, embedding_dim]
        output = F.relu(embedded)
        output, hidden = self.lstm(output, hidden)
        # output: [1, 1, hidden_size]
        output = self.softmax(self.out(output[0]))
        # output: [1, output_size (voc_size)]
        return output, hidden

    def init_hidden(self):
        h_0 = torch.randn(1, 1, self.hidden_size, device=Config.device)
        c_0 = torch.zeros(1, 1, self.hidden_size, device=Config.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=Config.max_gen_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(2 * self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, in_seq, hidden, encoder_outputs):
        # in_seq: [1, 1]
        embedded = self.embedding(in_seq).view(1, 1, -1)
        # embedded: [1, 1, hidden_size (embedding_dim)]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), dim=1)), dim=1)
        # attn_weights: [1, max_length]
        # encoder_outputs: [max_length, hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # attn_applied: [1, 1, hidden_size]
        output = self.attn_combine(torch.cat((embedded[0], attn_applied[0]), dim=1))
        output = F.relu(output.unsqueeze(0))
        # output: [1, 1, hidden_size]
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=Config.device)
