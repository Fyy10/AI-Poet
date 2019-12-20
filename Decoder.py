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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden, cell):
        # hidden/cell: [num_layers, batch_size, hidden_dim]
        embedded = self.embedding(input_seq)
        print('embed', embedded.size())
        embedded = embedded.unsqueeze(0)    # [1, batch_size, N]
        context = embedded  # [1, batch_size, N]
        print('un squeeze embed', embedded.size())
        rnn_input = torch.cat([embedded, context], 2)
        print('rnn_input', rnn_input.size())

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output: [seq_len? = 1, batch_size, hidden_dim(N?)]
        # hidden/cell: [num_layers, batch_size, hidden_dim]
        print('hidden', hidden.size(), 'cell', cell.size())
        print(output.squeeze(0).size())
        print('output', output.size())

        output = output.squeeze(0)  # [batch_size, N]
        context = context.squeeze(0)
        prediction = self.fc(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        # prediction: [batch_size, output_dim]
        return output, hidden, cell
