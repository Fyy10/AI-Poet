import Encoder
import Decoder
import torch
import torch.nn as nn
from config import *


class Seq2Seq(nn.Module):   # still some problems
    def __init__(self, voc_size, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder.EncoderRNN(input_size=voc_size,
                                          hidden_size=hidden_dim)
        self.decoder = Decoder.DecoderRNN(hidden_size=hidden_dim,
                                          output_size=voc_size)

    def forward(self, src, trg, hidden):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(Config.device)

        # last hidden state of encoder used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, None)
        # h_n/c_n: [num_layers, batch_size, hidden_dim]

        # first input to the decoder is <sos>
        in_seq = trg[0, :]
        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(in_seq, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            in_seq = top1
        return output, hidden


class CharRNN(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_dim):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=Config.num_layers)
        self.fc = nn.Linear(hidden_dim, voc_size)

    def forward(self, in_seq, hidden=None):
        seq_len, batch_size = in_seq.size()
        if hidden is None:
            h0 = in_seq.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c0 = in_seq.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h0, c0 = hidden
        embeds = self.embeddings(in_seq)
        # [seq_len, batch_size, embedding_dim]
        output, hidden = self.lstm(embeds, (h0, c0))
        # output: [seq_len, batch_size, hidden_dim]
        output = self.fc(output.view(seq_len * batch_size, -1))
        # output: [seq_len * batch_size, voc_size]
        return output, hidden
