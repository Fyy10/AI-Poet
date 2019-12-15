import Encoder
import Decoder
import torch.nn as nn


# Hyper parameters
INPUT_SIZE = 100
HIDDEN_SIZE = 50
OUTPUT_SIZE = 200


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder.EncoderRNN(input_size=INPUT_SIZE,
                                          hidden_size=HIDDEN_SIZE)
        self.decoder = Decoder.DecoderRNN(hidden_size=HIDDEN_SIZE,
                                          output_size=OUTPUT_SIZE)

    def forward(self, in_seq):
        state = self.encoder(in_seq, None)
        output = self.decoder(state, None)
        return output
