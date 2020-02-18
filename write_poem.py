import torch
from models import *
from config import *
from gen_poem import *


def write_poem():
    print('Loading model...')
    dataset = np.load(Config.data_path, allow_pickle=True)
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    model = encoder = decoder = None
    if Config.model_type == 'CharRNN':
        model = CharRNN(len(ix2word), Config.embedding_dim, Config.hidden_dim)
        model.load_state_dict(torch.load(Config.model_path, Config.device))
    elif Config.model_type == 'Seq2Seq':
        encoder = Encoder.EncoderRNN(len(word2ix), Config.hidden_dim).to(Config.device)
        decoder = Decoder.AttnDecoderRNN(Config.hidden_dim, len(word2ix)).to(Config.device)
        encoder.load_state_dict(torch.load('%s_%s.pth' % (Config.model_prefix, 'encoder')))
        decoder.load_state_dict(torch.load('%s_%s.pth' % (Config.model_prefix, 'decoder')))
    else:
        raise TypeError('Wrong model type!')
    print('Done!')
    while True:
        start_words = str(input())
        gen_poetry = ''
        if Config.model_type == 'CharRNN':
            gen_poetry = ''.join(generate_char_rnn(model, start_words, ix2word, word2ix, Config.prefix_words))
        elif Config.model_type == 'Seq2Seq':
            gen_poetry = ''.join(generate_seq2seq(encoder, decoder, start_words, ix2word, word2ix))
        print(gen_poetry)


write_poem()
