import torch
from models import *
from config import *
from gen_poem import *


def write_poem():
    print('Loading model...')
    dataset = np.load(Config.data_path, allow_pickle=True)
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    model = CharRNN(len(ix2word), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(torch.load(Config.model_path, Config.device))
    print('Done!')
    while True:
        start_words = str(input())
        gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix, Config.prefix_words))
        print(gen_poetry)


write_poem()
