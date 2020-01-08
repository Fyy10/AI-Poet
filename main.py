import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *
from config import *
from timer import *
from data import *
import time
import matplotlib.pyplot as plt
import random


def train_char_rnn():
    # get device
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = Config.device
    # fetch data
    dataset = np.load(Config.data_path, allow_pickle=True)
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    data = torch.from_numpy(data)
    data_loader = DataLoader(data, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    # model
    model = CharRNN(len(word2ix),
                    embedding_dim=Config.embedding_dim,
                    hidden_dim=Config.hidden_dim)
    model = model.to(device)
    print(model)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if Config.model_path:
        model.load_state_dict(torch.load(Config.model_path))

    learning_route = []

    # train
    for epoch in range(Config.epoch):
        for step, data_ in enumerate(data_loader):
            # data_ = [batch_size, seq_len]
            data_ = data_.long().transpose(1, 0).contiguous()   # data_ = [seq_len, batch_size]
            data_ = data_.to(device)

            optimizer.zero_grad()
            # input_ = 0..n-2 / target = 1..n-1
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            # output: [seq_len * batch_size, voc_size]
            # target: [seq_len, batch_size]
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                learning_route.append(loss.item())
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))
        torch.save(model.state_dict(), '%s_%s.pth' % (Config.model_prefix, epoch))
    torch.save(model.state_dict(), Config.model_path)
    print('Finished Training')

    # plot learning route
    plt.figure('Learning Route')
    plt.plot(learning_route)
    plt.xlabel('*100 steps')
    plt.ylabel('Loss')
    plt.show()


def train_seq2seq():
    # get device
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = Config.device
    # fetch data
    dataset = np.load(Config.data_path, allow_pickle=True)
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    data = torch.from_numpy(data)
    data_loader = DataLoader(data, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    # model
    encoder = Encoder.EncoderRNN(len(word2ix), Config.hidden_dim).to(Config.device)
    decoder = Decoder.DecoderRNN(Config.hidden_dim, len(word2ix)).to(Config.device)

    # optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    # criterion = nn.CrossEntropyLoss()
    # if Config.model_path:
    #     model.load_state_dict(torch.load(Config.model_path))
    #
    # learning_route = []

    # training iterations
    n_iters = 10000
    print_every = 100
    plot_every = 10
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # reset every print_every
    plot_loss_total = 0     # reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=Config.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=Config.learning_rate)

    data = data.numpy()
    training_pairs = [get_pair(random.choice(data)) for it in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                          criterion, word2ix)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_ave = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_ave))

        if iter % plot_every == 0:
            plot_loss_ave = plot_loss_total / plot_every
            plot_losses.append(plot_loss_ave)
            plot_loss_total = 0

    plt.plot(plot_losses)
    # todo: save model: encoder and decoder

    # train
    # for epoch in range(Config.epoch):
    #     for step, data_ in enumerate(data_loader):
    #         # data_ = [batch_size, seq_len]
    #         data_ = data_.long().transpose(1, 0).contiguous()   # data_ = [seq_len, batch_size]
    #         data_ = data_.to(device)
    #
    #         optimizer.zero_grad()
    #         # input_ = 0..n-2 / target = 1..n-1
    #         input_, target = data_[:-1, :], data_[1:, :]
    #         # input_, target: [seq_len, batch_size]
    #         output, _ = model(input_, target)
    #         print(output)
    #         print(target.view(-1))
    #         # output: [seq_len * batch_size, voc_size]
    #         # target: [seq_len, batch_size]
    #         loss = criterion(output, target.view(-1))
    #         loss.backward()
    #         optimizer.step()
    #         if step % 10 == 0:
    #             learning_route.append(loss.item())
    #         if step % 100 == 0:
    #             print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))
    #     torch.save(model.state_dict(), '%s_%s.pth' % (Config.model_prefix, epoch))
    # torch.save(model.state_dict(), Config.model_path)
    print('Finished Training')

    # plot learning route
    # plt.figure('Learning Route')
    # plt.plot(learning_route)
    # plt.xlabel('*10 steps')
    # plt.ylabel('Loss')
    # plt.show()


# todo: something wrong with training code here (decoder hidden size is not correct)
def train_step(input_tensor, target_tensor, encoder, decoder,
               encoder_optimizer, decoder_optimizer, criterion, word2ix, max_length=Config.max_gen_len):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=Config.device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[word2ix['<START>']]], device=Config.device)
    decoder_hidden = encoder_hidden

    # no teacher forcing
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()     # detach from history as input (?)

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == word2ix['<EOP>']:
            break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# todo: evaluating and generating poems
def evaluate(encoder, decoder, sentence, max_length = Config.max_gen_len):
    with torch.no_grad():
        pass


def main():
    if Config.model_type == 'CharRNN':
        train_char_rnn()
    elif Config.model_type == 'Seq2Seq':
        train_seq2seq()
    else:
        raise TypeError('Wrong model type!')


if __name__ == '__main__':
    main()
