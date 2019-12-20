import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *
from config import *
import matplotlib.pyplot as plt


def train():
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
    # model = Seq2Seq(len(word2ix), Config.hidden_dim)
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


def main():
    train()


if __name__ == '__main__':
    main()
