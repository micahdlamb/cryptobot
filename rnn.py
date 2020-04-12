import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

epochs = 500
lr = 0.001

avg_pool = 1
seq_len = 48
predict_steps = 12
batch_size = 8

hidden_dim = 512
layers = 2

buy_cutoff = .0001
trade_fee=.001


def batch_data(items, seq_len, batch_size):
    num_batches = (len(items)-1) // (seq_len * batch_size)
    keep = num_batches*seq_len*batch_size
    print(f"Discarding last {len(items) - keep} words")
    features = items[:keep]   .view(batch_size, num_batches, seq_len, -1).transpose(0,1).transpose(1,2).cuda()
    targets  = items[1:keep+1].view(batch_size, num_batches, seq_len, -1).transpose(0,1).transpose(1,2).cuda()
    # return [*zip(features, targets[:,-1])]
    return [*zip(features, targets)]


class RNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=.3)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(.3)

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)
        # print(x.shape, [h.shape for h in hidden])
        # x = x[-1, :, :]  # Only keep last sequence item output
        y = y.view(-1, hidden_dim)
        y = self.dropout(y)
        y = self.fc(y)
        return y.view(x.shape), hidden


if __name__ == '__main__':
    # scp root@hwsrv-209945.hostwindsdns.com:/root/cryptobot/data/* data/*
    df = pd.read_csv("data/BTC_USDT.csv", names=['time','price','ob', 'vol'])
    df['price'] /= df['price'].mean()
    df['price'] -= 1
    df['vol'] /= df['vol'].mean()
    df['vol'] -= 1
    tensor = torch.from_numpy(df[['price', 'ob', 'vol']].values).float()
    if avg_pool > 1:
        tensor = F.avg_pool1d(tensor.unsqueeze(0).transpose(1,2), avg_pool).transpose(1,2).squeeze()

    split = int(tensor.shape[0] * .8)
    train = batch_data(tensor[:split], seq_len, batch_size)
    test  = batch_data(tensor[split:], seq_len, batch_size)

    # tensor = torch.tensor([[np.sin(x)] for x in np.linspace(-5000, 5000, 50001)])
    # train = test = batch_data(tensor, seq_len, batch_size)

    model = RNN(tensor.shape[-1]).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    try :
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            hidden = None
            random.shuffle(train)
            for x, target in train:
                if hidden:
                    hidden = tuple(h.detach() for h in hidden)

                y, hidden = model(x, hidden)
                loss = criterion(y, target)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_loss = 0
                hidden = None
                for x, target in test:
                    y, hidden = model(x, hidden)
                    loss = criterion(y, target)
                    test_loss += loss.item()

                gains = []
                worked = []
                keep_prices = []
                keep_predict = []
                keep_actual = []
                for i, (x, target) in enumerate(train[:-1]):
                    y, hidden = model(x, None)
                    y = y[-1:, :, :]

                    next_batch = train[i+1][0]
                    predict = [y[0,:,0]]
                    actual  = [next_batch[0,:,0]]

                    for j in range(predict_steps):
                        y, hidden = model(y, hidden)
                        assert y.shape[0] == 1
                        predict.append(y[0, :, 0])
                        actual.append(next_batch[j+1, :, 0])

                    for b in range(y.shape[1]):
                        if predict[-1][b] - predict[0][b] > buy_cutoff:
                            gain = actual[-1][b] - actual[0][b]
                            gains.append(gain)
                            worked.append(gain > 0)
                            keep_prices.append(x[:,b,0].tolist())
                            keep_predict.append([t[b].tolist() for t in predict])
                            keep_actual.append([t[b].tolist() for t in actual])

            print(f"Epoc {epoch}",
                  f"train_loss={round(train_loss*1e4/len(train))}",
                  f"test_loss={round(test_loss*1e4/len(test))}",
                  f"worked {sum(worked)}/{len(worked)}",
                  f"gain {sum(gains)*100:.1f}%",
                  f"fees {len(worked)*trade_fee*2*100:.1f}%")

    except KeyboardInterrupt  as e:
        pass

for prices, predict, actual in zip(keep_prices, keep_predict, keep_actual):
    plt.plot(range(len(prices)+len(actual)), prices+actual, label='prices')
    plt.plot(range(len(prices), len(prices)+len(predict)), predict, label='predict')
    plt.legend()
    plt.show()