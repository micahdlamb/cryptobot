import torch
from torch import nn
from torch import optim

import pandas as pd

import matplotlib.pyplot as plt

def normalize(prices):
    _min = prices.min()
    _max = prices.max()
    return ((prices - _min) / (_max - _min))*2 - 1

def iter_data(data, input_len=32, label_len=8):
    data[:, 0] = normalize(data[:, 0])
    span = input_len + label_len
    for i in range(0, len(data) - span + 1, label_len):
        j = i+input_len
        input = data[i:j].transpose(0,1)
        label = data[j:j+label_len].transpose(0,1)[0]
        last_price = input[0][-1].item()
        input[0] -= last_price
        label -= last_price
        yield input.unsqueeze(0).cuda(), label.unsqueeze(0).cuda()

df = pd.read_csv("data/XRP_USDT.csv", names=['time','price','ob'])
tensor = torch.from_numpy(df[['price', 'ob']][::-1].copy().values).float()
split = int(tensor.shape[0] * .8)
train = list(iter_data(tensor[:split]))
test  = list(iter_data(tensor[split:]))

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

model = nn.Sequential(
    nn.Conv1d(2, 8, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    Reshape(1, 32*4),
    nn.Linear(32*4, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8)
)
model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):

    model.train()
    train_loss = 0
    for input, label in train:
        output = model(input)
        loss = criterion(output, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for input, label in test:
            output = model(input)
            loss = criterion(output, label)
            test_loss += loss.item()

    print(f"Epoc {epoch}",
          f"train_loss={round(train_loss*1e4/len(train))}",
          f"test_loss={round(test_loss*1e4/len(test))}")

worked = []
for input, label in test:
    y = input[0][0].cpu()
    ob = input[0][1].cpu()
    lbl = label.view(-1).cpu()
    model.eval()
    with torch.no_grad():
        output = model(input).cpu().view(-1)

    actual_up  = lbl[-1] - lbl[0] > 0
    predict_up = output[-1] - output[0] > 0
    worked.append(actual_up == predict_up)

    # plt.plot(range(len(y)), y, label='prices')
    # plt.plot(range(len(y)), ob, label='ob', linestyle=':')
    # plt.plot(range(len(y), len(y)+len(lbl)), lbl, label='actual')
    # plt.plot(range(len(y), len(y)+len(output)), output, label='predicted', linestyle='--')
    # plt.legend()
    # plt.show()

print(f"Worked {sum(worked)} / {len(worked)} times")

