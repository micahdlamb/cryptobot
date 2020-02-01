import torch
from torch import nn
from torch import optim

import pandas as pd

import matplotlib.pyplot as plt

epochs = 100

input_len = 32
label_len = 8
stride = 8

lr = 0.0003
buy_cutoff = .2

trade_fee=.001


def iter_data(data):
    span = input_len + label_len
    for i in range(0, len(data) - span + 1, stride):
        div = data[i:i+input_len+label_len].clone()
        input = div[:input_len].transpose(0,1)
        label = div[input_len:input_len+label_len].transpose(0,1)[0]

        _min = input[0].min().item()
        _max = input[0].max().item()
        _price = input[0][-1].item()
        input[0] = ((input[0] - _min) / (_max - _min)) * 2 - 1
        label    = ((label - _min) / (_max - _min)) * 2 - 1

        def to_gain(change, _min=_min, _max=_max, _price=_price):
            return change * .5 * (_max - _min) / _price

        yield input.unsqueeze(0).cuda(), label.unsqueeze(0).cuda(), to_gain


train, test = [], []
for file in "BCH_USDT.csv ETH_USDT.csv LTC_USDT.csv XRP_USDT.csv".split():
    df = pd.read_csv("data/"+file, names=['time','price','ob'])
    tensor = torch.from_numpy(df[['price', 'ob']].values).float()
    split = int(tensor.shape[0] * .8)
    train.extend(iter_data(tensor[:split]))
    test.extend(iter_data(tensor[split:]))

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# model = nn.Sequential(
#     nn.Conv1d(2, 8, kernel_size=7, stride=4, padding=3),
#     nn.ReLU(),
#     nn.Conv1d(8, 16, kernel_size=7, stride=4, padding=3),
#     nn.ReLU(),
#     # nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3),
#     # nn.ReLU(),
#     Reshape(1, 32),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 8)
# )

model = nn.Sequential(
    Reshape(1, input_len*2),
    nn.Linear(input_len*2, input_len),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(input_len, input_len),
    nn.ReLU(),
    nn.Linear(input_len, label_len)
)

model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for input, label, to_gain in train:
        output = model(input)
        loss = criterion(output, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    gains = []
    worked = []
    with torch.no_grad():
        test_loss = 0
        for input, label, to_gain in test:
            output = model(input)
            loss = criterion(output, label)
            test_loss += loss.item()

            last_input_price = input[0][0][-1].item()
            predict_changed = output[0][-1].item() - last_input_price
            actual_change = label[0][-1].item() - last_input_price
            if predict_changed > buy_cutoff:
                gain = to_gain(actual_change)
                gains.append(gain)
                worked.append(gain > 0)

    print(f"Epoc {epoch}",
          f"train_loss={round(train_loss*1e4/len(train))}",
          f"test_loss={round(test_loss*1e4/len(test))}",
          f"Worked {sum(worked)}/{len(worked)}",
          f"Gain {sum(gains)*100:.1f}%",
          f"Fees {len(worked)*trade_fee*2*100:.1f}%")


# for input, label, to_gain in test:
#     y = input[0][0].cpu()
#     ob = input[0][1].cpu()
#     lbl = label.view(-1).cpu()
#     model.eval()
#     with torch.no_grad():
#         output = model(input).cpu().view(-1)
#
#     if output[-1] - y[-1] > buy_cutoff:
#         plt.plot(range(len(y)), y, label='prices')
#         plt.plot(range(len(y)), ob, label='ob', linestyle=':')
#         plt.plot(range(len(y), len(y)+len(lbl)), lbl, label='actual')
#         plt.plot(range(len(y), len(y)+len(output)), output, label='predicted', linestyle='--')
#         plt.legend()
#         plt.show()

