""" Basic implementatiojn of the NALU """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import math


class NeuralAccumulatorCell(nn.Module):

    def __init__(self, input_size, output_size):
        super(NeuralAccumulatorCell, self).__init__()
        self.W_ = nn.Parameter(torch.Tensor(output_size, input_size))
        self.M_ = nn.Parameter(torch.Tensor(output_size, input_size))

        nn.init.kaiming_uniform(self.W_, a=math.sqrt(5))
        nn.init.kaiming_uniform(self.M_, a=math.sqrt(5))

    def forward(self, x):
        W = F.tanh(self.W_) * F.sigmoid(self.M_)
        return F.linear(x, W)


class NeuralArithmeticLogicUnitCell(nn.Module):

    def __init__(self, input_size, output_size):
        super(NeuralArithmeticLogicUnitCell, self).__init__()
        self.G = nn.Parameter(torch.Tensor(output_size, input_size))
        self.add_nac = NeuralAccumulatorCell(input_size, output_size)
        self.mul_nac = NeuralAccumulatorCell(input_size, output_size)

        nn.init.kaiming_uniform(self.G, a=math.sqrt(5))

    def forward(self, x):
        g = F.sigmoid(F.linear(x, self.G))
        m = torch.exp(self.mul_nac(torch.log(torch.abs(x) + 1e-10)))

        add_sub = g * self.add_nac(x)
        mul_div = (1 - g)*m
        y = add_sub + mul_div
        return y


class Network(nn.Module):
    """ a basic neural network to be compared to the NALU """
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 8), nn.Sigmoid(), nn.Linear(8, 1))

    def forward(self, x):
        out = self.model(x)
        return out


def test(x, y):
    data = [[int(x), int(y)]]
    b = Variable(torch.FloatTensor(data))
    t = Variable(b[:, 0].data * b[:, 1].data)
    t.unsqueeze(1)

    out = model(b)

    cost = F.mse_loss(out, t)
    print(x, '*', y, '=', out.data.numpy()[0], 'cost', cost.data[0])

# model = nn.Sequential(NeuralArithmeticLogicUnitCell(100, 64), NeuralArithmeticLogicUnitCell(64, 32), NeuralArithmeticLogicUnitCell(32, 1))
model = nn.Sequential(NeuralArithmeticLogicUnitCell(2, 8), NeuralArithmeticLogicUnitCell(8, 8), NeuralArithmeticLogicUnitCell(8, 1))

# model = Network()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

iter_num = 1e5

batch_size = 50
for i in range(int(iter_num)):
    """ you can try both with single numbers or with input size 100 """
    
    """data = [[random.randint(5, 10) for i in range(100)] for j in range(batch_size)]
    x = Variable(torch.FloatTensor(data))
    a, b = x[:, :50].sum(dim=1), x[:, 50:].sum(dim=1)
    t = Variable(a.data) * Variable(b.data)"""

    data = [[random.randint(5, 10) for i in range(2)] for j in range(batch_size)]
    x = Variable(torch.FloatTensor(data))
    t = Variable(x[:, 0].data * x[:, 1].data)
    t.unsqueeze(1)

    out = model(x)
    cost = F.mse_loss(out, t)

    cost.backward()

    optimizer.step()
    optimizer.zero_grad()

    if i % 2000 == 0:
        print('step', i, 'cost', cost.data[0])  # , a.data[0], '*', b.data[0], '=', out.data[0, 0])


for i in range(50):
    # x, y = random.randint(10, 50), random.randint(10, 50)
    x = input('x ')
    y = input('y ')
    test(x, y)

# TODO: check wether it is better to have a nac for addition and another one for multiplication, or using the same one for both.
# TODO: experiment with lenth 100
# TODO: experiment with more complex functions
