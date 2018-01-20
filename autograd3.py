import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x = np.arange(1, 10, 0.1, dtype=np.float32).reshape((90, 1))
w = 2
y = w * x

X = Variable(torch.from_numpy(x))
Y = Variable(torch.randn(90, 1))

linear = nn.Linear(1, 1)
error = nn.MSELoss()
optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)

for i in range(100):
    prediction = linear(X)
    loss = error(prediction, Y)
    print(loss.data[0])

    loss.backward()
    optimizer.step()
