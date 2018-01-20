import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

error = nn.MSELoss()
optimizer = torch.optim.ASGD(linear.parameters(), lr=0.01)

prediction = linear(x)
loss = error(prediction, y)
print(loss.data[0])

# back propagation
loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

optimizer.step()

prediction = linear(x)
loss = error(prediction, y)
print(loss.data[0])
