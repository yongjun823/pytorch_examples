from __future__ import print_function
import torch

x = torch.Tensor(2, 3)
print(x)

y = torch.rand(2, 3)
print(y)

print(torch.add(x, y))

ones = torch.ones(10)
print(ones.numpy())

import numpy as np

a = np.ones(4)
b = torch.from_numpy(a)

print(b)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)
