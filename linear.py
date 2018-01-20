import numpy as np

w = 3.0
x = np.arange(1, 5, 0.01)
y = x * w
learning_rate = 0.01
w_ = 2


def loss(y_, y):
    return (y - y_) ** 2


for i in range(1000):
    pass
