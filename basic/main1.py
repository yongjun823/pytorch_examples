import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.MNIST(root='/Temp',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# Data Loader (this provides queue and thread in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

# Actual usage of data loader is as below.
# for images, labels in train_loader:
#     # Your training code will be written here
#     print(images.shape)

# alex_net = torchvision.models.alexnet(pretrained=True)
res_net = torchvision.models.resnet152(pretrained=True)

res_net.fc = nn.Linear(res_net.fc.in_features, 100)  # 100 is for example.

images = Variable(torch.randn(10, 3, 224, 224))
outputs = res_net(images)
print(outputs)  # (10, 100)
