import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


num_epochs = 5
batch_size = 300
learning_rate = 0.001
input_size = 784
hidden_size = [520, 200, 50, 200, 520]

train_dataset = dsets.MNIST(root='/tmp/data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='/tmp/data/',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# CNN Model (2 conv layer)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder1 = nn.Linear(input_size, hidden_size[0])
        self.encoder2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.encoder3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.decoder1 = nn.Linear(hidden_size[2], hidden_size[3])
        self.decoder2 = nn.Linear(hidden_size[3], hidden_size[4])
        self.decoder3 = nn.Linear(hidden_size[4], input_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.encoder1(x))
        out = self.relu(self.encoder2(out))
        out = self.relu(self.encoder3(out))
        out = self.relu(self.decoder1(out))
        out = self.relu(self.decoder2(out))
        out = self.decoder3(out)

        return out


model = Model()
model.cuda()

mes_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, input_size)).cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = mes_loss(outputs, images)
        loss.backward()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

            # print(outputs.size(0))
            # auto_images = outputs.view(outputs.size(0), 1, 28, 28)
            # save_image(denorm(auto_images.data), './auto_images-%d.png' % (epoch + 1))
