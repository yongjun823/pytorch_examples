import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam

num_epochs = 5
learning_rate = 0.01
batch_size = 100

train_dataset = dsets.MNIST(root='/tmp/data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='/tmp/data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

Dnn = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
).cuda()

cross_entropy = nn.CrossEntropyLoss()
optimizer = Adam(Dnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        img_var = Variable(images.view(-1, 28 * 28)).cuda()
        label_var = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = Dnn(img_var)

        loss = cross_entropy(outputs, label_var)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28)).cuda()
    outputs = Dnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(Dnn.state_dict(), 'model_dnn.pkl')
