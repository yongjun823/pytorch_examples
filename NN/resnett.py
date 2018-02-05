import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt

num_epochs = 5
learning_rate = 0.01
batch_size = 16

# Image processing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

train_data_set = dsets.CIFAR10(root='/tmp',
                               train=True,
                               transform=transform,
                               download=True)

test_data_set = dsets.CIFAR10(root='/tmp',
                              train=False,
                              transform=transform)

# # Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data_set,
                                          batch_size=batch_size,
                                          shuffle=False)

# for images, labels in train_loader:
#     d = images[0].numpy()
#     d = d.reshape((224, 224, 3))
#     plt.imshow(d)
#     plt.show()
#     print(d.shape)
#     break

resnet = models.resnet18(pretrained=True).cuda()

# If you want to finetune only top layer of the model.
for param in resnet.parameters():
    param.requires_grad = False

# Replace top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 10).cuda()  # 100 is for example.

cross_entropy = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, resnet.parameters()),
                 lr=learning_rate)

for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        img_var = Variable(images).cuda()
        label_var = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = resnet(img_var)
        loss = cross_entropy(outputs, label_var)
        loss.backward()
        optimizer.step()

        if (idx + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, idx + 1, len(train_data_set) // batch_size, loss.data[0]))

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(resnet.state_dict(), 'model_resnet.pkl')
