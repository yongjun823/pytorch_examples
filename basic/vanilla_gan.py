import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])
# MNIST dataset
mnist = datasets.MNIST(root='/tmp/data/',
                       train=True,
                       transform=transform,
                       download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=1000,
                                          shuffle=True)

D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh()
)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

bce_loss = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=3e-4)
g_optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)

for epoch in range(100):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images_var = to_var(images.view(batch_size, -1))

        real_label = to_var(torch.ones(batch_size))
        fake_label = to_var(torch.zeros(batch_size))

        d_real_output = D(images_var)
        d_loss_real = bce_loss(d_real_output, real_label)

        z = to_var(torch.randn(batch_size, 64))
        g_output = G(z)
        d_fake_output = D(g_output)
        d_loss_fake = bce_loss(d_fake_output, fake_label)

        d_loss = d_loss_fake + d_loss_real
        g_loss = bce_loss(d_fake_output, real_label)

        D.zero_grad()
        G.zero_grad()

        d_loss.backward(retain_graph=True)
        g_loss.backward(retain_graph=True)

        d_optimizer.step()
        g_optimizer.step()

        print('Epoch {}, Step{}, d_loss: {}, '
              'g_loss: {}, D(x): {}, D(G(z)): {}'
              .format(epoch, i + 1, d_loss.data[0], g_loss.data[0],
                      d_real_output.data.mean(), d_fake_output.data.mean()))

"""
Epoch 0, Step1, d_loss: 1.3998018503189087, g_loss: 0.6804010272026062, D(x): 0.4998370110988617, D(G(z)): 0.5064182877540588
Epoch 0, Step2, d_loss: 1.2256580591201782, g_loss: 0.686850368976593, D(x): 0.5910760164260864, D(G(z)): 0.5031628608703613
Epoch 0, Step3, d_loss: 1.0857747793197632, g_loss: 0.6925272941589355, D(x): 0.6760351061820984, D(G(z)): 0.5003151893615723
Epoch 0, Step4, d_loss: 0.9678806066513062, g_loss: 0.6976541876792908, D(x): 0.7567821145057678, D(G(z)): 0.4977574050426483
Epoch 0, Step5, d_loss: 0.8765388131141663, g_loss: 0.6987298130989075, D(x): 0.8282754421234131, D(G(z)): 0.4972229599952698
Epoch 0, Step6, d_loss: 0.8101100921630859, g_loss: 0.695343554019928, D(x): 0.8879945278167725, D(G(z)): 0.4989088773727417
Epoch 0, Step7, d_loss: 0.7685821056365967, g_loss: 0.6902465224266052, D(x): 0.9302460551261902, D(G(z)): 0.5014583468437195
Epoch 0, Step8, d_loss: 0.7426013946533203, g_loss: 0.6872588396072388, D(x): 0.9575506448745728, D(G(z)): 0.502958357334137
Epoch 0, Step9, d_loss: 0.7220649123191833, g_loss: 0.6882836818695068, D(x): 0.9763380885124207, D(G(z)): 0.5024437308311462
Epoch 0, Step10, d_loss: 0.7063350677490234, g_loss: 0.6936255693435669, D(x): 0.9864791035652161, D(G(z)): 0.49976783990859985
"""
