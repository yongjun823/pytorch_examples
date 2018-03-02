import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image


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
                                          batch_size=100,
                                          shuffle=True)

# 분류기
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())

# 생성기
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh())

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

bce_loss = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(100):
    for i, (img, _) in enumerate(data_loader):
        batch_size = img.size(0)

        # torch constant
        img_var = to_var(img.view(batch_size, -1))
        real_label = to_var(torch.ones(batch_size))
        fake_label = to_var(torch.zeros(batch_size))

        # 분류기 학습
        d_real = D(img_var)
        D_real_loss = bce_loss(d_real, real_label)

        z = to_var(torch.randn(batch_size, 64))
        fake_img = G(z)
        d_fake = D(fake_img)
        D_fake_loss = bce_loss(d_fake, fake_label)

        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        d_optimizer.step()

        if i % 100 == 0:
            print('분류기 || Epoch [%d/%d], Step[%d/%d], D(x): %.4f, '
                  % (epoch, 100, i + 1, 600, d_real.data.mean()))

        # 생성기 학습
        z = to_var(torch.randn(batch_size, 64))
        fake_img = G(z)
        d_fake = D(fake_img)
        G_loss = bce_loss(d_fake, real_label)

        D.zero_grad()
        G.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print('생성기 || Epoch [%d/%d], Step[%d/%d], D(G(z)): %.4f, \n\n'
                  % (epoch, 100, i + 1, 600, d_fake.data.mean()))

    fake_images = fake_img.view(fake_img.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './fake_images-%d.png' % (epoch + 1))
