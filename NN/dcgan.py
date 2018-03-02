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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.fc1 = nn.Linear(12 * 12 * 128, 1)

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([100, 16, 24, 24])
        out = self.conv2(out)  # torch.Size([100, 32, 20, 20])
        out = self.conv3(out)  # torch.Size([100, 64, 16, 16])
        out = self.conv4(out)  # torch.Size([100, 128, 12, 12])
        out = out.view(out.size(0), -1)  # torch.Size([100, 18432])
        out = self.fc1(out)  # torch.Size([100, 1])
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=5),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([100, 16, 13, 13])
        out = self.conv2(out)  # torch.Size([100, 32, 16, 16])
        out = self.conv3(out)  # torch.Size([100, 64, 19, 19])
        out = self.conv4(out)  # torch.Size([100, 1, 28, 28])
        return out


if torch.cuda.is_available():
    D = Discriminator().cuda()
    G = Generator().cuda()
else:
    D = Discriminator()
    G = Generator()

bce_loss = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(100):
    for i, (img, _) in enumerate(data_loader):
        batch_size = img.size(0)

        # torch constant
        img_var = to_var(img)
        real_label = to_var(torch.ones(batch_size))
        fake_label = to_var(torch.zeros(batch_size))

        # 분류기 학습
        d_real = D(img_var)
        D_real_loss = bce_loss(d_real, real_label)

        z = to_var(torch.randn(batch_size, 1, 10, 10))
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
        z = to_var(torch.randn(batch_size, 1, 10, 10))
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
