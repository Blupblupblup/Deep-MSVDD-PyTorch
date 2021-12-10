import torch
import torch.nn as nn
import torch.nn.functional as F

"""
architectures from:
- https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/networks/mnist_LeNet.py
- https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/networks/fmnist_LeNet.py
- https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/networks/cifar10_LeNet.py
one should note that F.leaky_relu() uses leakiness alpha = 0.01 and not 0.1 as indicated in the paper http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
"""

#############
### MNIST ###
#############

class MNIST_LeNet(nn.Module):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class MNIST_LeNet_Decoder(nn.Module):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x.squeeze()


class MNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MNIST_LeNet(rep_dim=rep_dim)
        self.decoder = MNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

####################
### FashionMNIST ###
####################

class FashionMNIST_LeNet(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x


class FashionMNIST_LeNet_Decoder(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc3 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(8, 32, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(128 / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x.squeeze()


class FashionMNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = FashionMNIST_LeNet(rep_dim=rep_dim)
        self.decoder = FashionMNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

###############
### CIFAR10 ###
###############


class CIFAR10_LeNet(nn.Module):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        # x = x.view(-1, 3, 32, 32)
        x = torch.transpose(x,1,3)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class CIFAR10_LeNet_Decoder(nn.Module):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return torch.transpose(x,1,3)


class CIFAR10_LeNet_Autoencoder(nn.Module):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x