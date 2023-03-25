import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

import math

class Unet(nn.Module):

    def __init__(self, T: int, input_channels: int = 4, output_channels: int = 3):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.embed64 = TimestampEmbedding2d(d=64, T=T)
        self.embed32 = TimestampEmbedding2d(d=32, T=T)
        self.embed16 = TimestampEmbedding2d(d=16, T=T)
        self.embed8 = TimestampEmbedding2d(d=8, T=T)

        # input size 64x64x4
        self.resnet1a = Resnet(input_channels, 128)
        self.resnet1b = Resnet(128, 128)
        # input size 32x32x128
        self.resnet2a = Resnet(128, 256)
        self.resnet2b = Resnet(256, 256)
        #input size 16x16x256
        self.resnet3a = Resnet(256, 512)
        self.resnet3b = Resnet(512, 512)
        #input size 8x8x512
        self.resnet4a = Resnet(512, 1024)
        self.resnet4b = Resnet(1024, 1024)
        # transpose_conv to 16x 16 x512
        self.trans_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        #input size = 16x 16
        self.resnet5a = Resnet(1024, 512)
        self.resnet5b = Resnet(512, 512)
        # transpose conv tp 32x32 x256
        self.trans_conv2= nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        #input_size = 32x32
        self.resnet6a = Resnet(512, 256)
        self.resnet6b = Resnet(256, 256)
        # transpose conv to 64 x 64 x128
        self.trans_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        #input_size = 64x64
        self.resnet7a = Resnet(256, 128)
        self.resnet7b = Resnet(128, output_channels)

    def forward(self, x: Tensor, y_tilde: Tensor, t: Tensor):
        X = torch.cat((x, y_tilde), dim=1) # (1, 4, 64, 64)

        X = self.embed64(X, t)
        X = self.resnet1a(X)
        X1 = self.resnet1b(X)

        X = self.max_pool(X1)

        X = self.embed32(X, t)
        X = self.resnet2a(X)
        X2 = self.resnet2b(X)

        X = self.max_pool(X2)

        X = self.embed16(X, t)
        X = self.resnet3a(X)
        X3 = self.resnet3b(X)

        X = self.max_pool(X3)

        X = self.embed8(X, t)
        X = self.resnet4a(X)
        X = self.resnet4b(X)

        X = self.trans_conv1(X)

        x = self.embed16(X, t)
        X = self.resnet5a(torch.cat((X, X3), dim=1))
        X = self.resnet5b(X)

        X = self.trans_conv2(X)

        x = self.embed32(X, t)
        X = self.resnet6a(torch.cat((X, X2), dim=1))
        X = self.resnet6b(X)

        X = self.trans_conv3(X)

        X = self.embed64(X, t)
        X = self.resnet7a(torch.cat((X, X1), dim=1))
        X = self.resnet7b(X)

        return X

class Resnet(nn.Module):

    def __init__(self, nc_in, nc_out):
        super().__init__()
        self.conv_up = nn.Conv2d(nc_in, nc_out, kernel_size=3, stride=1, padding='same')
        self.conv_same = nn.Conv2d(nc_out, nc_out, kernel_size=3, stride=1, padding='same')
        self.sing_conv= nn.Conv2d(nc_in, nc_out, kernel_size=1, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
    def forward(self, input):
        y = self.sing_conv(input)
        x = self.conv_up(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_same(x)
        x = self.bn2(x)
        x = self.relu(x+y)
        return x


class TimestampEmbedding2d(nn.Module):

    def __init__(self, d: int, T: int):
        super().__init__()

        pos = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d**2, 2) * (-math.log(10000) / d**2))
        embedding = torch.zeros(T, d**2)
        embedding[:, 0::2] = torch.sin(pos * div_term)
        embedding[:, 1::2] = torch.cos(pos * div_term)[:, :d**2 // 2]
        embedding = embedding.reshape((T, d, d))

        self.register_buffer("_embedding", embedding)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = x + self._embedding[t].unsqueeze(1)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    lr = 0.003
    batch_size = 256

    # Model Initialization
    model = Unet(100).to(device)
    x = torch.rand(size=(1, 3, 64, 64)).to(device)
    y = torch.rand(size=(1, 1, 64, 64)).to(device)
    t = torch.tensor([5]).long().to(device)
    print(model.forward(y, x, t).size())



if __name__ == "__main__":
    main()

