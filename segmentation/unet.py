import torch
import torch.nn as nn
from torchvision import transforms


class MainConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.main_conv(inputs)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = MainConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        drop = self.conv(inputs)
        x = self.pool(drop)

        return drop, x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        two_out_channels = out_channels + out_channels
        self.conv = MainConv(two_out_channels, out_channels)

    def forward(self, inputs, drop):
        upper = self.up(inputs)
        concated = torch.cat([upper, drop], axis=1)
        x = self.conv(concated)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = Encoder(3, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        self.middle = MainConv(512, 1024)

        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)

        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        drop1, pool1 = self.down1(inputs)
        drop2, pool2 = self.down2(pool1)
        drop3, pool3 = self.down3(pool2)
        drop4, pool4 = self.down4(pool3)

        middle = self.middle(pool4)

        upper1 = self.up1(middle, drop4)
        upper2 = self.up2(upper1, drop3)
        upper3 = self.up3(upper2, drop2)
        upper4 = self.up4(upper3, drop1)

        outputs = self.outputs(upper4)

        return outputs
