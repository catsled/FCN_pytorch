import torch.nn as nn

from config import *


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.H = Height
        self.W = Width
        self.conv1 = nn.Sequential(
            nn.Conv2d(Channels, 64, kernel_size=3, padding=1),  # H x W x 64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # H x W x 64
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2 x W/2 x 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # H/2 x W/2 x 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # H/2 x W/2 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/4 x W/4 x 128
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # H/4 x W/4 x 256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # H/4 x W/4 x 256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # H/4 x W/4 x 256
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/8 x W/8 x 256
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # H/8 x W/8 x 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # H/8 x W/8 x 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # H/8 x W/8 x 512
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/16 x W/16 x 512
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # H/16 x W/16 x 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # H/16 x W/16 x 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # H/16 x W/16 x 512
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/32 x W/32 x 512
        )

    def forward(self, x):
        x = self.conv1(x)
        pool1 = x
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        pool3 = x
        x = self.conv4(x)
        pool4 = x
        # x = self.conv5(x)
        # pool5 = x

        return pool1, pool2, pool3, pool4


class FCNs(nn.Module):

    def __init__(self, feature_map="vgg16"):
        super(FCNs, self).__init__()
        self.feature_map = VGG16()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )  # H/16 x W/16 x 512
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )  # H/8 x W/8 x 256
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )  # H/4 x W/4 x 128
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )  # H/2 x W/2 x 64
        # self.deconv5 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32)
        # )  # H x W x 32
        self.classifer = nn.Conv2d(32, NUMCLASSES, kernel_size=1)   # H x W x NUMCLASSES

    def forward(self, x):
        feature_maps = self.feature_map(x)
        x = self.deconv1(feature_maps[3])
        x = x + feature_maps[2]
        x = self.deconv2(x)
        x = x + feature_maps[1]
        x = self.deconv3(x)
        x = x + feature_maps[0]
        x = self.deconv4(x)
        # x = x + feature_maps[0]
        # x = self.deconv5(x)
        x = self.classifer(x)
        return x
