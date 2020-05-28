# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

from network.vgg16_bn import vgg16_bn, init_weights
from network.resnet import ResNet50

from config import *

global cfg
cfg = init_cfg()


class upsample(nn.Module):
    def __init__(self, input_1, input_2, output):
        super(upsample, self).__init__()
        self.conv1 = nn.Conv2d(input_1 + input_2, input_2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(input_2)
        self.conv2 = nn.Conv2d(input_2, output, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UHT_Net(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(UHT_Net, self).__init__()

        if cfg.backbone == 'vgg16_bn':
            self.basenet = vgg16_bn(pretrained, freeze)
            self.upconv1 = upsample(1024, 512, 256)
            self.upconv2 = upsample(512, 256, 128)
            self.upconv3 = upsample(256, 128, 64)
            self.upconv4 = upsample(128, 64, 32)
            self.biggen = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
        elif cfg.backbone == 'resnet_50':
            self.basenet = ResNet50()
            self.upconv1 = upsample(2048, 1024, 512)
            self.upconv2 = upsample(512, 512, 256)
            self.upconv3 = upsample(256, 256, 128)
            self.upconv4 = upsample(128, 64, 32)
            self.biggen = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear')
            )

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        sources = self.basenet(x)

        if cfg.backbone == 'vgg16_bn':
            y = torch.cat([sources[0], sources[1]], dim=1)
            y = self.upconv1(y)

            y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[3]], dim=1)
            y = self.upconv3(y)

            y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[4]], dim=1)
            feature = self.upconv4(y)

            y = self.biggen(feature)
            y = self.conv_cls(y)

        elif cfg.backbone == 'resnet_50':
            y1 = F.interpolate(sources[4], size=sources[3].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y1, sources[3]], dim=1)
            y = self.upconv1(y)

            y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[1]], dim=1)
            y = self.upconv3(y)

            y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[0]], dim=1)
            feature = self.upconv4(y)

            y = self.biggen(feature)
            y = self.conv_cls(y)

        return y

if __name__ == '__main__':
    model = UHT_Net(pretrained=True).cuda()
    output = model(torch.randn(8, 3, 512, 512).cuda())
    print(output.shape)