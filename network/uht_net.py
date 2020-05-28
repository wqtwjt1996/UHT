import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from network.resnet import ResNet50

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class upsample(nn.Module):
    def __init__(self, input_1, input_2, output):
        super(upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_1 + input_2, input_2, kernel_size=1),
            nn.BatchNorm2d(input_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_2, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UHT_Net(nn.Module):
    def __init__(self):
        super(UHT_Net, self).__init__()

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
        FM = self.basenet(x)

        y1 = F.interpolate(FM[4], size=FM[3].size()[2:], mode='bilinear')
        y1 = torch.cat([y1, FM[3]], dim=1)
        y1 = self.upconv1(y1)

        y2 = F.interpolate(y1, size=FM[2].size()[2:], mode='bilinear')
        y2 = torch.cat([y2, FM[2]], dim=1)
        y2 = self.upconv2(y2)

        y3 = F.interpolate(y2, size=FM[1].size()[2:], mode='bilinear')
        y3 = torch.cat([y3, FM[1]], dim=1)
        y3 = self.upconv3(y3)

        y4 = F.interpolate(y3, size=FM[0].size()[2:], mode='bilinear')
        y4 = torch.cat([y4, FM[0]], dim=1)
        y_final = self.upconv4(y4)

        res = self.conv_cls(self.biggen(y_final))

        return res

if __name__ == '__main__':
    model = UHT_Net().cuda()
    output = model(torch.randn(1, 3, 512, 512).cuda())
    print(output.shape)