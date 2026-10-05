'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.layers import ConvBlock, PassportBlock
# from models.layers.passportconv2d_nonsignloss_top import PassportBlockTop

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet_Top(nn.Module):
    def __init__(self, block, num_blocks, model_config):
        super(ResNet_Top, self).__init__()
        self.cfg = model_config
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.linear = nn.Linear(1024, self.cfg.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_a, x_b=None):
        if x_b is not None:
            x = torch.cat([x_a, x_b], dim=1)
        elif isinstance(x_a, list):
            x = sum(x_a)
        else:
            x = x_a
        #x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        out = x
        out = self.linear(out)
        return out


class ResNet_Bottom(nn.Module):
    def __init__(self, block, num_blocks, model_config, num_classes=10):
        super(ResNet_Bottom, self).__init__()
        self.cfg = model_config
        self.num_blocks = num_blocks
        self.in_planes = 64
        if self.cfg.data == 'mnist':
            channel = 1
        else:
            channel = 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cfg=self.cfg)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cfg=self.cfg)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cfg=self.cfg)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cfg=self.cfg)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
            # self.set_passport()

    def _make_layer(self, block, planes, num_blocks, stride, cfg):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, cfg))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return out





def ResNet18_Bottom(model_config):
    return ResNet_Bottom(BasicBlock, [2, 2, 2, 2], model_config)


def ResNet18_Top(model_config):
    return ResNet_Top(BasicBlock, [2, 2, 2, 2], model_config)