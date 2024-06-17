import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution layer
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True, activation=F.relu):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,  activation=F.relu, is_train=True):
        super(Conv2D_BN, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# Residual block
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        #两个3*3的卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            #1*1卷积层
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True, bottle=True):
        super(Bottleneck, self).__init__()
        self.same_shape = same_shape
        self.bottle = bottle
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel*4)
        )
        if not same_shape or not bottle:
            self.conv4 = nn.Conv2d(in_channel, out_channel*4, kernel_size=1, stride=strides, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channel*4)
            print(self.conv4)
    def forward(self, x):
        print(x.size())
        out = self.block(x)
        print(out.size())
        if not self.same_shape or not self.bottle:
            x = self.bn4(self.conv4(x))
        return F.relu(out + x)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)

