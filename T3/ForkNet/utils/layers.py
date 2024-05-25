import torch
import torch.nn as nn
import torch.nn.functional as F

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