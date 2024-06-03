import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*self.resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*self.resnet_block(64, 128, num_blocks[1]))
        self.layer3 = nn.Sequential(*self.resnet_block(128, 256, num_blocks[2]))
        self.layer4 = nn.Sequential(*self.resnet_block(256, 512, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_out_channels = self.get_final_out_channels(block, num_blocks)
        self.fc = nn.Linear(final_out_channels if block == self.Residual else 2048, num_classes)
            
    # 残差块
    class Residual(nn.Module):
        def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
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

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(self.Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(self.Residual(num_channels, num_channels))
        return blk
    
    def get_final_out_channels(self, block, num_blocks):
        # 计算最终的输出通道数
        layers = [64, 128, 256, 512]
        for i in range(4):
            if num_blocks[i] == 0:
                return layers[i-1] if i > 0 else 64
        return layers[3] if block == self.Residual else layers[3] * 4
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# FCN模型
class FCN(nn.Module):
    def __init__(self, num_classes=10):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # GAP前卷积层，输出通道数应与类别数(10个数字)相同
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, padding=1)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.dropout(x)
        x = F.relu(self.conv2(x))
        #x = self.dropout(x)
        x = F.relu(self.conv3(x))
        #x = self.dropout(x)
        x = F.relu(self.conv4(x))
        #x = self.dropout(x)
        x = self.conv5(x)  # 最后一层卷积不使用ReLU激活
        
        # 使用全局平均池化
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # 输出形状变为(B, C, 1, 1)
        
        # 将四维张量展平为二维，以便进行损失计算
        x = x.view(x.size(0), -1)
        return x
    
# CNN模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 7*7*64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x