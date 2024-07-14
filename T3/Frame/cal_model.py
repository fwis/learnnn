import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
import torch.optim as optim
from torch.utils.data import  Dataset
import math
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import random
import numpy as np

'''
ResNet
'''
# Residual block
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

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=1))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet(nn.Module):
    def __init__(self, num_blocks=[1, 1,  2]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, num_blocks[1]))
        
        self.layer4_1 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        self.layer4_2 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        self.layer4_3 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        self.layer4_4 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        
        # Output layer
        # Depthwise separabel convolution
        self.conv1_1 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_4 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv2_1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_2 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_3 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_4 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)

        i0 = self.layer4_1(x)
        i0 = self.conv1_1(i0)
        i0 = self.conv2_1(i0)
        
        i45 = self.layer4_2(x)
        i45 = self.conv1_2(i45)
        i45 = self.conv2_2(i45)
        
        i90 = self.layer4_2(x)
        i90 = self.conv1_2(i90)
        i90 = self.conv2_2(i90)
        
        i135 = self.layer4_2(x)
        i135 = self.conv1_2(i135)
        i135 = self.conv2_2(i135)
        
        return i0, i45, i90, i135

'''
Dataset
'''
class MyDataset(Dataset):
    def __init__(self, file_path, transform=None):
        super(MyDataset, self).__init__()
        self.file_path = file_path
        self.transform = transform
        with h5py.File(self.file_path, 'r') as h5file:
            self.data_len = len(h5file['data'])
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5file:
            data = torch.from_numpy(h5file[f'data/data_{idx}'][...]).unsqueeze(0)
            i0 = torch.from_numpy(h5file[f'labels/label_{idx}/i0'][...]).unsqueeze(0)
            i45 = torch.from_numpy(h5file[f'labels/label_{idx}/i45'][...]).unsqueeze(0)
            i90 = torch.from_numpy(h5file[f'labels/label_{idx}/i90'][...]).unsqueeze(0)
            i135 = torch.from_numpy(h5file[f'labels/label_{idx}/i135'][...]).unsqueeze(0)
        if self.transform:
            data, i0, i45, i90, i135 = self.transform(data, i0, i45, i90, i135)
        
        return data, i0, i45, i90, i135

# Data augmentation transform
def custom_transform(data, i0, i45, i90, i135):
    # Random horizontal flip
    if random.random() > 0.5:
        data = TF.hflip(data)
        i0 = TF.hflip(i0)
        i45 = TF.hflip(i45)
        i90 = TF.hflip(i90)
        i135 = TF.hflip(i135)

    # Random vertical flip
    if random.random() > 0.5:
        data = TF.vflip(data)
        i0 = TF.vflip(i0)
        i45 = TF.vflip(i45)
        i90 = TF.vflip(i90)
        i135 = TF.vflip(i135)

    # Random 0, 90, 180, 270 degree rotation
    angle = random.choice([0, 90, 180, 270])
    data = TF.rotate(data, angle)
    i0 = TF.rotate(i0, angle)
    i45 = TF.rotate(i45, angle)
    i90 = TF.rotate(i90, angle)
    i135 = TF.rotate(i135, angle)

    return data, i0, i45, i90, i135

'''
Loss Functions
'''
class CustomLoss(nn.Module):
    def __init__(self, weight=1):
        super(CustomLoss, self).__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
        
    def forward(self, i0_pred, i0_true, i45_pred, i45_true, i90_pred, i90_true, i135_pred, i135_true):
        # Physics informed loss
        # s0_pred = (i0_pred + i45_pred + i90_pred + i135_pred) / 2
        # s0_true = (i0_true + i45_true + i90_true + i135_true) / 2
        # print('s0',torch.mean(abs(s0_true - s0_pred)))
        # dolp_pred = torch.sqrt(torch.square(torch.abs(i0_pred - i90_pred)) + torch.square(torch.abs(i45_pred - i135_pred))) / (s0_pred + 1e-8)
        # dolp_true = torch.sqrt(torch.square(torch.abs(i0_true - i90_true)) + torch.square(torch.abs(i45_true - i135_true))) / (s0_true + 1e-8)
        # print('dolp',torch.mean(abs(dolp_true - dolp_pred)))
        # aop_pred = 0.5 * torch.arctan(torch.abs(i45_pred - i135_pred) / (torch.abs(i0_pred - i90_pred) + 1e-8)) + math.pi/4.
        # aop_true = 0.5 * torch.arctan((i45_true - i135_true) / (i0_true - i90_true + 1e-8)) + math.pi/4.
        # print('aop',torch.mean(abs(aop_true - aop_pred)))
    
        # Total loss
        total_loss  = torch.mean(self.mse(i0_pred, i0_true) + self.mse(i45_pred,i45_true) + 
                                self.mse(i90_pred,i90_true) + self.mse(i135_pred,i135_true))
                    #   0.6 * abs(dolp_true - dolp_pred) + 
                    #   0.3 * abs(aop_true - aop_pred)) 
        # + physics_loss

        return total_loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=19):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
        # self.normalize = Normalize(mean=[0.485], std=[0.229])

    def forward(self, input, target):
        # input = self.normalize(input)
        input = input.repeat(1, 3, 1, 1)
        # target = self.normalize(target)
        target = target.repeat(1, 3, 1, 1)
        input_features = self.features(input)
        target_features = self.features(target)
        loss = self.criterion(input_features, target_features)
        return loss
    