import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM

class ForkNet(nn.Module):
    def __init__(self, padding='same'):
        super(ForkNet, self).__init__()
        self.padding = 'same'
        # if padding.lower() == 'valid':
        #     self.padding = 0
        # else:
        #     self.padding = 1

        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=self.padding)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        
        self.conv3_1 = nn.Conv2d(32, 24, kernel_size=3, padding=self.padding)
        self.conv4_1 = nn.Conv2d(24, 1, kernel_size=3, padding=self.padding)
        
        self.conv3_2 = nn.Conv2d(32, 24, kernel_size=3, padding=self.padding)
        self.conv4_2 = nn.Conv2d(24, 1, kernel_size=3, padding=self.padding)
        
        self.conv3_3 = nn.Conv2d(32, 24, kernel_size=3, padding=self.padding)
        self.conv4_3 = nn.Conv2d(24, 1, kernel_size=3, padding=self.padding)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        
        x3_1 = F.relu(self.conv3_1(x2))
        s0 = self.conv4_1(x3_1)
        
        x3_2 = F.relu(self.conv3_2(x2))
        dolp = self.conv4_2(x3_2)
        
        x3_3 = F.relu(self.conv3_3(x2))
        aop = self.conv4_3(x3_3)
        aop = torch.atan(aop) / 2. + math.pi / 4
        
        return s0, dolp, aop
    
    
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
    def __init__(self, num_blocks=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, num_blocks[1]))
        self.layer3 = nn.Sequential(*resnet_block(128, 64, num_blocks[2]))
        
        self.layer4_1 = nn.Sequential(*resnet_block(64, 32, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        self.layer4_2 = nn.Sequential(*resnet_block(64, 32, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        self.layer4_3 = nn.Sequential(*resnet_block(64, 32, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        
        self.conv_1 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,)
        self.conv_2 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,)
        self.conv_3 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        aop = self.layer4_1(x)
        aop = self.conv_1(aop)
        aop = torch.atan(aop) / 2. + math.pi / 4
        
        dolp = self.layer4_2(x)
        dolp = self.conv_2(dolp)
        
        s0 = self.layer4_3(x)
        s0 = self.conv_3(s0)
        
        return aop, dolp, s0


class MyDataset(Dataset):
    def __init__(self, file_path) -> None:
        super(MyDataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as h5file:
            self.data_len = len(h5file['data'])
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5file:
            data = torch.from_numpy(h5file[f'data/data_{idx}'][...]).unsqueeze(0)
            aop = torch.from_numpy(h5file[f'labels/label_{idx}/aop'][...]).unsqueeze(0)
            dolp = torch.from_numpy(h5file[f'labels/label_{idx}/dolp'][...]).unsqueeze(0)
            s0 = torch.from_numpy(h5file[f'labels/label_{idx}/s0'][...]).unsqueeze(0)
        return data, aop, dolp, s0


'''
Loss Functions
'''
class CustomLoss(nn.Module):
    def __init__(self, weight=1):
        super(CustomLoss, self).__init__()
        self.weight = weight
        
    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Data-based loss
        mse_loss_aop = nn.MSELoss()(aop_pred, aop_true)
        mse_loss_dolp = nn.MSELoss()(dolp_pred, dolp_true)
        mse_loss_s0 = nn.MSELoss()(s0_pred, s0_true)
        
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
        Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
        U_true = dolp_true * s0_true * torch.sin(2 * aop_true)

       # Physics informed loss
        mse_loss_Q = nn.MSELoss()(Q_pred, Q_true)
        mse_loss_U = nn.MSELoss()(U_pred, U_true)

        # Total loss
        total_loss  = torch.mean(0.5 * mse_loss_aop + 
                       mse_loss_dolp + 
                      0.5 * mse_loss_s0) + 0.1*(mse_loss_Q + mse_loss_U)
        
        return total_loss
    