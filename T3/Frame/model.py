import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import h5py


class ForkNet(nn.Module):
    def __init__(self, padding='same'):
        super(ForkNet, self).__init__()
        self.padding = 'same'
        # if padding.lower() == 'valid':
        #     self.padding = 0
        # else:
        #     self.padding = 1

        self.conv1 = nn.Conv2d(1, 96, kernel_size=5, padding=self.padding)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=self.padding)
        
        self.conv3_1 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=5, padding=self.padding)
        
        self.conv3_2 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_2 = nn.Conv2d(32, 1, kernel_size=5, padding=self.padding)
        
        self.conv3_3 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_3 = nn.Conv2d(32, 1, kernel_size=5, padding=self.padding)

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


class ResNet18(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ResNet18, self).__init__(*args, **kwargs)
        pass


class MyDataset(Dataset):
    def __init__(self, file_path) -> None:
        super(MyDataset, self).__init__()
        self.file_path = file_path
        self.h5file = h5py.File(self.file_path, 'r')
        self.data = self.h5file['data']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.h5file[f'data/data_{idx}'][...])
        aop = torch.from_numpy(self.h5file[f'labels/label_{idx}/aop'][...])
        dolp = torch.from_numpy(self.h5file[f'labels/label_{idx}/dolp'][...])
        s0 = torch.from_numpy(self.h5file[f'labels/label_{idx}/s0'][...])
        return data, aop, dolp, s0


file_path = r"E:\data\data_h5\data_pim.h5"
batch_size = 10

custom_dataset = MyDataset(file_path)
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


for i, (data,aop,dolp,s0) in enumerate(data_loader):
    print("Batch", i+1)
    print("Data tensor shape:", data.shape)
    print("Labels tensor shape:", aop.shape)


'''
Loss Functions
'''
def mae_loss(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    loss = torch.mean(0.1 * torch.abs(s0_true - s0_pred) + 
                      torch.abs(dolp_true - dolp_pred) + 
                      0.05 * torch.abs(aop_true - aop_pred))
    return loss


def smooth_l1_loss(reg_true, reg_pred, sigma):
    sigma2 = sigma ** 2
    diff = reg_true - reg_pred
    abs_diff = torch.abs(diff)
    mask = abs_diff < (1. / sigma2)
    
    loss = torch.where(mask, 0.5 * sigma2 * diff ** 2, abs_diff - 0.5 / sigma2)
    return loss.mean()


def smooth_loss(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    loss = (0.1 * smooth_l1_loss(s0_true, s0_pred, 2) + 
            smooth_l1_loss(dolp_true, dolp_pred, 2) + 
            0.01 * smooth_l1_loss(aop_true, aop_pred, 2))
    return loss


def LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2):
    L, C, S, ssim_loss_value = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = torch.mean(0.1 * torch.abs(s0_true - s0_pred) + 
                      torch.abs(dolp_true - dolp_pred) + 
                      0.05 * torch.abs(aop_true - aop_pred)) - 0.02 * torch.log(C)
    return loss


def MSE_LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2):
    L, C, S, ssim_loss_value = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = torch.mean(0.1 * torch.square(s0_true - s0_pred) + 
                      torch.square(dolp_true - dolp_pred) + 
                      0.032 * torch.square(aop_true - aop_pred)) - 0.02 * torch.log(C)
    return loss


def std_variance(x):
    x_mean = torch.mean(x)
    variance = torch.sqrt(torch.sum((x - x_mean) ** 2) / (x.numel() - 1))
    return variance


def covariance(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    covariance = torch.sum((x - x_mean) * (y - y_mean)) / (x.numel() - 1)
    return covariance


def ssim_loss(x, y, mv, k1=0.01, k2=0.03):
    c1 = (k1 * mv) ** 2
    c2 = (k2 * mv) ** 2
    c3 = c2 / 2

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_std_var = std_variance(x)
    y_std_var = std_variance(y)
    xy_covar = covariance(x, y)

    L = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
    C = (2 * x_std_var * y_std_var + c2) / (x_std_var ** 2 + y_std_var ** 2 + c2)
    S = (xy_covar + c3) / (x_std_var * y_std_var + c3)
    SSIM = L * C * S

    loss = 1 - SSIM

    return L, C, S, loss