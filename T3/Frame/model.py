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
    def __init__(self, file_path, batch_size) -> None:
        super(MyDataset, self).__init__()
        self.filepath = file_path
        self.batch_size = batch_size
        self.file = h5py.File(file_path, 'r')
        self.dataset_shape = self.file['data'].shape
        
    def __len__(self):
        return self.dataset_shape[0] // self.batch_size
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.dataset_shape[0])
 
        data_tensor = torch.tensor(self.file['data'][start:end])
        print('datatensor:',data_tensor.shape)
        labels_tensor = torch.tensor(self.file['labels'][start:end])
        
        return data_tensor, labels_tensor



def read_h5(file_path, batch_size):
    with h5py.File(file_path, 'r') as f:
        dataset_shape = f['data'].shape
        for start in range(0, dataset_shape[0], batch_size):
            end = min(start + batch_size, dataset_shape[0])  

            data_chunk = f['data'][start:end]
            labels_chunk = f['labels'][start:end]
            data_tensor = torch.tensor(data_chunk)
            labels_tensor = torch.tensor(labels_chunk)

            yield data_tensor, labels_tensor

file_path = r"D:\VScodeProjects\dataset\OL_DATA\labels.h5" 
batch_size = 10

custom_dataset = MyDataset(file_path, batch_size)
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=None, shuffle=True)


for i, (data, labels) in enumerate(data_loader):
    print("Batch", i+1)
    print("Data tensor shape:", data.shape)
    print("Labels tensor shape:", labels.shape)


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
