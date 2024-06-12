import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import random
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

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
        # aop = torch.atan(aop) / 2. + math.pi / 4
        
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
    def __init__(self, num_blocks=[2, 1, 1, 2]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh =nn.Tanh()
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(128, 128, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(128, 64, num_blocks[1]))
        self.layer3 = nn.Sequential(*resnet_block(64, 32, num_blocks[2]))
        
        self.layer4_1 = nn.Sequential(*resnet_block(32, 24, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        self.layer4_2 = nn.Sequential(*resnet_block(32, 24, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        self.layer4_3 = nn.Sequential(*resnet_block(32, 24, num_blocks[3])) if num_blocks[3] > 0 else nn.Identity()
        
        # Depthwise Separable Convolution
        self.conv1_1 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1,groups=24)
        self.conv1_2 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1,groups=24)
        self.conv1_3 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1,groups=24)
        
        self.conv2_1 = nn.Conv2d(24,1,kernel_size=1,stride=1)
        self.conv2_2 = nn.Conv2d(24,1,kernel_size=1,stride=1)
        self.conv2_3 = nn.Conv2d(24,1,kernel_size=1,stride=1)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)

        aop = self.layer4_1(x)
        aop = self.conv1_1(aop)
        aop = self.relu(aop)
        aop = self.conv2_1(aop)
        
        dolp = self.layer4_2(x)
        dolp = self.conv1_2(dolp)
        dolp = self.relu(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.layer4_3(x)
        s0 = self.conv1_3(s0)
        s0 = self.relu(s0)
        s0 = self.conv2_3(s0)
        
        return aop, dolp, s0

class ResNetFPN(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2]):
        super(ResNetFPN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(128, 128, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(128, 64, num_blocks[1]))
        self.layer3 = nn.Sequential(*resnet_block(64, 32, num_blocks[2]))
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        
        # Top-down layers
        self.toplayer = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        # Depthwise Separable Convolution
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        
        self.conv2_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv2_2 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv2_3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.2)
        
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c4 = self.dropout(c4)
        
        # Top-down pathway
        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        
        # Head: AOP, DOLP, S0
        aop = self.conv1_1(p1)
        aop = self.relu(aop)
        aop = self.conv2_1(aop)
        # aop = torch.atan(aop) / 2. + math.pi / 4
        
        dolp = self.conv1_2(p1)
        dolp = self.relu(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.conv1_3(p1)
        s0 = self.relu(s0)
        s0 = self.conv2_3(s0)
        
        return aop, dolp, s0

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
            aop = torch.from_numpy(h5file[f'labels/label_{idx}/aop'][...]).unsqueeze(0)
            dolp = torch.from_numpy(h5file[f'labels/label_{idx}/dolp'][...]).unsqueeze(0)
            s0 = torch.from_numpy(h5file[f'labels/label_{idx}/s0'][...]).unsqueeze(0)

        if self.transform:
            data, aop, dolp, s0 = self.transform(data, aop, dolp, s0)
        
        return data, aop, dolp, s0

# Data augmentation transform
def custom_transform(data, aop, dolp, s0):
    # Random horizontal flip
    if random.random() > 0.5:
        data = TF.hflip(data)
        aop = TF.hflip(aop)
        dolp = TF.hflip(dolp)
        s0 = TF.hflip(s0)

    # Random vertical flip
    if random.random() > 0.5:
        data = TF.vflip(data)
        aop = TF.vflip(aop)
        dolp = TF.vflip(dolp)
        s0 = TF.vflip(s0)

    # Random 0, 90, 180, 270 degree rotation
    angle = random.choice([0, 90, 180, 270, 0])
    data = TF.rotate(data, angle)
    aop = TF.rotate(aop, angle)
    dolp = TF.rotate(dolp, angle)
    s0 = TF.rotate(s0, angle)

    return data, aop, dolp, s0


'''
Loss Functions
'''
class CustomLoss(nn.Module):
    def __init__(self, weight=1):
        super(CustomLoss, self).__init__()
        self.weight = weight
        
    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Physics informed loss        
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
        Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
        U_true = dolp_true * s0_true * torch.sin(2 * aop_true)        
        loss_Q = torch.mean(Q_pred - Q_true)**2
        loss_U = torch.mean(U_pred - U_true)**2
        loss_dolp = torch.mean((dolp_pred - dolp_true)**2)
        loss_s0 = torch.mean((s0_pred - s0_true)**2)
        physics_loss = torch.abs(loss_s0 * loss_dolp - (loss_Q + loss_U))

        # Total loss
        total_loss  = torch.mean(0.1 * abs(s0_true - s0_pred) + 
                      0.5 * abs(dolp_true - dolp_pred) + 
                      0.03 * abs(aop_true - aop_pred)) +  0.8 * physics_loss - 0.002 * SSIM(aop_pred,aop_true, data_range= math.pi/2)

        return total_loss
    
    
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Initialize log-variance parameters for each loss component
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Data-based loss
        mse_loss_aop = nn.MSELoss()(aop_pred, aop_true)
        mse_loss_dolp = nn.MSELoss()(dolp_pred, dolp_true)
        mse_loss_s0 = nn.MSELoss()(s0_pred, s0_true)

        # Physics-informed loss
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
        Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
        U_true = dolp_true * s0_true * torch.sin(2 * aop_true)

        ssim_loss_Q = 1 - SSIM(Q_pred, Q_true)
        ssim_loss_U = 1 - SSIM(U_pred, U_true)
        ssim_loss_Q = torch.relu(ssim_loss_Q)
        ssim_loss_U = torch.relu(ssim_loss_U)
        # loss_Q = nn.MSELoss()(Q_pred, Q_true)
        # loss_U = nn.MSELoss()(U_pred, U_true)
        
        # Calculate the total loss using uncertainty weighting
        loss_aop = mse_loss_aop * torch.exp(-self.log_vars[0]) + self.log_vars[0]
        loss_dolp = mse_loss_dolp * torch.exp(-self.log_vars[1]) + self.log_vars[1]
        loss_s0 = mse_loss_s0 * torch.exp(-self.log_vars[2]) + self.log_vars[2]
        phys_loss_Q = ssim_loss_Q * torch.exp(-self.log_vars[3]) + self.log_vars[3]
        phys_loss_U = ssim_loss_U * torch.exp(-self.log_vars[4]) + self.log_vars[4]
        
        loss_aop = torch.relu(loss_aop)
        loss_dolp = torch.relu(loss_dolp)
        loss_s0 = torch.relu(loss_s0)
        phys_loss_Q = torch.relu(phys_loss_Q)
        phys_loss_U = torch.relu(phys_loss_U)
        regularization_loss = torch.sum(torch.abs(self.log_vars))
        
        total_loss = loss_aop + loss_dolp + loss_s0 + phys_loss_Q + phys_loss_U + 0.01 * regularization_loss

        return total_loss
    
    
class GradNormLoss(nn.Module):
    def __init__(self, model, alpha=0.5):
        super(GradNormLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        mse_loss_aop = nn.MSELoss()(aop_pred, aop_true)
        mse_loss_dolp = nn.MSELoss()(dolp_pred, dolp_true)
        mse_loss_s0 = nn.MSELoss()(s0_pred, s0_true)

        precision_aop = torch.exp(-self.log_vars[0])
        precision_dolp = torch.exp(-self.log_vars[1])
        precision_s0 = torch.exp(-self.log_vars[2])

        loss_aop = mse_loss_aop * precision_aop + self.log_vars[0]
        loss_dolp = mse_loss_dolp * precision_dolp + self.log_vars[1]
        loss_s0 = mse_loss_s0 * precision_s0 + self.log_vars[2]

        total_loss = loss_aop + loss_dolp + loss_s0

        # GradNorm
        grads = torch.autograd.grad(total_loss, self.model.parameters(), create_graph=True)
        grads_norm = torch.stack([grad.norm() for grad in grads]).sum()

        grad_norm_loss = (self.alpha / grads_norm) * total_loss

        return grad_norm_loss