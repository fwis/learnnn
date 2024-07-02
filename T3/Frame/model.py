import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import random
import functools

'''
CNN
'''
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
        
        x3_3 = F.tanh(self.conv3_3(x2))
        aop = self.conv4_3(x3_3)
        aop = torch.atan(aop) / 2. + math.pi / 4
        
        return s0, dolp, aop
    
class ForkLoss(nn.Module):
    def __init__(self, weight=1):
        super(ForkLoss, self).__init__()
        self.weight = weight
        
    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Total loss
        total_loss  = torch.mean(0.1 * abs(s0_true - s0_pred) + 
                      abs(dolp_true - dolp_pred) + 
                      0.05 * abs(aop_true - aop_pred))  - 0.02 * SSIM(aop_pred,aop_true, data_range= math.pi/2)

        return total_loss
    
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
    def __init__(self, num_blocks=[1, 1, 1, 2]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, num_blocks[1]))
        # self.layer3 = nn.Sequential(*resnet_block(64, 32, num_blocks[2]))
        
        self.layer4_1 = nn.Sequential(*resnet_block(128, 32, num_blocks[3]))
        self.layer4_2 = nn.Sequential(*resnet_block(128, 32, num_blocks[3]))
        self.layer4_3 = nn.Sequential(*resnet_block(128, 32, num_blocks[3]))
        
        # Output layer
        # Depthwise separabel convolution
        self.conv1_1 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv2_1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_2 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_3 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)

        aop = self.layer4_1(x)
        aop = self.conv1_1(aop)
        aop = self.conv2_1(aop)
        
        dolp = self.layer4_2(x)
        dolp = self.conv1_2(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.layer4_3(x)
        s0 = self.conv1_3(s0)
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
ConvNeXt
'''
class NeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
def convnext_block(dim, dim_out, num_blocks):
    blk = []
    for i in range(num_blocks):
        blk.append(NeXtBlock(dim=dim))
    if dim != dim_out:
        blk.append(nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=1, stride=1),
            LayerNorm(dim_out, eps=1e-6, data_format="channels_first")))
    return blk

class ConvNeXtNet(nn.Module):
    def __init__(self, num_blocks=[1, 1, 3, 1], dim=[64, 128, 64, 32]):
        super(ConvNeXtNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        
        # ConvNeXt blocks
        self.layer1 = nn.Sequential(*convnext_block(64, dim[0], num_blocks[0])) 
        self.layer2 = nn.Sequential(*convnext_block(dim[0],dim[1], num_blocks[1]))
        
        self.layer3_1 = nn.Sequential(*convnext_block(dim[1],dim[2], num_blocks[2]))
        self.layer3_2 = nn.Sequential(*convnext_block(dim[1],dim[2], num_blocks[2]))
        self.layer3_3 = nn.Sequential(*convnext_block(dim[1],dim[2], num_blocks[2]))
        
        self.layer4_1 = nn.Sequential(*convnext_block(dim[2],dim[3], num_blocks[3]))
        self.layer4_2 = nn.Sequential(*convnext_block(dim[2],dim[3], num_blocks[3]))
        self.layer4_3 = nn.Sequential(*convnext_block(dim[2],dim[3], num_blocks[3]))
        
        # Output layer
        # Depthwise separabel convolution
        self.conv1_1 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv1_3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)
        self.conv2_1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_2 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv2_3 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.layer1(x)
        x = self.layer2(x)

        aop = self.layer3_1(x)
        aop = self.layer4_1(aop)
        aop = self.conv1_1(aop)
        aop = self.conv2_1(aop)
        
        dolp = self.layer3_2(x)
        dolp = self.layer4_2(dolp)
        dolp = self.conv1_2(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.layer3_3(x)
        s0 = self.layer4_3(s0)
        s0 = self.conv1_3(s0)
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
    angle = random.choice([0, 90, 180, 270])
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
        loss_Q = torch.mean(abs(Q_pred - Q_true))
        loss_U = torch.mean(abs(U_pred - U_true))
        physics_loss = loss_Q + loss_U
    
        # Total loss
        total_loss  = torch.mean(0.1 * abs(s0_true - s0_pred) + 
                      0.6 * abs(dolp_true - dolp_pred) + 
                      0.3 * abs(aop_true - aop_pred))  + physics_loss
        # - 0.03 * SSIM(aop_pred,aop_true, data_range= math.pi/2)

        return total_loss
    
'''
GAN
'''
# Generator
class ResNetGenerator(nn.Module):
    def __init__(self, num_blocks=[1, 1, 2]):
        super(ResNetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, num_blocks[1]))
        
        self.layer3_1 = nn.Sequential(*resnet_block(128, 48, num_blocks[2]))
        self.layer3_2 = nn.Sequential(*resnet_block(128, 48, num_blocks[2]))
        self.layer3_3 = nn.Sequential(*resnet_block(128, 48, num_blocks[2]))
        
        # Output layer
        # Depthwise separable convolution
        self.conv1_1 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48)
        self.conv1_2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48)
        self.conv1_3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48)
        self.conv2_1 = nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)

        aop = self.layer3_1(x)
        aop = self.conv1_1(aop)
        aop = self.conv2_1(aop)
        
        dolp = self.layer3_2(x)
        dolp = self.conv1_2(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.layer3_3(x)
        s0 = self.conv1_3(s0)
        s0 = self.conv2_3(s0)
        
        return aop, dolp, s0

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Shared part
        self.shared = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Branches for aop, dolp, s0
        self.branch_aop = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2))
        self.branch_dolp = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2))
        self.branch_s0 = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2))

    def forward(self, aop, dolp, s0):
        out_aop = self.shared(aop)
        out_aop = self.branch_aop(out_aop)

        out_dolp = self.shared(dolp)
        out_dolp = self.branch_dolp(out_dolp)

        out_s0 = self.shared(s0)
        out_s0 = self.branch_s0(out_s0)
        
        return out_aop, out_dolp, out_s0


# Loss
class CustomGANLoss(nn.Module):
    def __init__(self):
        super(CustomGANLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Physics informed loss        
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
        Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
        U_true = dolp_true * s0_true * torch.sin(2 * aop_true)        
        loss_Q = torch.mean(abs(Q_pred - Q_true))
        loss_U = torch.mean(abs(U_pred - U_true))
        physics_loss = loss_Q + loss_U
        # L1 loss
        loss_s0 = self.l1_loss(s0_pred, s0_true)
        loss_dolp = self.l1_loss(dolp_pred, dolp_true)
        loss_aop = self.l1_loss(aop_pred, aop_true)
        # Total loss
        total_loss  = torch.mean(0.1 * loss_s0 + 
                      0.6 * loss_dolp + 
                      0.3 * loss_aop) - 0.02 * SSIM(aop_pred,aop_true, data_range= math.pi/2) + physics_loss
        
        return total_loss


# ESRGAN
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
