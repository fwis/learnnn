import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
from torch.utils.data import  Dataset
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import random

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
    def __init__(self, num_blocks=[1, 1, 3, 1]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 256, num_blocks[1]))
        self.layer3 = nn.Sequential(*resnet_block(256, 128, num_blocks[2]))
        self.layer4 = nn.Sequential(*resnet_block(128, 64, num_blocks[3]))
        
        # Pixel shuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        
    def Chanel_seprate(self, x):
        batch_size, channels, height, width = x.size()
        input = torch.empty((batch_size, 4, height // 2, width // 2), device=x.device)
        input[:, 0, :, :] = x[:, 0, 0::2, 0::2]  # 0
        input[:, 1, :, :] = x[:, 0, 0::2, 1::2]  # 45
        input[:, 2, :, :] = x[:, 0, 1::2, 0::2]  # 90
        input[:, 3, :, :] = x[:, 0, 1::2, 1::2]  # 135
        return input 
    
    def forward(self, x):
        x = self.Chanel_seprate(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        x = self.conv2(x)

        return x
    
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
            
        lables = torch.concat((i0,i45,i90,i135))
        
        return data, lables

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
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, labels, output):
        loss = self.mse(labels, output)
        s0_pred = torch.sum(output, dim=1)/2
        s0_true = torch.sum(labels, dim=1)/2
        s0_loss = self.mse(s0_true, s0_pred)
        Q_loss = self.mse(output[:, 0, :, :].unsqueeze(1) - output[:, 2, :, :].unsqueeze(1), labels[:, 0, :, :].unsqueeze(1) - labels[:, 2, :, :].unsqueeze(1))
        U_loss = self.mse(output[:, 1, :, :].unsqueeze(1) - output[:, 3, :, :].unsqueeze(1), labels[:, 1, :, :].unsqueeze(1) - labels[:, 3, :, :].unsqueeze(1))
        # dolp_pred = torch.mean(torch.sqrt(torch.square(output[:, 0, :, :] - output[:, 2, :, :]) + torch.square(output[:, 1, :, :] - output[:, 3, :, :])) / (torch.mean(s0_pred) + 1e-8))
        # dolp_true = torch.mean(torch.sqrt(torch.square(labels[:, 0, :, :] - labels[:, 2, :, :]) + torch.square(labels[:, 1, :, :] - labels[:, 3, :, :])) / (torch.mean(s0_true) + 1e-8))
        # dolp_loss = self.mse(dolp_pred,dolp_true)
        total_loss = loss + 0.2 * s0_loss + 0.5 * (Q_loss + U_loss) 
        
        return total_loss

# class PerceptualLoss(nn.Module):
#     def __init__(self, feature_layer=19):
#         super(PerceptualLoss, self).__init__()
#         vgg = models.vgg19(pretrained=True).features
#         self.features = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
#         for param in self.features.parameters():
#             param.requires_grad = False
#         self.criterion = nn.MSELoss()
#         # self.normalize = Normalize(mean=[0.485], std=[0.229])

#     def forward(self, input, target):
#         # input = self.normalize(input)
#         input = input.repeat(1, 3, 1, 1)
#         # target = self.normalize(target)
#         target = target.repeat(1, 3, 1, 1)
#         input_features = self.features(input)
#         target_features = self.features(target)
#         loss = self.criterion(input_features, target_features)
#         return loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=None, use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        if feature_layers is None:
            self.feature_layers = [5, 10, 19]
        else:
            self.feature_layers = feature_layers
        
        self.vgg = nn.Sequential(*[self.vgg[i] for i in range(max(self.feature_layers) + 1)])
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.MSELoss()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.vgg.to(self.device)
    
    def forward(self, inputs, targets):
        total_loss = 0
        for input, target in zip(inputs, targets):
            input = input.repeat(1, 3, 1, 1)
            # print(input.shape)
            target = target.repeat(1, 3, 1, 1)
            # print(target.shape)
            input = self.normalize(input)
            target = self.normalize(target)
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            input_features = self.extract_features(input)
            target_features = self.extract_features(target)
            
            loss = 0
            for inp_feat, tgt_feat in zip(input_features, target_features):
                loss += self.criterion(inp_feat, tgt_feat)
            
            total_loss += loss
        
        return total_loss
    
    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features