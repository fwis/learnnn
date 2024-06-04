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
    
    
# file_path = r"D:\WORKS\Polarization\data\data_h5\data_ol.h5"
# batch_size = 10

# custom_dataset = MyDataset(file_path)
# data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


# for i, (data,aop,dolp,s0) in enumerate(data_loader):
#     print("Batch", i+1)
#     print("Data tensor shape:", data.shape)
#     print("Labels tensor shape:", aop.shape)


'''
Loss Functions
'''

