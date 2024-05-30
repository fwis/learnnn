import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import h5py
from model import ForkNet, MSE_LOSS, LOSS, smooth_loss
from utils.utils import dolp, psnr, normalize, aop, gs_rand_choice
import matplotlib.pyplot as plt
import os
import math
import csv

lr = 0.001
batch_size = 64
num_img = 110
num_epochs = 1
patch_height = 40
patch_width = 40
learning_rate_decay_steps = 600
learning_rate_decay_rate = 0.988
dsp_itv = 10
save_best = True
early_stop = False
metrics = 'training loss'
patient = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomHDF5Dataset(Dataset):
    def __init__(self, labels_filename, inputs_filename, transform=None):
        self.labels_filename = labels_filename
        self.inputs_filename = inputs_filename
        self.transform = transform

        # 打开 HDF5 文件并读取数据集
        with h5py.File(self.labels_filename, 'r') as f:
            self.labels = f['labels'][:]
            self.inputs = f['inputs'][:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_data = self.inputs[idx]

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label


def data_loader():
    train_loader = 
    test_loader = 
    
    return train_loader, test_loader
    pass


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 从数据集获取数据以及对应标签
        data, target = data.to(device), target.to(device)
        # 清空所有优化过的梯度
        optimizer.zero_grad()
        # 执行模型forward
        output = model(data)
        # 计算损失
        loss = torch.nn.functional.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 优化模型参数
        optimizer.step()

            
            
def main():
    pass

if __name__ == "__main__":
    main()