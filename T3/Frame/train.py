import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import h5py
from model import ForkNet, MSE_LOSS, LOSS, smooth_loss
from utils.batch_generator import patch_batch_generator
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

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches =  len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            timer.stop()
            
            
def main():
    pass

if __name__ == "__main__":
    main()