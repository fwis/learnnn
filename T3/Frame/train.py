import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import csv
from model import MyDataset, ForkNet


lr = 0.001
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}')

file_path = r"D:\WORKS\data.h5"
batch_size = 64


def train(amodel, dataloader, adevice, num_epochs):
    model = amodel.to(adevice)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data,aop,dolp,s0) in enumerate(dataloader, 0):

            inputs = data.to(adevice)
            aop_labels = aop.to(adevice)
            dolp_labels = dolp.to(adevice)
            s0_labels = s0.to(adevice)
            # print(inputs.shape)
            optimizer.zero_grad()

            aop_preds, dolp_preds, s0_preds = model(inputs)

            loss_aop = criterion(aop_preds, aop_labels)
            loss_dolp = criterion(dolp_preds, dolp_labels)
            loss_s0 = criterion(s0_preds, s0_labels)
            loss = loss_aop + loss_dolp + loss_s0

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # 每 10 个 mini-batches 打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                print(f'inputs device: {inputs.device}')
                print(f'aop_preds device: {aop_preds.device}')
                print(f'loss device: {loss.device}')
                print(f'model device: {next(model.parameters()).device}')
        torch.cuda.synchronize()       
    print('Finished Training!')


if __name__ == "__main__":
    model = ForkNet()
    custom_dataset = MyDataset(file_path)

    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    train(model, data_loader, adevice=device, num_epochs=num_epochs)