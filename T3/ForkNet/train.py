import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
patch_weight = 40

train_img_index_path = '../ForkNet-master/list/train_image_index.list'
val_img_index_path = '../ForkNet-master/list/val_image_index.list'
Y_path = '../ForkNet-master/data/training_set/Y.h5'
labels_path = '../ForkNet-master/data/training_set/Labels.h5'
ckpt_path = '../ForkNet-master/best_model/model_1/model_1.pt'
csv_path = '../ForkNet-master/list/psnr_record_1.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(batch_size = batch_size, train_img_index_path = train_img_index_path,
              val_img_index_path = val_img_index_path, Y_path = Y_path, labels_path = labels_path):
    '''
    Divide the training set and validation set.
    Return two generator to generate batches of data.
    '''
    # read data from h5 file
    with h5py.File(Y_path, 'r') as h1:
        Y = np.array(h1.get('inputs'))

    with h5py.File(labels_path, 'r') as h2:
        label = np.array(h2.get('labels'))

    # Input = np.concatenate((Y, bic), axis=-1)
    Input = Y
    patch_num = Y.shape[0]
    patch_num_per_img = patch_num // num_img

    train_img_index_str = open(train_img_index_path).read()
    train_img_index = [int(idx) for idx in train_img_index_str.split(',')]

    val_img_index_str = open(val_img_index_path).read()
    val_img_index = [int(idx) for idx in val_img_index_str.split(',')]

    patch_index_train = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in train_img_index])
    patch_index_val = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in val_img_index])

    patch_num_train = len(patch_index_train)
    # training steps in one epoch
    train_steps = int(np.ceil(patch_num_train * 1. / batch_size))
    print('# Training Patches: {}.'.format(patch_num_train))

    patch_num_val = len(patch_index_val)
    # validation steps in one epoch
    val_steps = int(np.ceil(patch_num_val * 1. / batch_size))
    print('# Validation Patches: {}.'.format(patch_num_val))

    train_Y = Input[patch_index_train]
    train_label = label[patch_index_train]
    val_Y = Input[patch_index_val]
    val_para = label[patch_index_val]

    return train_steps, train_Y, train_label, val_steps, val_Y, val_para


def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')


def main():
    lr = 0.001
    batch_size = 64
    num_img = 110
    num_epochs = 1
    patch_height = 40
    patch_weight = 40

    train_img_index_path = './list/train_image_index.list'
    val_img_index_path = './list/val_image_index.list'
    Y_path = './data/training_set/Y.h5'
    labels_path = './data/training_set/Labels.h5'
    ckpt_path = './best_model/model_1/model_1.pt'
    csv_path = './list/psnr_record_1.csv'

    train_loader, val_loader = load_data(batch_size, train_img_index_path, val_img_index_path, Y_path, labels_path)

    model = ForkNet()  # 替换为你的模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = LOSS

    train(model, loss_fn, optimizer, train_loader, val_loader, num_epochs)
    
if __name__ == "__main__":
    main()