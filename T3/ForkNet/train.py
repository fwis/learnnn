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
dsp_itv = 6
save_best = True
early_stop = False
metrics = 'training loss'
patient = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_img_index_path = '../learnnn/T3/ForkNet/list/train_image_index.list'
val_img_index_path = '../learnnn/T3/ForkNet/list/val_image_index.list'
Y_path = '../learnnn/T3/ForkNet/data/training_set/Y.h5'
labels_path = '../learnnn/T3/ForkNet/data/training_set/Labels.h5'
ckpt_path = '../learnnn/T3/ForkNet/best_model/model_1/model_1.pt'
csv_path = '../learnnn/T3/ForkNet/list/psnr_record_1.csv'

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
    input = Y
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

    train_Y = input[patch_index_train]
    train_label = label[patch_index_train]
    val_Y = input[patch_index_val]
    val_para = label[patch_index_val]
    # 将维度转换为 PyTorch 默认顺序
    train_Y = np.transpose(train_Y, (0, 3, 1, 2)) 
    train_label = np.transpose(train_label, (0, 3, 1, 2))  
    val_Y = np.transpose(val_Y, (0, 3, 1, 2))
    val_para = np.transpose(val_para, (0, 3, 1, 2))
    print(train_Y.shape)
    return train_steps, train_Y, train_label, val_steps, val_Y, val_para

       
        
# TODO
def train(model, patch_width, patch_height, num_epochs, batch_size, lr, learning_rate_decay_steps, learning_rate_decay_rate, dsp_itv, ckpt_path, save_best, early_stop):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=learning_rate_decay_rate)

    # train_Y, train_label, val_Y, val_label = load_data(batch_size, 'train_img_index_path', 'val_img_index_path', 'Y_path', 'labels_path', 110)
    train_steps, train_Y, train_label, val_steps, val_Y, val_para = load_data(batch_size = batch_size, train_img_index_path = train_img_index_path,
              val_img_index_path = val_img_index_path, Y_path = Y_path, labels_path = labels_path)
    
    best_val_loss = float('inf')
    wait = 0
    psnr_record = []

    for epoch in range(num_epochs):
        model.train()
        train_generator = patch_batch_generator(train_Y, train_label, batch_size, patch_width, patch_height, random_shuffle=True)
        val_generator = patch_batch_generator(val_Y, val_para, batch_size, patch_width, patch_height, random_shuffle=False)
        print('=======================================Epoch:{}/{}======================================='.format(epoch, num_epochs))
        # train
        total_train_loss = 0
        for train_steps, (Input_batch_train, Para_batch_train) in enumerate(train_generator):
            # print(f"Para_batch_train shape: {Para_batch_train.shape}")
            Y_batch_train = torch.tensor(Input_batch_train[:, 0:, :, :], dtype=torch.float32).to(device)
            S0_batch_train = torch.tensor(Para_batch_train[:, 0:1, :, :], dtype=torch.float32).to(device)
            DoLP_batch_train = torch.tensor(Para_batch_train[:, 1:2, :,:], dtype=torch.float32).to(device)
            AoP_batch_train = torch.tensor(Para_batch_train[:, 2:3, :, :], dtype=torch.float32).to(device)

            # print(f"Y_batch_train shape: {Y_batch_train.shape}")
            # print(f"S0_batch_train shape: {S0_batch_train.shape}")
            # print(f"DoLP_batch_train shape: {DoLP_batch_train.shape}")
            # print(f"AoP_batch_train shape: {AoP_batch_train.shape}")
            
            optimizer.zero_grad()
            S0_hat, DoLP_hat, AoP_hat = model(Y_batch_train)

            # print(f"S0_hat shape: {S0_hat.shape}")
            # print(f"DoLP_hat shape: {DoLP_hat.shape}")
            # print(f"AoP_hat shape: {AoP_hat.shape}")      
                  
            loss = LOSS(S0_hat, S0_batch_train, DoLP_hat, DoLP_batch_train, AoP_hat, AoP_batch_train)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 打印步骤信息
            if train_steps % dsp_itv == 0:
                print(f'Step {train_steps}, Training Loss: {loss.item()}')

            train_steps += 1

        print(f'Epoch {epoch+1}/{num_epochs} completed.')

        scheduler.step()

        model.eval()
        total_val_loss = 0
        total_S0_PSNR = 0
        total_DoLP_PSNR = 0
        total_AoP_PSNR = 0

        with torch.no_grad():
            for Input_batch_val, Para_batch_val in val_generator:
                Y_batch_val = torch.tensor(Input_batch_val[:, 0:, :, :], dtype=torch.float32).to(device)
                S0_batch_val = torch.tensor(Para_batch_val[:, 0:1, :, :], dtype=torch.float32).to(device)
                DoLP_batch_val = torch.tensor(Para_batch_val[:, 1:2, :, :], dtype=torch.float32).to(device)
                AoP_batch_val = torch.tensor(Para_batch_val[:, 2:3, :, :], dtype=torch.float32).to(device)

                S0_hat_val, DoLP_hat_val, AoP_hat_val = model(Y_batch_val)
                loss_val = LOSS(S0_hat_val, S0_batch_val, DoLP_hat_val, DoLP_batch_val, AoP_hat_val, AoP_batch_val)
                total_val_loss += loss_val.item()

                total_S0_PSNR += psnr(S0_batch_val.cpu().numpy(), S0_hat_val.cpu().numpy(), 2)
                total_DoLP_PSNR += psnr(DoLP_batch_val.cpu().numpy(), DoLP_hat_val.cpu().numpy(), 1)
                total_AoP_PSNR += psnr(AoP_batch_val.cpu().numpy(), AoP_hat_val.cpu().numpy(), math.pi / 2)

        avg_train_loss = total_train_loss / len(train_generator)
        avg_val_loss = total_val_loss / len(val_generator)
        avg_S0_PSNR = total_S0_PSNR / len(val_generator)
        avg_DoLP_PSNR = total_DoLP_PSNR / len(val_generator)
        avg_AoP_PSNR = total_AoP_PSNR / len(val_generator)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        print(f'PSNR - S0: {avg_S0_PSNR}, DoLP: {avg_DoLP_PSNR}, AoP: {avg_AoP_PSNR}')

        if save_best and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'Model saved at {ckpt_path}')

        if early_stop:
            if avg_val_loss < best_val_loss:
                wait = 0
            else:
                wait += 1
                if wait > learning_rate_decay_steps:
                    print('Early stopping')
                    break

        psnr_record.append([avg_S0_PSNR, avg_DoLP_PSNR, avg_AoP_PSNR])

    with open('psnr_record.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(psnr_record)


def main():
    train(model=ForkNet().to(device), patch_width=40, patch_height=40, num_epochs=1, batch_size=256, lr=0.001,
        learning_rate_decay_steps=600, learning_rate_decay_rate=0.988, dsp_itv=6, ckpt_path='model.ckpt',
        save_best=True, early_stop=False)

    
if __name__ == "__main__":
    main()