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

# Parameters
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_STEPS = 600
LEARNING_RATE_DECAY_RATE = 0.988
IMG_NUM = 110
EPOCH_NUM = 300
BATCH_SIZE = 128
PATCH_WIDTH = 40
PATCH_HEIGHT = 40
GPUS = "2"
DSP_ITV = 6
metrics = 'training loss'
save_best = True
early_stop = False
patient = 5
train_img_index_path = './list/train_image_index.list'
val_img_index_path = './list/val_image_index.list'
Y_path = './data/training_set/Y.h5'
labels_path = './data/training_set/Labels.h5'
BIC_path = './data/training_set/BIC.h5'
ckpt_path = './best_model/model_1/model_1.pt'
csv_path = './list/psnr_record_1.csv'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolarizationDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    

def load_data(batch_size = BATCH_SIZE, train_img_index_path = train_img_index_path,
              val_img_index_path = val_img_index_path, Y_path = Y_path, labels_path = labels_path):
    '''
    Divide the training set and validation set.
    Return two generator to generate batches of data.
    '''
    # read data from h5 file
    with h5py.File(Y_path, 'r') as h1:
        Y = np.array(h1.get('y'))

    with h5py.File(labels_path, 'r') as h2:
        label = np.array(h2.get('labels'))

    with h5py.File(BIC_path, 'r') as h3:
        bic = np.array(h3.get('BIC'))

    Input = np.concatenate((Y, bic), axis=-1)

    patch_num = Y.shape[0]
    patch_num_per_img = patch_num // IMG_NUM

    train_img_index_str = open(train_img_index_path).read()
    train_img_index = [int(idx) for idx in train_img_index_str.split(',')]
    # train_img_num = len(train_img_index)

    val_img_index_str = open(val_img_index_path).read()
    val_img_index = [int(idx) for idx in val_img_index_str.split(',')]
    # val_img_num = len(val_img_index)

    patch_index_train = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in train_img_index])
    patch_index_val = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in val_img_index])

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
    # val_bic = bic[patch_index_val]

    return train_steps, train_Y, train_label, val_steps, val_Y, val_para

# def load_data(batch_size=BATCH_SIZE, train_img_index_path=train_img_index_path, val_img_index_path=val_img_index_path, Y_path=Y_path, labels_path=labels_path):
#     with h5py.File(Y_path, 'r') as h1:
#         Y = np.array(h1.get('Y'))

#     with h5py.File(labels_path, 'r') as h2:
#         label = np.array(h2.get('labels'))

#     with h5py.File(BIC_path, 'r') as h3:
#         BIC = np.array(h3.get('BIC'))

#     Input = np.concatenate((Y, BIC), axis=-1)

#     patch_num = Y.shape[0]
#     patch_num_per_img = patch_num // IMG_NUM

#     train_img_index_str = open(train_img_index_path).read()
#     train_img_index = [int(idx) for idx in train_img_index_str.split(',')]

#     val_img_index_str = open(val_img_index_path).read()
#     val_img_index = [int(idx) for idx in val_img_index_str.split(',')]

#     patch_index_train = np.concatenate(
#         [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in train_img_index])
#     patch_index_val = np.concatenate(
#         [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in val_img_index])

#     patch_num_train = len(patch_index_train)
#     patch_num_val = len(patch_index_val)

#     train_Y = Input[patch_index_train]
#     train_label = label[patch_index_train]
#     val_Y = Input[patch_index_val]
#     val_label = label[patch_index_val]

#     train_dataset = PolarizationDataset(train_Y, train_label)
#     val_dataset = PolarizationDataset(val_Y, val_label)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, patch_num_train, patch_num_val


# def train(model,patch_width = PATCH_WIDTH, patch_height = PATCH_HEIGHT, epoch_num = EPOCH_NUM, batch_size = BATCH_SIZE,  learning_rate = LEARNING_RATE,
#           learning_rate_decay_steps = LEARNING_RATE_DECAY_STEPS, learning_rate_decay_rate = LEARNING_RATE_DECAY_RATE,
#           dsp_itv = DSP_ITV, ckpt_path = ckpt_path, save_best = save_best, early_stop = early_stop):
#     Y = torch.randn(batch_size, patch_height, patch_width, 1, dtype=torch.float32)
#     S0 = torch.randn(batch_size, patch_height, patch_width, 1, dtype=torch.float32)
#     AoP = torch.randn(batch_size, patch_height, patch_width, 1, dtype=torch.float32)
#     DoLP =torch.randn(batch_size, patch_height, patch_width, 1, dtype=torch.float32)
    
#     model=ForkNet()
    
    
#     pass



def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
          epochs=5, ckpt_path='model.pth', save_best=True, early_stop=False, 
          metrics='validation loss', patient=5, csv_path='psnr_record.csv'):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForkNet()
    model.to(device)
    
    min_loss = np.inf
    wait = 0
    psnr_record = []
    
    for epoch in range(epochs):
        print(f"=======================================Epoch: {epoch+1}/{epochs}=======================================")
        model.train()
        total_train_loss = 0
        
        for batch_idx, (Y_batch_train, Para_batch_train) in enumerate(train_loader):
            Y_batch_train, Para_batch_train = Y_batch_train.to(device), Para_batch_train.to(device)
            
            optimizer.zero_grad()
            S0_hat, DoLP_hat, AoP_hat = model(Y_batch_train)
            loss = criterion(S0_hat, Para_batch_train[:,:,:,0:1], 
                             DoLP_hat, Para_batch_train[:,:,:,1:2], 
                             AoP_hat, Para_batch_train[:,:,:,2:3])
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        print(f"Training loss: {total_train_loss / len(train_loader)}")
        
        model.eval()
        total_val_loss = 0
        total_S0_PSNR = 0
        total_DoLP_PSNR = 0
        total_AoP_PSNR = 0
        
        with torch.no_grad():
            for Y_batch_val, Para_batch_val in val_loader:
                Y_batch_val, Para_batch_val = Y_batch_val.to(device), Para_batch_val.to(device)
                
                S0_hat_val, DoLP_hat_val, AoP_hat_val = model(Y_batch_val)
                loss = criterion(S0_hat_val, Para_batch_val[:,:,:,0:1], 
                                 DoLP_hat_val, Para_batch_val[:,:,:,1:2], 
                                 AoP_hat_val, Para_batch_val[:,:,:,2:3])
                
                total_val_loss += loss.item()
                
                # Calculate PSNR
                S0_batch_val = Para_batch_val[:,:,:,0].detach().cpu().numpy()
                DoLP_batch_val = Para_batch_val[:,:,:,1].detach().cpu().numpy()
                AoP_batch_val = Para_batch_val[:,:,:,2].detach().cpu().numpy()
                
                S0_hat_val = S0_hat_val.detach().cpu().numpy()
                DoLP_hat_val = DoLP_hat_val.detach().cpu().numpy()
                AoP_hat_val = AoP_hat_val.detach().cpu().numpy()
                
                total_S0_PSNR += psnr(S0_batch_val, S0_hat_val, 2)
                total_DoLP_PSNR += psnr(DoLP_batch_val, DoLP_hat_val, 1)
                total_AoP_PSNR += psnr(AoP_batch_val, AoP_hat_val, math.pi / 2.)
                
        print(f"Validation loss: {total_val_loss / len(val_loader)}")
        print(" ————————————————————————————————————————————————————————————————————————————————")
        print(f"| PSNR of S_0: {total_S0_PSNR / len(val_loader)}    |   PSNR of DoLP: {total_DoLP_PSNR / len(val_loader)}   |")
        print(f"| PSNR of AoP: {total_AoP_PSNR / len(val_loader)}   |")
        print(" ————————————————————————————————————————————————————————————————————————————————")
        
        psnr_record.append([total_S0_PSNR / len(val_loader), total_DoLP_PSNR / len(val_loader), total_AoP_PSNR / len(val_loader)])
        
        if save_best or early_stop:
            if metrics == 'validation loss':
                current_loss = total_val_loss / len(val_loader)
            elif metrics == 'training loss':
                current_loss = total_train_loss / len(train_loader)
                
            if current_loss < min_loss:
                print(f"Validation loss decreased from {min_loss} to {current_loss}")
                min_loss = current_loss
                
                if save_best:
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Model saved in file: {ckpt_path}")
                    
                if early_stop:
                    wait = 0
            else:
                print("Validation loss did not decrease.")
                
                if early_stop:
                    wait += 1
                    
                    if wait > patient:
                        print("Early stop!")
                        break
        
    if not save_best:
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved in file: {ckpt_path}")
        
    with open(csv_path,'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(psnr_record)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    train()
