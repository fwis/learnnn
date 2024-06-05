import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import csv
from model import MyDataset, ForkNet, ResNet
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

def train(model, dataset, device, num_epochs=10, batch_size=32, learning_rate=0.001, train_ratio=0.8, checkpoint_path='ResNet.pth'):
    # Split the training set and the test set
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, criterion and optimizer
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    
    # Load model
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f'Loaded best model from epoch {best_epoch} with loss {best_loss:.4f}')
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (data, aop, dolp, s0) in enumerate(train_loader):
            inputs = data.to(device)
            aop_labels = aop.to(device)
            dolp_labels = dolp.to(device)
            s0_labels = s0.to(device)
            
            optimizer.zero_grad()
            
            aop_preds, dolp_preds, s0_preds = model(inputs)
            
            loss_aop = criterion(aop_preds, aop_labels)
            loss_dolp = criterion(dolp_preds, dolp_labels)
            loss_s0 = criterion(s0_preds, s0_labels)
            loss = loss_aop + loss_dolp + loss_s0
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Train loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, aop, dolp, s0 in val_loader:
                inputs = data.to(device)
                aop_labels = aop.to(device)
                dolp_labels = dolp.to(device)
                s0_labels = s0.to(device)
                
                aop_preds, dolp_preds, s0_preds = model(inputs)
                
                loss_aop = criterion(aop_preds, aop_labels)
                loss_dolp = criterion(dolp_preds, dolp_labels)
                loss_s0 = criterion(s0_preds, s0_labels)
                loss = loss_aop + loss_dolp + loss_s0
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f'Best model saved with loss: {best_loss:.4f}')
    
    print('Finished Training')
    

    
    
if __name__ == "__main__":

    lr = 0.001
    num_epochs = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ForkNet()
    model = ResNet()
    file_path = r"D:\WORKS\dataset\patch_data\data_ol.h5"
    # file_path = r"D:\WORKS\dataset\patch_data\data_pif.h5"
    batch_size = 16
    custom_dataset = MyDataset(file_path)

    train(model=model, dataset=custom_dataset, num_epochs=1, batch_size=batch_size, device=device)
    