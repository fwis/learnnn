import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import os
from sr_model import ResNet, MyDataset, CustomLoss, PerceptualLoss, custom_transform
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import GradScaler, autocast

def train(model, train_loader, val_loader, device, num_epochs, learning_rate=0.001, weight_decay=1e-4, checkpoint_path='T3/SR/ckpt/ResNet_sr_5.pth', savebest=True):
    # Model, criterion and optimizer
    model = model.to(device)
    criterion = CustomLoss().to(device)
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    log_dir = "T3/SR/logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    # Load model
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint.')
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        start_time = time.time()
        for i, (data, labels) in enumerate(train_loader):
            inputs = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            with autocast(): 
                output = model(inputs)
                content_loss = criterion(output,labels)
                # print('contentloss', content_loss)
                perceptual_loss = (perceptual_criterion(output[:, 0, :, :].unsqueeze(1),labels[:, 0, :, :].unsqueeze(1)) + 
                                   perceptual_criterion(output[:, 1, :, :].unsqueeze(1),labels[:, 1, :, :].unsqueeze(1)) + 
                                   perceptual_criterion(output[:, 2, :, :].unsqueeze(1),labels[:, 2, :, :].unsqueeze(1)) + 
                                   perceptual_criterion(output[:, 3, :, :].unsqueeze(1),labels[:, 3, :, :].unsqueeze(1)))/4
                # print('perceptual loss', perceptual_loss)
                loss = content_loss + 1e-4 * perceptual_loss
                # print('loss',loss)
                # print('p loss',1e-3*perceptual_loss)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            train_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Train loss: {running_loss / 50:.4f}')
                running_loss = 0.0
                
        # Calculate elapsed time and training loss
        elapsed_time = time.time() - start_time
        writer.add_scalar('Training Time', elapsed_time, epoch + 1)
        avg_train_loss = train_loss / len(train_loader)
        train_loss = 0.0
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                inputs = data.to(device)
                labels = labels.to(device)
                output = model(inputs)
                
                content_loss_val = criterion(labels,output)
                perceptual_loss_val = (perceptual_criterion(output[:, 0, :, :].unsqueeze(1),labels[:, 0, :, :].unsqueeze(1)) + 
                    perceptual_criterion(output[:, 1, :, :].unsqueeze(1),labels[:, 1, :, :].unsqueeze(1)) + 
                    perceptual_criterion(output[:, 2, :, :].unsqueeze(1),labels[:, 2, :, :].unsqueeze(1)) + 
                    perceptual_criterion(output[:, 3, :, :].unsqueeze(1),labels[:, 3, :, :].unsqueeze(1)))/4
                loss = content_loss_val + 1e-4 * perceptual_loss_val
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalars('Loss', {'train loss': avg_train_loss, 
                                    'validate loss':val_loss},global_step=epoch+1)
        
        torch.cuda.empty_cache()
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict()
        }, checkpoint_path)
        print('Checkpoint saved.')

        # Save best
        if savebest:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': best_epoch,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print('Best checkpoint saved.')
        torch.cuda.empty_cache()
    writer.close()
    print('Finished Training')
    
    
if __name__ == "__main__":
    lr = 0.001
    num_epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet()
    batch_size = 64
    weight_decay = 1e-4
    
    train_file_path = r'T3\Frame\data\patches\sr_train\OL_sr_train.h5'
    test_file_path = r'T3\Frame\data\patches\sr_test\OL_sr_test.h5'
    train_dataset = MyDataset(train_file_path, transform=None)
    val_dataset = MyDataset(test_file_path)
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=10, pin_memory=True,  shuffle=False)
    
    train(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=lr, weight_decay=weight_decay, device=device)
    