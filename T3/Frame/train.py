import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import os
from model import MyDataset, ForkNet, ResNet, CustomLoss, custom_transform, ResNetFPN, ConvNeXtNet, ForkLoss
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import GradScaler, autocast


def train(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001, weight_decay=1e-4, checkpoint_path='T3/Frame/ckpt/ForkNet_OL.pth', savebest=True):
    # Model, criterion and optimizer
    model = model.to(device)
    # criterion = CustomLoss().to(device)
    criterion = ForkLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()  # Initialize gradient scaler for mixed precision training
    log_dir = "T3/Frame/logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
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
        for i, (data, aop, dolp, s0) in enumerate(train_loader):
            inputs = data.to(device)
            aop_true = aop.to(device)
            dolp_true = dolp.to(device)
            s0_true = s0.to(device)
            
            optimizer.zero_grad()
            
            with autocast():  # Enable autocast for mixed precision training
                aop_pred, dolp_pred, s0_pred = model(inputs)
                loss = criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)

            scaler.scale(loss).backward()  # Scale loss for mixed precision
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()  # Update the scale for next iteration
            
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
            for data, aop, dolp, s0 in val_loader:
                inputs = data.to(device)
                aop_true = aop.to(device)
                dolp_true = dolp.to(device)
                s0_true = s0.to(device)
                aop_pred, dolp_pred, s0_pred = model(inputs)
                
                loss = criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
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
    model = ForkNet()
    # model = ResNet()
    # model = ResNetFPN()
    # model = ConvNeXtNet()
    batch_size = 64
    weight_decay = 1e-4
    
    train_file_path = r'T3\Frame\data\patches\OL_train1.h5'
    test_file_path = r'T3\Frame\data\patches\OL_test1.h5'
    train_dataset = MyDataset(train_file_path, transform=custom_transform)
    val_dataset = MyDataset(test_file_path)
    
    # train_file_path1 = r"T3\Frame\data\patches\train_patches_100\OL_train_100.h5"
    # train_file_path2 = r"T3\Frame\data\patches\train_patches_100\Fork_train_100.h5"
    # train_file_path3 = r"T3\Frame\data\patches\train_patches_100\pid_train_100.h5"
    # train_file_path4 = r"T3\Frame\data\patches\train_patches_100\PIF_train_100.h5"
    # train_file_path5 = r"T3\Frame\data\patches\train_patches_100\tokyo_train_100.h5"
    
    # test_file_path1 = r'T3\Frame\data\patches\test_patches_100\OL_test_100.h5'
    # test_file_path2 = r"T3\Frame\data\patches\test_patches_100\Fork_test_100.h5"
    # test_file_path3 = r"T3\Frame\data\patches\test_patches_100\pid_test_100.h5"
    # test_file_path4 = r"T3\Frame\data\patches\test_patches_100\PIF_test_100.h5"
    # test_file_path5 = r"T3\Frame\data\patches\test_patches_100\tokyo_test_100.h5"

    # train_dataset1 = MyDataset(file_path=train_file_path1, transform=custom_transform)
    # val_dataset1 = MyDataset(file_path=test_file_path1, transform=None)
    # train_dataset2 = MyDataset(file_path=train_file_path2, transform=custom_transform)
    # val_dataset2 = MyDataset(file_path=test_file_path2, transform=None)
    # train_dataset3 = MyDataset(file_path=train_file_path3, transform=custom_transform)
    # val_dataset3 = MyDataset(file_path=test_file_path3, transform=None)
    # train_dataset4 = MyDataset(file_path=train_file_path4, transform=custom_transform)
    # val_dataset4 = MyDataset(file_path=test_file_path4, transform=None)
    # train_dataset5 = MyDataset(file_path=train_file_path5, transform=custom_transform)
    # val_dataset5 = MyDataset(file_path=test_file_path5, transform=None)
    
    # train_dataset = ConcatDataset([train_dataset1,train_dataset2,train_dataset3,train_dataset4,train_dataset5])
    # val_dataset = ConcatDataset([val_dataset1,val_dataset2,val_dataset3,val_dataset4,val_dataset5])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, num_workers=10, pin_memory=True,  shuffle=False)
    
    train(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=lr, weight_decay=weight_decay, device=device)
    