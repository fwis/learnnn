import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import os
from model import MyDataset, ResNetGenerator, Discriminator, CustomGANLoss, custom_transform
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import GradScaler, autocast

def train(generator, discriminator, train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, weight_decay=1e-4, checkpoint_path='T3/Frame/ckpt/GAN1.pth', savebest=True):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    # Loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion = CustomGANLoss().to(device)
    pixel_criterion = nn.L1Loss().to(device)
    
    # TensorBoard
    log_dir = "T3/Frame/logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f'Loaded checkpoint.')
        
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        start_time = time.time()
        
        for i, (data, aop, dolp, s0) in enumerate(train_loader):
            inputs = data.to(device)
            aop_true = aop.to(device)
            dolp_true = dolp.to(device)
            s0_true = s0.to(device)
            
            # Update generator
            optimizer_G.zero_grad()
            
            with autocast():
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                pixel_loss = pixel_criterion(s0_pred, s0_true)
                content_loss = content_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
                disc_outputs = discriminator(aop_pred, dolp_pred, s0_pred)
                adversarial_loss = (adversarial_criterion(disc_outputs[0], aop_true) +
                                    adversarial_criterion(disc_outputs[1], dolp_true) +
                                    adversarial_criterion(disc_outputs[2], s0_true)) / 3
                loss_G = pixel_loss + content_loss + 1e-3 * adversarial_loss

            scaler_G.scale(loss_G).backward(retain_graph=True)
            scaler_G.step(optimizer_G)
            scaler_G.update()
            running_loss_G += loss_G.item()
            
            with torch.no_grad():
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                
            optimizer_D.zero_grad()
            
            with autocast():
                real_loss = (adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[0], aop_true) +
                             adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[1], dolp_true) +
                             adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[2], s0_true)) / 3
                fake_loss = (adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[0], aop_pred) +
                             adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[1], dolp_pred) +
                             adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[2], s0_pred)) / 3
                loss_D = (real_loss + fake_loss) / 2

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            running_loss_D += loss_D.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Generator loss: {running_loss_G / 50:.4f}, Discriminator loss: {running_loss_D / 50:.4f}')
                running_loss_G = 0.0
                running_loss_D = 0.0
                
        # Calculate elapsed time and training loss
        elapsed_time = time.time() - start_time
        writer.add_scalar('Training Time', elapsed_time, epoch + 1)
        
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)
        writer.add_scalars('Loss', {'Generator loss': avg_loss_G, 'Discriminator loss': avg_loss_D}, global_step=epoch + 1)

        # Validation
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, aop, dolp, s0 in val_loader:
                inputs = data.to(device)
                aop_true = aop.to(device)
                dolp_true = dolp.to(device)
                s0_true = s0.to(device)
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                
                loss = content_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        
        torch.cuda.empty_cache()

        # Save checkpoint
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict()
        }, checkpoint_path)
        print('Checkpoint saved.')

        # Save best
        if savebest:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'epoch': best_epoch,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print('Best checkpoint saved.')
        torch.cuda.empty_cache()
    writer.close()
    print('Finished Training')


if __name__ == "__main__":
    lr = 0.001
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = ResNetGenerator()
    discriminator = Discriminator()

    batch_size = 40
    weight_decay = 1e-4

    train_file_path = r'T3\Frame\data\patches\train_patches_100\OL_train_100.h5'
    test_file_path = r'T3\Frame\data\patches\test_patches_100\OL_test_100.h5'
    train_dataset = MyDataset(train_file_path, transform=custom_transform)
    val_dataset = MyDataset(test_file_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=10, pin_memory=True, shuffle=False)
    
    train(generator=generator, discriminator=discriminator, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=lr, weight_decay=weight_decay, device=device)