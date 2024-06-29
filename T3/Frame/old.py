# def train(model, dataloader, device, num_epochs, lr=0.001):
#     model = model.to(device)
#     criterion = nn.MSELoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (data,aop,dolp,s0) in enumerate(dataloader, 0):

#             inputs = data.to(device)
#             aop_labels = aop.to(device)
#             dolp_labels = dolp.to(device)
#             s0_labels = s0.to(device)
#             # print(inputs.shape)
#             optimizer.zero_grad()

#             aop_preds, dolp_preds, s0_preds = model(inputs)

#             loss_aop = criterion(aop_preds, aop_labels)
#             loss_dolp = criterion(dolp_preds, dolp_labels)
#             loss_s0 = criterion(s0_preds, s0_labels)
#             loss = loss_aop + loss_dolp + loss_s0

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if i % 5 == 4:    # Print per 5 batches
#                 print('[epoch:%1d, batch:%1d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 5))
#                 running_loss = 0.0
#                 # print(f'inputs device: {inputs.device}')
#                 # print(f'aop_preds device: {aop_preds.device}')
#                 # print(f'loss device: {loss.device}')
#                 # print(f'model device: {next(model.parameters()).device}')
#         # torch.cuda.synchronize()       
#     print('Finished Training!')
    
# lr = 0.001
# num_epochs = 1
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # model = ForkNet()
# model = ResNet()
# file_path = r"D:\WORKS\dataset\patch_data\data_ol.h5"
# batch_size = 12

# custom_dataset = MyDataset(file_path)

# data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# train(model=model, dataloader=data_loader, device=device, num_epochs=num_epochs,lr=lr)



        
        # # Save best
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_loss': best_loss,
        #     }, checkpoint_path)
        #     print(f'Best model saved with loss: {best_loss:.4f}')
         
         
         
# def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
#         # Physics informed loss        
#         Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
#         U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
#         Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
#         U_true = dolp_true * s0_true * torch.sin(2 * aop_true)

#         loss_Q = torch.mean(Q_pred - Q_true)
#         loss_U = torch.mean(U_pred - U_true)
#         loss_dolp = torch.mean((dolp_pred - dolp_true))**2
#         loss_s0 = torch.mean((s0_pred - s0_true))**2

#         physics_loss = torch.abs(loss_s0 * loss_dolp - (loss_Q**2 + loss_U**2))

#         # Total loss
#         total_loss  = torch.mean(0.1 * torch.square(s0_true - s0_pred) + 
#                         0.5 * torch.square(dolp_true - dolp_pred) + 
#                         0.05 * torch.square(aop_true - aop_pred)) + 0.01 * physics_loss - 0.02 * torch.log(torch.relu(SSIM(aop_pred,aop_true)))
                
#         return total_loss



    
        # T1 = Q_true**2 + U_true**2
        # T2 = nn.Sigmoid()(U_true/(Q_true + 1e-6))
        # P1 = Q_pred**2 + U_pred**2
        # P2 = nn.Sigmoid()(U_pred/(Q_pred + 1e-6))
        # total_loss =  torch.mean(0.5 * ((T1 - P1)/2)**2 + 0.5* (T2 - P2)**2)
        

    
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Initialize log-variance parameters for each loss component
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # Data-based loss
        mse_loss_aop = nn.MSELoss()(aop_pred, aop_true)
        mse_loss_dolp = nn.MSELoss()(dolp_pred, dolp_true)
        mse_loss_s0 = nn.MSELoss()(s0_pred, s0_true)

        # Physics-informed loss
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * aop_pred)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * aop_pred)
        Q_true = dolp_true * s0_true * torch.cos(2 * aop_true)
        U_true = dolp_true * s0_true * torch.sin(2 * aop_true)

        ssim_loss_Q = 1 - SSIM(Q_pred, Q_true)
        ssim_loss_U = 1 - SSIM(U_pred, U_true)
        ssim_loss_Q = torch.relu(ssim_loss_Q)
        ssim_loss_U = torch.relu(ssim_loss_U)
        # loss_Q = nn.MSELoss()(Q_pred, Q_true)
        # loss_U = nn.MSELoss()(U_pred, U_true)
        
        # Calculate the total loss using uncertainty weighting
        loss_aop = mse_loss_aop * torch.exp(-self.log_vars[0]) + self.log_vars[0]
        loss_dolp = mse_loss_dolp * torch.exp(-self.log_vars[1]) + self.log_vars[1]
        loss_s0 = mse_loss_s0 * torch.exp(-self.log_vars[2]) + self.log_vars[2]
        phys_loss_Q = ssim_loss_Q * torch.exp(-self.log_vars[3]) + self.log_vars[3]
        phys_loss_U = ssim_loss_U * torch.exp(-self.log_vars[4]) + self.log_vars[4]
        
        loss_aop = torch.relu(loss_aop)
        loss_dolp = torch.relu(loss_dolp)
        loss_s0 = torch.relu(loss_s0)
        phys_loss_Q = torch.relu(phys_loss_Q)
        phys_loss_U = torch.relu(phys_loss_U)
        regularization_loss = torch.sum(torch.abs(self.log_vars))
        
        total_loss = loss_aop + loss_dolp + loss_s0 + phys_loss_Q + phys_loss_U + 0.01 * regularization_loss

        return total_loss
    
    
class GradNormLoss(nn.Module):
    def __init__(self, model, alpha=0.5):
        super(GradNormLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        mse_loss_aop = nn.MSELoss()(aop_pred, aop_true)
        mse_loss_dolp = nn.MSELoss()(dolp_pred, dolp_true)
        mse_loss_s0 = nn.MSELoss()(s0_pred, s0_true)

        precision_aop = torch.exp(-self.log_vars[0])
        precision_dolp = torch.exp(-self.log_vars[1])
        precision_s0 = torch.exp(-self.log_vars[2])

        loss_aop = mse_loss_aop * precision_aop + self.log_vars[0]
        loss_dolp = mse_loss_dolp * precision_dolp + self.log_vars[1]
        loss_s0 = mse_loss_s0 * precision_s0 + self.log_vars[2]

        total_loss = loss_aop + loss_dolp + loss_s0

        # GradNorm
        grads = torch.autograd.grad(total_loss, self.model.parameters(), create_graph=True)
        grads_norm = torch.stack([grad.norm() for grad in grads]).sum()

        grad_norm_loss = (self.alpha / grads_norm) * total_loss

        return grad_norm_loss
    
    
    
    
# def train(generator, discriminator, train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, weight_decay=1e-4, checkpoint_path='T3/Frame/ckpt/GAN3.pth', savebest=True):
#     generator = generator.to(device)
#     discriminator = discriminator.to(device)
    
#     optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     scaler_G = GradScaler()
#     scaler_D = GradScaler()
    
#     # Loss functions
#     adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
#     content_criterion = CustomGANLoss().to(device)
#     # pixel_criterion = nn.L1Loss().to(device)
    
#     # TensorBoard
#     log_dir = "T3/Frame/logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir)
    
#     best_val_loss = float('inf')

#     # Load checkpoint if exists
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         generator.load_state_dict(checkpoint['generator_state_dict'])
#         discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
#         print(f'Loaded checkpoint.')
        
#     for epoch in range(num_epochs):
#         generator.train()
#         discriminator.train()
#         running_loss_G = 0.0
#         running_loss_D = 0.0
#         train_loss_G = 0.0
#         train_loss_D = 0.0
#         start_time = time.time()
        
#         for i, (data, aop, dolp, s0) in enumerate(train_loader):
#             inputs = data.to(device)
#             aop_true = aop.to(device)
#             dolp_true = dolp.to(device)
#             s0_true = s0.to(device)
            
#             # Update generator
#             optimizer_G.zero_grad()
            
#             with autocast():
#                 aop_pred, dolp_pred, s0_pred = generator(inputs)
#                 # pixel_loss = pixel_criterion(s0_pred, s0_true)
#                 content_loss = content_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
#                 disc_outputs = discriminator(aop_pred, dolp_pred, s0_pred)
#                 adversarial_loss = (adversarial_criterion(disc_outputs[0], aop_true) +
#                                     adversarial_criterion(disc_outputs[1], dolp_true) +
#                                     adversarial_criterion(disc_outputs[2], s0_true)) / 3
#                 loss_G = content_loss + 1e-3 * adversarial_loss

#             scaler_G.scale(loss_G).backward(retain_graph=True)
#             scaler_G.step(optimizer_G)
#             scaler_G.update()
#             running_loss_G += loss_G.item()
#             train_loss_G += loss_G.item()
            
#             with torch.no_grad():
#                 aop_pred, dolp_pred, s0_pred = generator(inputs)
                
#             optimizer_D.zero_grad()
            
#             with autocast():
#                 real_loss = (adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[0], aop_true) +
#                              adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[1], dolp_true) +
#                              adversarial_criterion(discriminator(aop_true, dolp_true, s0_true)[2], s0_true)) / 3
#                 fake_loss = (adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[0], aop_pred) +
#                              adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[1], dolp_pred) +
#                              adversarial_criterion(discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())[2], s0_pred)) / 3
#                 loss_D = (real_loss + fake_loss) / 2

#             scaler_D.scale(loss_D).backward()
#             scaler_D.step(optimizer_D)
#             scaler_D.update()
#             running_loss_D += loss_D.item()
#             train_loss_D += loss_D.item()
            
#             if i % 100 == 99:
#                 print(f'[Epoch {epoch + 1}, Batch {i + 1}] Generator loss: {running_loss_G / 100:.4f}, Discriminator loss: {running_loss_D / 100:.4f}')
#                 running_loss_G = 0.0
#                 running_loss_D = 0.0
                
#         # Calculate elapsed time and training loss
#         elapsed_time = time.time() - start_time
#         writer.add_scalar('Training Time', elapsed_time, epoch + 1)
        
#         avg_loss_G = train_loss_G / len(train_loader)
#         avg_loss_D = train_loss_D / len(train_loader)
#         writer.add_scalars('Loss', {'Generator loss': avg_loss_G, 'Discriminator loss': avg_loss_D}, global_step=epoch + 1)
#         torch.cuda.empty_cache()
        
#         # Validation
#         generator.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for data, aop, dolp, s0 in val_loader:
#                 inputs = data.to(device)
#                 aop_true = aop.to(device)
#                 dolp_true = dolp.to(device)
#                 s0_true = s0.to(device)
#                 aop_pred, dolp_pred, s0_pred = generator(inputs)
                
#                 loss = content_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
#                 val_loss += loss.item()
        
#         val_loss /= len(val_loader)
#         print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
#         writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        
#         torch.cuda.empty_cache()

#         # Save checkpoint
#         torch.save({
#             'generator_state_dict': generator.state_dict(),
#             'discriminator_state_dict': discriminator.state_dict()
#         }, checkpoint_path)
#         print('Checkpoint saved.')

#         # Save best
#         if savebest:
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch + 1
#                 best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
#                 torch.save({
#                     'generator_state_dict': generator.state_dict(),
#                     'discriminator_state_dict': discriminator.state_dict(),
#                     'epoch': best_epoch,
#                     'val_loss': val_loss,
#                 }, best_checkpoint_path)
#                 print('Best checkpoint saved.')
#         torch.cuda.empty_cache()
#     writer.close()
#     print('Finished Training')
