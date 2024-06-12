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