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
