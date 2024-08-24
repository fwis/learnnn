import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sr_model import MyDataset, ResNet
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import mean_squared_error as MSE

def test(model, dataloader, device='cuda'):
    # Set the model to evaluation mode
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # SSIM_aop = 0.0
        # SSIM_dolp = 0.0
        # SSIM_s0 = 0.0
        # PSNR_aop = 0.0
        # PSNR_dolp = 0.0
        # PSNR_s0 = 0.0
        # RMSE_aop = 0.0
        # RMSE_dolp = 0.0
        # RMSE_s0 = 0.0
        SSIM_I0 = 0.0
        SSIM_I45 = 0.0
        SSIM_I90 = 0.0
        SSIM_I135 = 0.0
        PSNR_I0 = 0.0
        PSNR_I45 = 0.0
        PSNR_I90 = 0.0
        PSNR_I135 = 0.0
        RMSE_I0 = 0.0
        RMSE_I45 = 0.0
        RMSE_I90 = 0.0
        RMSE_I135 = 0.0
        num_samples = len(dataloader)
        
        for i, (data, labels) in enumerate(dataloader):
            inputs = data.to(device)
            labels = labels.to(device)
            output = model(inputs)
            I0_pred = output[:, 0, :, :].unsqueeze(1)
            I45_pred = output[:, 1, :, :].unsqueeze(1)
            I90_pred = output[:, 2, :, :].unsqueeze(1)
            I135_pred = output[:, 3, :, :].unsqueeze(1)
            I0_true = labels[:, 0, :, :].unsqueeze(1)
            I45_true = labels[:, 1, :, :].unsqueeze(1)
            I90_true = labels[:, 2, :, :].unsqueeze(1)
            I135_true = labels[:, 3, :, :].unsqueeze(1)
            
            # Calculate SSIM, PSNR, MSE
            SSIM_I0 += SSIM(I0_pred, I0_true, data_range=1.0).item()
            SSIM_I45 += SSIM(I45_pred, I45_true, data_range=1.0).item()
            SSIM_I90 += SSIM(I90_pred, I90_true, data_range=1.0).item()
            SSIM_I135 += SSIM(I135_pred, I135_true, data_range=1.0).item()

            RMSE_I0 += torch.mean((I0_pred - I0_true) ** 2).item()
            RMSE_I45 += torch.mean((I45_pred - I45_true) ** 2).item()
            RMSE_I90 += torch.mean((I90_pred - I90_true) ** 2).item()
            RMSE_I135 += torch.mean((I135_pred - I135_true) ** 2).item()

            PSNR_I0 += PSNR(I0_pred, I0_true, data_range=1.0).item()
            PSNR_I45 += PSNR(I45_pred, I45_true, data_range=1.0).item()
            PSNR_I90 += PSNR(I90_pred, I90_true, data_range=1.0).item()
            PSNR_I135 += PSNR(I135_pred, I135_true, data_range=1.0).item()

        SSIM_I0 /= num_samples
        SSIM_I45 /= num_samples
        SSIM_I90 /= num_samples
        SSIM_I135 /= num_samples
        PSNR_I0 /= num_samples
        PSNR_I45 /= num_samples
        PSNR_I90 /= num_samples
        PSNR_I135 /= num_samples
        RMSE_I0 /= num_samples
        RMSE_I45 /= num_samples
        RMSE_I90 /= num_samples
        RMSE_I135 /= num_samples
        
        print(f'SSIM of I0: {SSIM_I0}')
        print(f'SSIM of I45: {SSIM_I45}')
        print(f'SSIM of I90: {SSIM_I90}')
        print(f'SSIM of I135: {SSIM_I135}\n')

        print(f'RMSE of I0: {RMSE_I0}')
        print(f'RMSE of I45: {RMSE_I45}')
        print(f'RMSE of I90: {RMSE_I90}')
        print(f'RMSE of I135: {RMSE_I135}\n')

        print(f'PSNR of I0: {PSNR_I0}')
        print(f'PSNR of I45: {PSNR_I45}')
        print(f'PSNR of I90: {PSNR_I90}')
        print(f'PSNR of I135: {PSNR_I135}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_file_path = r'T3\Frame\data\patches\sr_test\OL_sr_test.h5'
val_dataset = MyDataset(file_path=test_file_path, transform=None)
val_loader = DataLoader(val_dataset, batch_size=10, num_workers=0, pin_memory=True,  shuffle=False)
model = ResNet()
# model = ForkNet()
checkpoint_path = 'T3/SR/ckpt/ResNet_sr_5_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
test(model=model, dataloader=val_loader)
