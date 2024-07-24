import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from sr_model import ResNet
import warnings
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from PIL import Image
import torchvision.transforms as transforms

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet()

# Load checkpoint
checkpoint_path = 'T3\SR\ckpt\ResNet_sr_1_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Read image
image_path = r'.\T3\Frame\test\0_net_input.png'
image = Image.open(image_path).convert('L')  # Convert image to grayscale

# Transform image to tensor
transform = transforms.ToTensor()

img = transform(image).unsqueeze(0).to(device)

# forward
with torch.no_grad():
    data = model(img)
    i0, i45, i90, i135 = data[0][0], data[0][1], data[0][2], data[0][3]
    i0 = i0.unsqueeze(0).unsqueeze(0).cpu()
    i45 = i45.unsqueeze(0).unsqueeze(0).cpu()
    i90 = i90.unsqueeze(0).unsqueeze(0).cpu()
    i135 = i135.unsqueeze(0).unsqueeze(0).cpu()

i0_true = transform(Image.open(r'.\T3\Frame\test\0_0.png').convert('L')).unsqueeze(0).cpu()
i45_true = transform(Image.open(r'.\T3\Frame\test\0_45.png').convert('L')).unsqueeze(0).cpu()
i90_true = transform(Image.open(r'.\T3\Frame\test\0_90.png').convert('L')).unsqueeze(0).cpu()
i135_true = transform(Image.open(r'.\T3\Frame\test\0_135.png').convert('L')).unsqueeze(0).cpu()

SSIM_i0 = SSIM(i0,i0_true,data_range=1.0)
SSIM_i45 = SSIM(i45,i45_true,data_range=1.0)
SSIM_i90 = SSIM(i90,i90_true,data_range=1.0)
SSIM_i135 = SSIM(i135,i135_true,data_range=1.0)

RMSE_i0 = torch.mean((i0-i0_true)**2)
RMSE_i45 = torch.mean((i45-i45_true)**2)
RMSE_i90 = torch.mean((i90-i90_true)**2)
RMSE_i135 = torch.mean((i135-i135_true)**2)

PSNR_i0= PSNR(i0,i0_true,data_range=1.0)
PSNR_i45= PSNR(i45,i45_true,data_range=1.0)
PSNR_i90= PSNR(i90,i90_true,data_range=1.0)
PSNR_i135= PSNR(i135,i135_true,data_range=1.0)

print(f'SSIM of I0: {SSIM_i0}')
print(f'SSIM of I45: {SSIM_i45}')
print(f'SSIM of I90: {SSIM_i90}')
print(f'SSIM of I135: {SSIM_i135}\n')

print(f'RMSE of I0: {RMSE_i0}')
print(f'RMSE of I45: {RMSE_i45}')
print(f'RMSE of I90: {RMSE_i90}')
print(f'RMSE of I135: {RMSE_i135}\n')

print(f'PSNR of I0: {PSNR_i0}')
print(f'PSNR of I45: {PSNR_i45}')
print(f'PSNR of I90: {PSNR_i90}')
print(f'PSNR of I135: {PSNR_i135}')


i0 = i0.cpu().squeeze().numpy()
i45 = i45.cpu().squeeze().numpy()
i90 = i90.cpu().squeeze().numpy()
i135 = i135.cpu().squeeze().numpy()

i0_true = i0_true.cpu().squeeze().numpy()
i45_true = i45_true.cpu().squeeze().numpy()
i90_true = i90_true.cpu().squeeze().numpy()
i135_true = i135_true.cpu().squeeze().numpy()

# Plot results
i0_min, i0_max = 0, 1
i45_min, i45_max = 0, 1
i90_min, i90_max = 0, 1
i135_min, i135_max = 0, 1

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

im0 = axs[0].imshow(i0, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[0].set_title('I0')
fig.colorbar(im0, ax=axs[0], orientation='vertical')
im45 = axs[1].imshow(i45, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[1].set_title('I45')
fig.colorbar(im45, ax=axs[1], orientation='vertical')
im90 = axs[2].imshow(i90, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[2].set_title('I90')
fig.colorbar(im90, ax=axs[2], orientation='vertical')
im135 = axs[3].imshow(i135, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[3].set_title('I135')
fig.colorbar(im135, ax=axs[3], orientation='vertical')

plt.tight_layout()
plt.show()

# Plot results
i0_min, i0_max = 0, 1
i45_min, i45_max = 0, 1
i90_min, i90_max = 0, 1
i135_min, i135_max = 0, 1

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

im0 = axs[0].imshow(i0_true, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[0].set_title('I0')
fig.colorbar(im0, ax=axs[0], orientation='vertical')
im45 = axs[1].imshow(i45_true, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[1].set_title('I45')
fig.colorbar(im45, ax=axs[1], orientation='vertical')
im90 = axs[2].imshow(i90_true, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[2].set_title('I90')
fig.colorbar(im90, ax=axs[2], orientation='vertical')
im135 = axs[3].imshow(i135_true, cmap='gray', vmin=i0_min, vmax=i0_max)
axs[3].set_title('I135')
fig.colorbar(im135, ax=axs[3], orientation='vertical')

plt.tight_layout()
plt.show()