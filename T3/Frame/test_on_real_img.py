import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from model import MyDataset, ForkNet, ResNet, UncertaintyWeightedLoss, ResNetFPN
import warnings
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet()
criterion = UncertaintyWeightedLoss()
# model = ForkNet()
model = ResNetFPN()

# Load checkpoint
checkpoint_path = r'ResNet.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

input_data_path = r'D:\projects\learnnn\T3\Frame\test\83_data.pt'
img = torch.load(input_data_path).unsqueeze(0).unsqueeze(0).to(device)

# forward
with torch.no_grad():
    aop, dolp, s0 = model(img)

aop = aop.cpu()
dolp= dolp.cpu()
s0 = s0.cpu()

aop_true = torch.load(r'D:\projects\learnnn\T3\Frame\test\83_aop.pt').cpu().unsqueeze(0).unsqueeze(0)
dolp_true = torch.load(r'D:\projects\learnnn\T3\Frame\test\83_dolp.pt').cpu().unsqueeze(0).unsqueeze(0)
s0_true = torch.load(r'D:\projects\learnnn\T3\Frame\test\83_s0.pt').cpu().unsqueeze(0).unsqueeze(0)/2

SSIM_aop = SSIM(aop,aop_true,data_range=torch.pi/2)
SSIM_dolp = SSIM(dolp,dolp_true,data_range=1.0)
SSIM_s0 = SSIM(s0,s0_true,data_range=1.0)

RMSE_s0 = torch.mean((s0-s0_true)**2)
RMSE_aop = torch.mean((aop-aop_true)**2)
RMSE_dolp = torch.mean((dolp-dolp_true)**2)

PSNR_aop = PSNR(aop,aop_true,data_range=torch.pi/2)
PSNR_dolp = PSNR(dolp,dolp_true,data_range=1.0)
PSNR_s0= PSNR(s0,s0_true,data_range=1.0)

print(f'SSIM of AoP: {SSIM_aop}')
print(f'SSIM of DoLP: {SSIM_dolp}')
print(f'SSIM of S0: {SSIM_s0}\n')

print(f'RMSE of AoP: {RMSE_aop}')
print(f'RMSE of DoLP: {RMSE_dolp}')
print(f'RMSE of S0: {RMSE_s0}\n')

print(f'PSNR of AoP: {PSNR_aop}')
print(f'PSNR of DoLP: {PSNR_dolp}')
print(f'PSNR of S0: {PSNR_s0}')

aop = aop.cpu().squeeze().numpy()
dolp = dolp.cpu().squeeze().numpy()
s0 = s0.cpu().squeeze().numpy()

aop_true = aop_true.cpu().squeeze().numpy()
dolp_true = dolp_true.cpu().squeeze().numpy()
s0_true = s0_true.cpu().squeeze().numpy()

# Plot results
dolp_min, dolp_max = 0, 1
s0_min, s0_max = 0, 1
aop_min, aop_max = 0, np.pi / 2

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im0 = axs[0].imshow(aop, cmap='gray', vmin=aop_min, vmax=aop_max)
axs[0].set_title('AOP')
fig.colorbar(im0, ax=axs[0], orientation='vertical')
im1 = axs[1].imshow(dolp, cmap='gray', vmin=dolp_min, vmax=dolp_max)
axs[1].set_title('DoLP')
fig.colorbar(im1, ax=axs[1], orientation='vertical')
im2 = axs[2].imshow(s0, cmap='gray', vmin=s0_min, vmax=s0_max)
axs[2].set_title('S0')
fig.colorbar(im2, ax=axs[2], orientation='vertical')

plt.tight_layout()
plt.show()

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im0 = axs[0].imshow(aop_true, cmap='gray', vmin=aop_min, vmax=aop_max)
axs[0].set_title('AOP')
fig.colorbar(im0, ax=axs[0], orientation='vertical')
im1 = axs[1].imshow(dolp_true, cmap='gray', vmin=dolp_min, vmax=dolp_max)
axs[1].set_title('DoLP')
fig.colorbar(im1, ax=axs[1], orientation='vertical')
im2 = axs[2].imshow(s0_true, cmap='gray', vmin=s0_min, vmax=s0_max)
axs[2].set_title('S0')
fig.colorbar(im2, ax=axs[2], orientation='vertical')

plt.tight_layout()
plt.show()