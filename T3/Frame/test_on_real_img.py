import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from model import MyDataset, ForkNet, ResNet, ResNetFPN, ResNetGenerator
import warnings
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
import cv2

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")


def normalize(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
        norm_data = np.zeros(data.shape)
    else:  
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data

'''
Calculate the AoP
'''
def aop(x_0, x_45, x_90, x_135, normalization = True):
    AoP = 0.5 * np.arctan2((x_45 - x_135), (x_0 - x_90)) # range in [-pi/2, pi/2]
    
    if normalization:
        AoP = (AoP + np.pi / 2) / np.pi

    return AoP

'''
Calculate the DoLP
'''
def dolp(x_0, x_45, x_90, x_135, normalization = False):
    s0 = 0.5*(x_0 + x_45 + x_90 + x_135)   
    DoLP = np.sqrt(np.square(x_0-x_90) + np.square(x_45-x_135))/(s0+1e-8)
    DoLP[np.where(s0==0)] = 0
    if normalization:
        DoLP = normalize(DoLP,0,1)
    
    return DoLP



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet()
model = ResNetGenerator()
# criterion = UncertaintyWeightedLoss()
# model = ForkNet()
# model = ResNetFPN()

# Load checkpoint
checkpoint_path = 'T3\Frame\ckpt\GAN1_OL_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['generator_state_dict'])
model = model.to(device)
model.eval()

# Load and normalize the image data to [0, 1]
input_image_path = 'T3/Frame/test/dofp_ol.png'
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
img = img / 255.0  # Normalize to [0, 1]
img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

i0 =cv2.imread('T3/Frame/test/83_gt_0.png', cv2.IMREAD_GRAYSCALE) / 255.0
i45 =cv2.imread('T3/Frame/test/83_gt_45.png', cv2.IMREAD_GRAYSCALE) / 255.0
i90 =cv2.imread('T3/Frame/test/83_gt_90.png', cv2.IMREAD_GRAYSCALE) / 255.0
i135 =cv2.imread('T3/Frame/test/83_gt_135.png', cv2.IMREAD_GRAYSCALE) / 255.0

# Load and normalize the true images
aop_true = aop(i0,i45,i90,i135)
dolp_true = dolp(i0,i45,i90,i135)
s0_true = 0.25 * (i0+i45+i90+i135)

# Convert the true images to tensors
aop_true = torch.from_numpy(aop_true).float().unsqueeze(0).unsqueeze(0)
dolp_true = torch.from_numpy(dolp_true).float().unsqueeze(0).unsqueeze(0)
s0_true = torch.from_numpy(s0_true).float().unsqueeze(0).unsqueeze(0)

# forward
with torch.no_grad():
    aop_pred, dolp_pred, s0_pred = model(img)

aop_pred = aop_pred.cpu()
dolp_pred= dolp_pred.cpu()
s0_pred = s0_pred.cpu()


SSIM_aop = SSIM(aop_pred,aop_true,data_range=1.0)
SSIM_dolp = SSIM(dolp_pred,dolp_true,data_range=1.0)
SSIM_s0 = SSIM(s0_pred,s0_true,data_range=1.0)

RMSE_s0 = torch.mean((s0_pred-s0_true)**2)
RMSE_aop = torch.mean((aop_pred-aop_true)**2)
RMSE_dolp = torch.mean((dolp_pred-dolp_true)**2)

PSNR_aop = PSNR(aop_pred,aop_true,data_range=1.0)
PSNR_dolp = PSNR(dolp_pred,dolp_true,data_range=1.0)
PSNR_s0= PSNR(s0_pred,s0_true,data_range=1.0)

print(f'SSIM of AoP: {SSIM_aop}')
print(f'SSIM of DoLP: {SSIM_dolp}')
print(f'SSIM of S0: {SSIM_s0}\n')

print(f'RMSE of AoP: {RMSE_aop}')
print(f'RMSE of DoLP: {RMSE_dolp}')
print(f'RMSE of S0: {RMSE_s0}\n')

print(f'PSNR of AoP: {PSNR_aop}')
print(f'PSNR of DoLP: {PSNR_dolp}')
print(f'PSNR of S0: {PSNR_s0}')

aop_pred = aop_pred.cpu().squeeze().numpy()
dolp_pred = dolp_pred.cpu().squeeze().numpy()
s0_pred = s0_pred.cpu().squeeze().numpy()

aop_true = aop_true.cpu().squeeze().numpy()
dolp_true = dolp_true.cpu().squeeze().numpy()
s0_true = s0_true.cpu().squeeze().numpy()

# Plot results
dolp_min, dolp_max = 0, 1
s0_min, s0_max = 0, 1
aop_min, aop_max = 0, np.pi

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im0 = axs[0].imshow(aop_pred, cmap='gray', vmin=aop_min, vmax=aop_max)
axs[0].set_title('AOP')
fig.colorbar(im0, ax=axs[0], orientation='vertical')
im1 = axs[1].imshow(dolp_pred, cmap='gray', vmin=dolp_min, vmax=dolp_max)
axs[1].set_title('DoLP')
fig.colorbar(im1, ax=axs[1], orientation='vertical')
im2 = axs[2].imshow(s0_pred, cmap='gray', vmin=s0_min, vmax=s0_max)
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

