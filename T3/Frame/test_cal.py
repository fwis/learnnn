import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from cal_model import ResNet
import warnings
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet()

# Load checkpoint
checkpoint_path = 'T3\Frame\ckpt\ResNet_cal_1_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

input_data_path = r'.\T3\Frame\test\83_data.pt'
img = torch.load(input_data_path).unsqueeze(0).unsqueeze(0).to(device)

# forward
with torch.no_grad():
    i0, i45, i90, i135 = model(img)

i0 = i0.cpu().squeeze().numpy()
i45 = i45.cpu().squeeze().numpy()
i90 = i90.cpu().squeeze().numpy()
i135 = i135.cpu().squeeze().numpy()


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
