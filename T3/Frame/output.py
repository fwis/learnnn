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
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = ResNetGenerator()
model = ForkNet()

# Load checkpoint
checkpoint_path = 'T3/Frame/ckpt/ForkNet_Fork2.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load and normalize the image data to [0, 1]
input_image_path = r'T3\Frame\test\dofp.png'
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
img = img / 255.0  # Normalize to [0, 1]
img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    aop, dolp, s0 = model(img)

# Move outputs to CPU
aop = aop.cpu()
dolp = dolp.cpu()
s0 = s0.cpu()

# Create output directory if it does not exist
output_dir = 'T3/Frame/output/'
os.makedirs(output_dir, exist_ok=True)

# Convert and save 'aop' as 8-bit PNG
aop_np = aop.squeeze().numpy()  # Remove extra dimensions
aop_np = np.clip(aop_np * 255, 0, 255).astype(np.uint8)  # Scale and convert to 8-bit
cv2.imwrite(os.path.join(output_dir, 'aop.png'), aop_np)

# Convert and save 'dolp' as 8-bit PNG
dolp_np = dolp.squeeze().numpy()
dolp_np = np.clip(dolp_np * 255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 'dolp.png'), dolp_np)

# Convert and save 's0' as 8-bit PNG
s0_np = s0.squeeze().numpy()
s0_np = np.clip(s0_np * 255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 's0.png'), s0_np)

print('finished')