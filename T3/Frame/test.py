import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from model import MyDataset, ForkNet, ResNet
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet()
# model = ForkNet()

# Load checkpoint
checkpoint_path = r'D:\VScodeProjects\learnnn\T3\Frame\ckpt\ResNet.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

model.eval()

input_data_path = r'D:\VScodeProjects\learnnn\T3\Frame\test\13_data.pt'
img = torch.load(input_data_path).unsqueeze(0).unsqueeze(0).to(device)

# forward
with torch.no_grad():
    aop, dolp, s0 = model(img)

aop = aop.cpu().squeeze().numpy()
dolp = dolp.cpu().squeeze().numpy()
s0 = s0.cpu().squeeze().numpy()

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(aop, cmap='gray')
axs[0].set_title('AOP')

axs[1].imshow(dolp, cmap='gray')
axs[1].set_title('DoLP')
axs[2].imshow(s0, cmap='gray')
axs[2].set_title('S0')

plt.tight_layout()
plt.show()