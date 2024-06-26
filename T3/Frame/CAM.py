import torch
from torch import nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from model import ConvNeXtNet, ResNet
import matplotlib.pyplot as plt

# Multi-task to Single-task
class single_task_model_wrapper(nn.Module):
    def __init__(self, model, task_index):
        super(single_task_model_wrapper, self).__init__()
        self.model = model
        self.task_index = task_index

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.task_index]

class image_target:
    def __init__(self, original_image):
        self.original_image = original_image

    def __call__(self, model_output):
        # Calculate MSE
        loss = ((model_output - self.original_image) ** 2).mean()
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNeXtNet()
checkpoint_path = 'T3/Frame/ckpt/ConvNeXtNet1_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)
task_index = 1 # [0, 1, 2] = aop, dolp, s0
single_task_model = single_task_model_wrapper(model, task_index)

# Set target layer
target_layer = single_task_model.model.layer3_2

# Create GradCAM instance
cam = GradCAM(model=single_task_model, target_layers=[target_layer])

# Input image
input_tensor = torch.load(r'T3\Frame\test\7_data.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
input_tensor = input_tensor[200:1000,200:1000].unsqueeze(0).unsqueeze(0)
original_image = input_tensor.clone()

# Generate CAM heatmap
targets = [image_target(original_image)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
grayscale_cam = cv2.normalize(grayscale_cam, None, 0, 1, cv2.NORM_MINMAX)

# Visualize CAM heatmap
image = input_tensor[0].permute(1, 2, 0).cpu().numpy()
bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
bgr_image = cv2.normalize(bgr_image, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)

# Generate visualization image
visualization = show_cam_on_image(bgr_image, grayscale_cam, image_weight=0.6)
torch.cuda.empty_cache()

# Show image
cv2.namedWindow('Grad-CAM', cv2.WINDOW_NORMAL)
cv2.imshow('Grad-CAM', visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()