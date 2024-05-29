import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
from utils import  dolp, aop, normalize, view_bar
import h5py
import cv2
import math
import torch
import torchvision.transforms.functional as F
import os


def load_images(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.bmp'):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

            if image is not None:
                try:
                    angle = int(filename.split('_')[1].split('.')[0])
                except ValueError:
                    try:
                        angle = int(filename.split('_')[2].split('.')[0])
                    except ValueError:
                        print(f"Invalid angle in filename: {filename}")
                        continue
                images[angle] = torch.from_numpy(image)
    return images


#FIXME
def create_labels(root_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dofp_list = []
    s0_list = []
    dolp_list = []
    aop_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == os.path.basename(root_folder):
                continue
            dir_fullpath = os.path.join(root_folder, dirname)
            images = load_images(dir_fullpath)

            if len(images) == 4 and 0 in images and 45 in images and 90 in images and 135 in images:
                dofp = torch.zeros_like(images[0].to(device))
                img_aop = torch.zeros_like(images[0].to(device))
                img_dolp = torch.zeros_like(images[0].to(device))
                img_s0 = torch.zeros_like(images[0].to(device))
                dofp[0::2, 0::2] = images[0][0::2, 0::2]
                dofp[0::2, 1::2] = images[45][0::2, 1::2]
                dofp[1::2, 0::2] = images[90][1::2, 0::2]
                dofp[1::2, 1::2] = images[135][1::2, 1::2]
                img_aop = aop(images[0], images[45], images[90], images[135])
                img_dolp = dolp(images[0], images[45], images[90], images[135])
                img_s0 = 1/2 * (images[0] + images[45] + images[90] + images[135])
                # print(dofp.shape)
                # print(img_aop.shape)
                dofp_list.append(dofp)
                aop_list.append (img_aop)
                dolp_list.append(img_dolp)
                s0_list.append(img_s0)                        
                    
    dofp_tensor = torch.stack(dofp_list,dim=0).unsqueeze(1)
    s0_tensor = torch.stack(s0_list, dim=0).unsqueeze(1)
    dolp_tensor = torch.stack(dolp_list, dim=0).unsqueeze(1)
    aop_tensor = torch.stack(aop_list, dim=0).unsqueeze(1)
        
    return dofp_tensor, s0_tensor, aop_tensor, dolp_tensor


 
def generate_labels(dofp_tensor, s0_tensor, aop_tensor, dolp_tensor, label_path):
    labels = torch.cat([s0_tensor, dolp_tensor, aop_tensor], dim=1).cpu()
    Y = dofp_tensor.cpu()
    
    with h5py.File(label_path, 'w') as f:
        f.create_dataset('labels', data=labels)
        f.create_dataset('data', data=Y)


root_path = r'D:\VScodeProjects\dataset\OL_DATA'
label_path = r'D:\VScodeProjects\dataset\OL_DATA\labels.h5'
dofp_tensor, s0_tensor, aop_tensor, dolp_tensor = create_labels(root_path)
print(dofp_tensor.shape)
print(dolp_tensor.shape)
generate_labels(dofp_tensor, s0_tensor, aop_tensor, dolp_tensor, label_path)