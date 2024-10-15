import numpy as np
from utils import  dolp, aop
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
                images[angle] = torch.from_numpy(image).float() / 255.0
    return images


def create_labels(root_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dofp_list = []
    i0_list = []
    i45_list = []
    i90_list = []
    i135_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == os.path.basename(root_folder):
                continue
            dir_fullpath = os.path.join(root_folder, dirname)
            images = load_images(dir_fullpath)

            if len(images) == 4 and 0 in images and 45 in images and 90 in images and 135 in images:
                dofp = torch.zeros_like(images[0].to(device))
                dofp[0::2, 0::2] = images[0][0::2, 0::2]
                dofp[0::2, 1::2] = images[45][0::2, 1::2]
                dofp[1::2, 1::2] = images[90][1::2, 1::2]
                dofp[1::2, 0::2] = images[135][1::2, 0::2]
                img_i0 = images[0]
                img_i45 = images[45]
                img_i90 = images[90]
                img_i135 = images[135]

                dofp_list.append(dofp)
                i0_list.append (img_i0)
                i45_list.append(img_i45)
                i90_list.append(img_i90)
                i135_list.append(img_i135)                     
                    
    dofp_tensor = torch.stack(dofp_list,dim=0).unsqueeze(1)
    i0_tensor = torch.stack(i0_list, dim=0).unsqueeze(1)
    i45_tensor = torch.stack(i45_list, dim=0).unsqueeze(1)
    i90_tensor = torch.stack(i90_list, dim=0).unsqueeze(1)
    i135_tensor = torch.stack(i135_list, dim=0).unsqueeze(1)
        
    return dofp_tensor, i0_tensor, i45_tensor, i90_tensor, i135_tensor


def merge_ptfiles(root_folder, output_file):
    data_list = []
    labels_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            labels_path = os.path.join(dirpath, dirname, "labels")
            if os.path.exists(labels_path):
                i0_path = os.path.join(labels_path, dirname + '_i0.pt')
                i45_path = os.path.join(labels_path, dirname + '_i45.pt')
                i90_path = os.path.join(labels_path, dirname + '_i90.pt')
                i135_path = os.path.join(labels_path, dirname + '_i135.pt')
                
                if os.path.exists(i0_path) and os.path.exists(i45_path) and os.path.exists(i90_path) and os.path.exists(i135_path):
                    try:
                        img_i0 = torch.load(i0_path).numpy()
                        img_i45 = torch.load(i45_path).numpy()
                        img_i90 = torch.load(i90_path).numpy()
                        img_i135 = torch.load(i135_path).numpy()
                    except Exception as e:
                        print(f"Error loading label files in {labels_path}: {e}")
                        continue
                    
                    data_path = os.path.join(dirpath, dirname, "data")
                    if os.path.exists(data_path):
                        data_path = os.path.join(data_path, dirname + '_data.pt')
                        try:
                            data = torch.load(data_path).numpy()
                            data_list.append(data)
                            labels_list.append((img_i0, img_i45, img_i90, img_i135))
                        except Exception as e:
                            print(f"Error loading data file in {data_path}: {e}")

    with h5py.File(output_file, 'w') as hf:
        data_group = hf.create_group('data')
        labels_group = hf.create_group('labels')

        for i, data in enumerate(data_list):
            data_group.create_dataset(f'data_{i}', data=data)
        
        for i, (i0, i45, i90, i135) in enumerate(labels_list):
            label_group = labels_group.create_group(f'label_{i}')
            label_group.create_dataset('i0', data=i0)
            label_group.create_dataset('i45', data=i45)
            label_group.create_dataset('i90', data=i90)
            label_group.create_dataset('i135', data=i135)
    
    print(f'Merged data saved to {output_file}')


# Generate patches
def slice_and_save_to_h5(dofp_tensor, i0_tensor, i45_tensor, i90_tensor, i135_tensor, label_path, patch_size, stride, slice=True):
    data_patches = []
    i0_patches = []
    i45_patches = []
    i90_patches = []
    i135_patches = []
    data_list = []
    labels_list = []
    if slice:
        for i in range(dofp_tensor.shape[0]):
            dofp = dofp_tensor[i]
            i0 = i0_tensor[i]
            i45 = i45_tensor[i]
            i90 = i90_tensor[i]
            i135 = i135_tensor[i]
            
            # print(dofp.shape)
            for y in range(0, dofp.shape[1] - patch_size + 1, stride):
                for x in range(0, dofp.shape[2] - patch_size + 1, stride):
                    data_patches.append(dofp[:, y:y+patch_size, x:x+patch_size])
                    i0_patches.append(i0[:, y:y+patch_size, x:x+patch_size])
                    i45_patches.append(i45[:, y:y+patch_size, x:x+patch_size])
                    i90_patches.append(i90[:, y:y+patch_size, x:x+patch_size])
                    i135_patches.append(i135[:, y:y+patch_size, x:x+patch_size])
            
            # Process edge region
            if y + patch_size < dofp.shape[1]:
                data_patches.append(dofp[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                i0_patches.append(i0[:, dofp.shape[1] - patch_size:, x:x+patch_size])               
                i45_patches.append(i45[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                i90_patches.append(i90[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                i135_patches.append(i135[:, dofp.shape[1] - patch_size:, x:x+patch_size])

            if x + patch_size < dofp.shape[2]:
                data_patches.append(dofp[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                i0_patches.append(i0[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                i45_patches.append(i45[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                i90_patches.append(i90[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                i135_patches.append(i135[:, y:y+patch_size, dofp.shape[2] - patch_size:])
        
        data_patches = torch.stack(data_patches)
        i0_patches = torch.stack(i0_patches)
        i45_patches = torch.stack(i45_patches)
        i90_patches = torch.stack(i90_patches)
        i135_patches = torch.stack(i135_patches)
        
        data_list.append(data_patches)
        labels_list.append((i0_patches, i45_patches, i90_patches, i135_patches))
    
    else:
        data_patches = dofp_tensor
        i0_patches = i0_tensor
        i45_patches = i45_tensor
        i90_patches = i90_tensor
        i135_patches = i135_tensor
        
        data_list.append(data_patches)
        labels_list.append((i0_patches, i45_patches, i90_patches, i135_patches))
            
    with h5py.File(label_path, 'w') as hf:
        data_group = hf.create_group('data')
        labels_group = hf.create_group('labels')

        for i, data in enumerate(data_patches):
            data_group.create_dataset(f'data_{i}', data=data.squeeze(0).cpu().numpy())
        
        for i, (i0, i45, i90, i135) in enumerate(zip(i0_patches, i45_patches, i90_patches, i135_patches)):
            label_group = labels_group.create_group(f'label_{i}')
            label_group.create_dataset('i0', data=i0.squeeze(0).cpu().numpy())
            label_group.create_dataset('i45', data=i45.squeeze(0).cpu().numpy())
            label_group.create_dataset('i90', data=i90.squeeze(0).cpu().numpy())
            label_group.create_dataset('i135', data=i135.squeeze(0).cpu().numpy())
            
    print('Finished!')
            
            
root_path = r'D:\WORKS\dataset\data_test\Fork_test'
label_path = r'D:\WORKS\dataset\patches\sr_test\Fork_sr_test.h5'
dofp_tensor, i0_tensor, i45_tensor, i90_tensor, i135_tensor = create_labels(root_path)
patch_size = 100
coincide = 10
stride = patch_size - coincide
slice_and_save_to_h5(dofp_tensor, i0_tensor, i45_tensor, i90_tensor, i135_tensor, label_path, patch_size, stride, slice=True)
