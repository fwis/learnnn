import os
import cv2
import numpy as np
import shutil
from PIL import Image
import math
import numpy as np
import torch
import h5py


def createDofp(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == os.path.basename(root_folder):
                continue
            dir_fullpath = os.path.join(root_folder, dirname)
            dofp_path = os.path.join(dir_fullpath, "data")
            os.makedirs(dofp_path, exist_ok=True)
            dofp = None

            for filename in os.listdir(dir_fullpath):
                if filename.endswith('.png') or filename.endswith('.bmp'):
                    image_path = os.path.join(dir_fullpath, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

                    if dofp is None:
                        dofp = np.zeros_like(image)
                        dofp.shape = image.shape

                    # 从文件名中提取角度信息
                    try:
                        angle = int(filename.split('_')[1].split('.')[0])  # 假设文件名格式为"abc_0.png"、"xyz_45.png"等
                    except ValueError:
                        try:
                            angle = int(filename.split('_')[2].split('.')[0])  # 处理文件名格式为"abc_abc_0"等
                        except ValueError:
                            print(f"Invalid angle in filename: {filename}")
                            continue

                    if angle == 0:
                        dofp[0::2, 0::2] = image[0::2, 0::2]
                    elif angle == 45:
                        dofp[0::2, 1::2] = image[0::2, 1::2]
                    elif angle == 90:
                        dofp[1::2, 0::2] = image[1::2, 0::2]
                    elif angle == 135:
                        dofp[1::2, 1::2] = image[1::2, 1::2]
                        
            dofp = dofp.astype(np.float32)/255
            dofp= torch.from_numpy(dofp)
            dofp_path = os.path.join(dofp_path, dirname + '_data.pt')
            torch.save(dofp,dofp_path)


def create_labels(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == os.path.basename(root_folder):
                continue
            dir_fullpath = os.path.join(root_folder, dirname)
            labels_path = os.path.join(dir_fullpath, "labels")
            os.makedirs(labels_path, exist_ok=True)
            images=[]
            for filename in os.listdir(dir_fullpath):
                if filename.endswith('.png') or filename.endswith('.bmp'):
                    image_path = os.path.join(dir_fullpath, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

                    # 从文件名中提取角度信息
                    try:
                        angle = int(filename.split('_')[1].split('.')[0])  # 假设文件名格式为"abc_0.png"、"xyz_45.png"等
                    except ValueError:
                        print(f"Invalid angle in filename: {filename}")
                        continue
                    
                    if angle == 0:
                        image0 = image.astype(np.float32)/255
                        images.append(image0)
                    elif angle == 135:
                        image135 = image.astype(np.float32)/255
                        images.append(image135)
                    elif angle == 45:
                        image45 = image.astype(np.float32)/255
                        images.append(image45)
                    elif angle == 90:
                        image90 = image.astype(np.float32)/255
                        images.append(image90)
                    if len(images)>= 4:
                        img_aop = aop(images[0], images[2], images[3], images[1], normalization = False)
                        img_dolp = dolp(images[0], images[2], images[3], images[1], normalization = False)
                        img_dolp[img_dolp > 1] = 1
                        img_s0 = 1/2 * (images[0] + images[1] + images[2] + images[3])
                        img_aop = torch.from_numpy(img_aop)
                        img_dolp = torch.from_numpy(img_dolp)
                        img_s0 = torch.from_numpy(img_s0)
                        
                        aop_path = os.path.join(labels_path, dirname + '_aop.pt')
                        dolp_path = os.path.join(labels_path, dirname + '_dolp.pt')
                        s0_path = os.path.join(labels_path, dirname + '_s0.pt')
                        torch.save(img_aop, aop_path)
                        torch.save(img_dolp, dolp_path)
                        torch.save(img_s0, s0_path)


def merge_ptfiles(root_folder, output_file):
    data_list = []
    labels_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            labels_path = os.path.join(dirpath, dirname, "labels")
            if os.path.exists(labels_path):
                aop_path = os.path.join(labels_path, dirname + '_aop.pt')
                dolp_path = os.path.join(labels_path, dirname + '_dolp.pt')
                s0_path = os.path.join(labels_path, dirname + '_s0.pt')
                
                if os.path.exists(aop_path) and os.path.exists(dolp_path) and os.path.exists(s0_path):
                    try:
                        img_aop = torch.load(aop_path).numpy()
                        img_dolp = torch.load(dolp_path).numpy()
                        img_s0 = torch.load(s0_path).numpy()
                    except Exception as e:
                        print(f"Error loading label files in {labels_path}: {e}")
                        continue
                    
                    data_path = os.path.join(dirpath, dirname, "data")
                    if os.path.exists(data_path):
                        data_path = os.path.join(data_path, dirname + '_data.pt')
                        try:
                            data = torch.load(data_path).numpy()
                            data_list.append(data)
                            labels_list.append((img_aop, img_dolp, img_s0))
                        except Exception as e:
                            print(f"Error loading data file in {data_path}: {e}")

    with h5py.File(output_file, 'w') as hf:
        data_group = hf.create_group('data')
        labels_group = hf.create_group('labels')

        for i, data in enumerate(data_list):
            data_group.create_dataset(f'data_{i}', data=data)
        
        for i, (aop, dolp, s0) in enumerate(labels_list):
            label_group = labels_group.create_group(f'label_{i}')
            label_group.create_dataset('aop', data=aop)
            label_group.create_dataset('dolp', data=dolp)
            label_group.create_dataset('s0', data=s0)
    
    print(f'Merged data saved to {output_file}')


def slice_and_save_to_h5(dofp_tensor, s0_tensor, aop_tensor, dolp_tensor, label_path, patch_size, stride):
    data_patches = []
    s0_patches = []
    aop_patches = []
    dolp_patches = []

    for i in range(dofp_tensor.shape[0]):
        dofp = dofp_tensor[i]
        s0 = s0_tensor[i]
        aop = aop_tensor[i]
        dolp = dolp_tensor[i]

        for y in range(0, dofp.shape[2] - patch_size + 1, stride):
            for x in range(0, dofp.shape[3] - patch_size + 1, stride):
                data_patches.append(dofp[:, y:y+patch_size, x:x+patch_size])
                s0_patches.append(s0[:, y:y+patch_size, x:x+patch_size])
                aop_patches.append(aop[:, y:y+patch_size, x:x+patch_size])
                dolp_patches.append(dolp[:, y:y+patch_size, x:x+patch_size])

    data_patches = torch.stack(data_patches)
    s0_patches = torch.stack(s0_patches)
    aop_patches = torch.stack(aop_patches)
    dolp_patches = torch.stack(dolp_patches)

    with h5py.File(label_path, 'w') as f:
        f.create_dataset('data', data=data_patches.cpu().numpy())
        f.create_dataset('s0', data=s0_patches.cpu().numpy())
        f.create_dataset('aop', data=aop_patches.cpu().numpy())
        f.create_dataset('dolp', data=dolp_patches.cpu().numpy())


def normalize(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
        norm_data = np.zeros(data.shape)
    else:  
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data


def aop(x_0, x_45, x_90, x_135, normalization = False):
    '''
    Calculate the AoP
    '''
    AoP = 0.5 * np.arctan((x_45 - x_135) / (x_0 - x_90 + 1e-8)) + math.pi/4.
    if normalization:
        AoP = normalize(AoP,0,1)

    return AoP


def dolp(x_0, x_45, x_90, x_135, normalization = False):
    '''
    Calculate the DoLP
    '''
    Int = 0.5*(x_0 + x_45 + x_90 + x_135)   
    DoLP = np.sqrt(np.square(x_0-x_90) + np.square(x_45-x_135))/(Int+1e-8)
    DoLP[np.where(Int==0)] = 0   #if Int==0, set the DoLP to 0
    if normalization:
        DoLP = normalize(DoLP,0,1)
    
    return DoLP


'''
下面是一些其他处理
'''
def convert_folder_to_grayscale(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(file_path, gray_img)


def crop_img(img_path):
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                if img is None:
                    print(f"Could not read {file}. Skipping.")
                    continue
                
                # 从中心剪切图片
                h, w = img.shape[:2]
                y, x = 1024,760
                img = img[h//2-y//2:h//2+y//2-1, w//2-x//2:w//2+x//2-1]
                cv2.imwrite(img_path, img)
                print(f"Cropped {file} to {w} * {h}.")


def rotate_images(input_folder, output_folder):
    output_folder_rotated90 = os.path.join(output_folder, 'rotated90')
    output_folder_rotated180 = os.path.join(output_folder, 'rotated180')
    output_folder_rotated270 = os.path.join(output_folder, 'rotated270')
    os.makedirs(output_folder_rotated90, exist_ok=True)
    os.makedirs(output_folder_rotated180, exist_ok=True)
    os.makedirs(output_folder_rotated270, exist_ok=True)

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 在输出文件夹中创建对应的子文件夹
        output_subfolder_rotated90 = os.path.join(output_folder_rotated90, folder_name)
        output_subfolder_rotated180 = os.path.join(output_folder_rotated180, folder_name)
        output_subfolder_rotated270 = os.path.join(output_folder_rotated270, folder_name)
        os.makedirs(output_subfolder_rotated90, exist_ok=True)
        os.makedirs(output_subfolder_rotated180, exist_ok=True)
        os.makedirs(output_subfolder_rotated270, exist_ok=True)

        for file_name in os.listdir(folder_path):
            input_image_path = os.path.join(folder_path, file_name)

            img = cv2.imread(input_image_path)
            if img is None:
                print(f"无法读取图像: {input_image_path}")
                continue

            height, width = img.shape[:2]

            img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(output_subfolder_rotated90, file_name), img_rotated_90)

            img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(output_subfolder_rotated180, file_name), img_rotated_180)

            img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(output_subfolder_rotated270, file_name), img_rotated_270)


def rename_and_move_images(base_folders, output_folder='sorted_images'):
    """
    将图片重命名为camera序号_角度.png格式，并移动到新的分类文件夹中。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for folder in base_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                # 提取标号部分
                parts = filename.split('_')
                if len(parts) >= 3:
                    camera_part = parts[0]  # camera03
                    time_part = parts[1]  # 11-07-48
                    angle_part = os.path.splitext(parts[2])[0]  # 0 (removing .png)

                    # 提取序号后缀
                    time_suffix = time_part.split('-')[2]
                    new_filename = f'camera{time_suffix}_{angle_part}.png'
                    
                    identifier = f'camera{time_suffix}'  # camera48
                    target_folder = os.path.join(output_folder, identifier)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    # 移动并重命名文件
                    source_path = os.path.join(folder, filename)
                    target_path = os.path.join(target_folder, new_filename)
                    shutil.move(source_path, target_path)
                else:
                    print(f"文件名格式不正确: {filename}")

    print("图片重命名和移动完成。")


def organize_images_by_identifier(base_folder, output_folder='sorted_images'):
    """
    将同一标号但不同角度的图片从指定文件夹中移动到一个新的分类文件夹中。
    :param base_folder: 包含图片的基础文件夹路径
    :param output_folder: 分类后图片的输出文件夹
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(base_folder):
        if filename.endswith('.bmp'):
            # 提取标号部分
            parts = filename.split('_')
            if len(parts) > 2:
                identifier = parts[0] + '_' + parts[1]
                target_folder = os.path.join(output_folder, identifier)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                # 移动文件到目标文件夹
                source_path = os.path.join(base_folder, filename)
                target_path = os.path.join(target_folder, filename)
                try:
                    shutil.move(source_path, target_path)
                    print(f"已移动文件: {source_path} 到 {target_path}")
                except Exception as e:
                    print(f"移动文件 {filename} 时出错: {e}")

    print("图片分类完成。")


def move_files_to_parent(source_parent_folder):
    for i in range(1,106):
        source_folder = os.path.join(source_parent_folder, str(i))
        sub_directories = [os.path.join(source_folder, d) for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
        print(sub_directories)
        for sub_directory in sub_directories:
            files = [os.path.join(sub_directory, f) for f in os.listdir(sub_directory) if os.path.isfile(os.path.join(sub_directory, f))]       
            # 移动文件到上一级文件夹
            for file in files:
                shutil.move(file, source_folder)

        for sub_directory in sub_directories:
            os.rmdir(sub_directory)


def remove_input(source_parent_folder):
    for i in range(1,106):
        source_folder = os.path.join(source_parent_folder, str(i))
        sub_directories = [os.path.join(source_parent_folder, d) for d in os.listdir(source_parent_folder) if os.path.isdir(os.path.join(source_parent_folder, d))]
        for sub_directory in sub_directories:
            files = [os.path.join(sub_directory, f) for f in os.listdir(sub_directory) if os.path.isfile(os.path.join(sub_directory, f))]
            for file in files:
                if file.endswith('net_input.png'):
                    os.remove(file)
       
                    
def resize_images(img_path, scale_factor=0.5):
    """
    对文件夹内的所有图像进行缩放。
    """
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_img.save(img_path)
                    print(f"图像 {file} 已按比例 {scale_factor} 缩放。")
                except Exception as e:
                    print(f"处理图像 {file} 时出错：{e}")


def img_blur(img_path,kernelsize = (5,5) ,sigma = 0):
     for root, _, files in os.walk(img_path):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                try:
                    img = cv2.imread(img_path)
                    blur_img = cv2.GaussianBlur(img, kernelsize, sigma)
                    cv2.imwrite(img_path,blur_img)
                    print(f"图像 {file} 已做高斯模糊。")
                except Exception as e:
                    print(f"处理图像 {file} 时出错：{e}")   


def rename_fork(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                i,j = filename.split('image')[1].split('_')
                new_filename = f'{i}_{j}.png'
                old_filepath = os.path.join(foldername, filename)
                new_filepath = os.path.join(foldername, new_filename)
                os.rename(old_filepath, new_filepath)

        for subfolder in subfolders:
            i = subfolder.split('_')[1]
            new_subfolder = f'{i}'
            old_subfolder_path = os.path.join(foldername, subfolder)
            new_subfolder_path = os.path.join(foldername, new_subfolder)
            os.rename(old_subfolder_path, new_subfolder_path)


def rename_OL(root_dir):          
    for root, dirs, files in os.walk(root_dir):
        for file in files:

            if "_" in file:
                parts = file.split("_")
                # if len(parts) > 2 :
                new_file_name = os.path.basename(root) + "_" + parts[-1]
                new_file_path = os.path.join(root, new_file_name)
                os.rename(os.path.join(root, file), new_file_path)


def rename_PIF(root_dir):
    sub_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    for folder in sub_folders:
        folder_name = os.path.basename(folder)       
        new_folder_name = folder_name.replace('camera', '')
        new_folder_path = os.path.join(root_dir, new_folder_name)
        os.rename(folder, new_folder_path)
        files = [f for f in os.listdir(new_folder_path) if os.path.isfile(os.path.join(new_folder_path, f))]
        
        for file in files:
            new_file_name = file.replace('camera', '')
            original_file_path = os.path.join(new_folder_path, file)
            new_file_path = os.path.join(new_folder_path, new_file_name)
            os.rename(original_file_path, new_file_path)

    print("重命名完成！")


def sort_folders(root_dir):
    sub_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    sub_folders.sort()
    for index, folder in enumerate(sub_folders, start=1):
        folder_name = os.path.basename(folder)
        new_folder_name = str(index)
        new_folder_path = os.path.join(root_dir, new_folder_name)
        os.rename(folder, new_folder_path)

        files = [f for f in os.listdir(new_folder_path) if os.path.isfile(os.path.join(new_folder_path, f))]
        files.sort()

    print("文件夹排序完成！")

def rename_tokyo(root_folder):
    # 遍历根文件夹内的所有子文件夹
    for folder_name in os.listdir(root_folder):
        # 获取子文件夹的完整路径
        folder_path = os.path.join(root_folder, folder_name)
        # 检查路径是否为文件夹
        if os.path.isdir(folder_path):
            print(f"正在处理子文件夹: {folder_name}")
            # 遍历子文件夹内的所有文件
            for filename in os.listdir(folder_path):
                # 检查文件是否为PNG文件
                if filename.endswith(".png"):
                    # 构建新的文件名
                    new_filename = f"{folder_name}_{filename}"
                    # 构建旧文件路径和新文件路径
                    old_path = os.path.join(folder_path, filename)
                    new_path = os.path.join(folder_path, new_filename)
                    # 重命名文件
                    os.rename(old_path, new_path)
                    print(f"已将文件 {filename} 重命名为 {new_filename}")
                    
                    
def remove_labes(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir, topdown=False):
        for subfolder in subfolders:
            if subfolder == 'data' or subfolder == 'labels':
                subfolder_path = os.path.join(foldername, subfolder)
                for root, dirs, files in os.walk(subfolder_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                os.rmdir(subfolder_path)
                
def sort_folders(root_dir, target_dir):
    sub_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    sub_folders.sort()
    
    for index, folder in enumerate(sub_folders, start=1):
        folder_name = os.path.basename(folder)
        new_folder_name = str(index)
        new_folder_path = os.path.join(target_dir, new_folder_name)
        
        # 创建新的目标文件夹
        os.makedirs(new_folder_path, exist_ok=True)
        
        # 移动文件到新文件夹
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            shutil.move(os.path.join(folder, file), os.path.join(new_folder_path, file))
        
    print("文件夹排序完成！")

# 使用示例
sort_folders(r"D:\WORKS\dataset\data_test\tokyo_test", r"D:\WORKS\dataset\data_test\sort")

                
# root_dir = r'D:\WORKS\dataset\sorted_images'
# output_file = r'D:\WORKS\Polarization\Machine_Learning\Tokyo_dataset\data.h5'
# rename_fork(root_dir)
# rename_img(root_dir)
# rename_PIF(root_dir)
# create_labels(root_dir)
# createDofp(root_dir)
# merge_ptfiles(root_dir, output_file)
# rename_OL(root_dir)
# remove_labes(root_dir)

# source_parent_folder = r'C:\Users\lhr\Desktop\OL_DATA'
# move_files_to_parent(source_parent_folder)

# source_parent_folder = r'C:\Users\lhr\Desktop\OL_DATA'
# 获取最低一级子文件夹

# base_folder = r"D:\WORKS\Polarization\test_set"
# organize_images_by_identifier(base_folder)

# folders = [r'C:\Users\lhr\Desktop\PIF_dataset\0', r'C:\Users\lhr\Desktop\PIF_dataset\45', r'C:\Users\lhr\Desktop\PIF_dataset\90', r'C:\Users\lhr\Desktop\PIF_dataset\135']
# rename_and_move_images(folders)

# folder_path = r"C:\Users\lhr\Desktop\OL_DATA"
# convert_folder_to_grayscale(folder_path)

# base_dir = r"C:\Users\lhr\Desktop\dataset\Polarization Image Dataset"
# # 遍历0-105的数字
# for i in range(111):
#     folder_path = os.path.join(base_dir,'image_' + str(i), "net_input")
#     # 检查文件夹是否存在
#     if os.path.exists(folder_path):
#         # 如果存在，则删除文件夹及其内容
#         print("Deleting folder:", folder_path)
#         os.system("rmdir /s /q " + folder_path)
