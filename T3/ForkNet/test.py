import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import ForkNet  # Ensure you have defined ForkNet in model.py
from utils.utils import dolp, psnr, normalize, view_bar, aop, pad_shift, count_para, plot_feature_map, fig2array
import imageio as imgio
import os
import math
import time
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim

# 超参数
IMG_NUM = 10
IMG_WIDTH = 1280
IMG_HEIGHT = 960
output_images = []
vis_feature_map = False
hot_map = True
plot_dir = './images/feature_maps/'
test_img_path = './data/test_set'
model_path = './best_model/model_1/model_1.pth'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 创建结果存储矩阵
bic_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
origin_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
msc_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 1], np.float32)
bic_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)

# 降采样参数
m_1 = np.arange(0, 959, 2)
n_1 = np.arange(0, 1279, 2)
m_2 = np.arange(1, 960, 2)
n_2 = np.arange(1, 1280, 2)
i_0, j_0 = np.meshgrid(m_1, n_1, indexing='ij')
i_45, j_45 = np.meshgrid(m_1, n_2, indexing='ij')
i_90, j_90 = np.meshgrid(m_2, n_2, indexing='ij')
i_135, j_135 = np.meshgrid(m_2, n_1, indexing='ij')
ds_index = [(i_0, j_0), (i_45, j_45), (i_90, j_90), (i_135, j_135)]

# 初始化模型
model = ForkNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

count_para()

total_S0_PSNR = np.zeros((IMG_NUM))
total_S0_PSNR_BIC = np.zeros((IMG_NUM))
total_DoLP_PSNR = np.zeros((IMG_NUM))
total_DoLP_PSNR_BIC = np.zeros((IMG_NUM))
total_AoP_PSNR = np.zeros((IMG_NUM))
total_AoP_PSNR_BIC = np.zeros((IMG_NUM))
total_time = 0

with torch.no_grad():
    for i in range(IMG_NUM):
        for j in range(4):
            path_origin = os.path.join(test_img_path, f'image_{i + 1}_{j * 45}.bmp')
            img = np.array(Image.open(path_origin), np.float32) / 255.
            msc_img[i, ds_index[j][0], ds_index[j][1], 0] = img[ds_index[j]]
            bic_img[i, :, :, j] = cv2.resize(img[ds_index[j]], (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_CUBIC)
            origin_img[i, :, :, j] = img

        bic_img[i] = pad_shift(bic_img[i])

        start = time.time()
        input_tensor = torch.tensor(msc_img[i:i+1]).permute(0, 3, 1, 2).to(device)
        S0_hat_test, DoLP_hat_test, AoP_hat_test = model(input_tensor)
        end = time.time()
        total_time += end - start

        # 把输出放到cpu上处理
        S0_hat_test = S0_hat_test.squeeze().cpu().numpy()
        DoLP_hat_test = DoLP_hat_test.squeeze().cpu().numpy()
        AoP_hat_test = AoP_hat_test.squeeze().cpu().numpy()

        S0_hat_test = np.clip(S0_hat_test, 0, 2)
        DoLP_hat_test = np.clip(DoLP_hat_test, 0, 1)
        AoP_hat_test = np.clip(AoP_hat_test, 0, math.pi/2)

        pred_s0[i] = S0_hat_test
        pred_dolp[i] = DoLP_hat_test
        pred_aop[i] = AoP_hat_test

        # 计算ground_truth
        S0_true = 0.5 * (origin_img[i, :, :, 0] + origin_img[i, :, :, 1] + origin_img[i, :, :, 2] + origin_img[i, :, :, 3])
        DoLP_true = dolp(origin_img[i, :, :, 0], origin_img[i, :, :, 1], origin_img[i, :, :, 2], origin_img[i, :, :, 3])
        AoP_true = aop(origin_img[i, :, :, 0], origin_img[i, :, :, 1], origin_img[i, :, :, 2], origin_img[i, :, :, 3]) + math.pi/4.
        origin_s0[i] = S0_true
        origin_dolp[i] = DoLP_true
        origin_aop[i] = AoP_true

        # 计算bicubic
        S0_BIC = 0.5 * (bic_img[i, :, :, 0] + bic_img[i, :, :, 1] + bic_img[i, :, :, 2] + bic_img[i, :, :, 3])
        DoLP_BIC = dolp(bic_img[i, :, :, 0], bic_img[i, :, :, 1], bic_img[i, :, :, 2], bic_img[i, :, :, 3])
        AoP_BIC = aop(bic_img[i, :, :, 0], bic_img[i, :, :, 1], bic_img[i, :, :, 2], bic_img[i, :, :, 3]) + math.pi / 4.
        bic_s0[i] = S0_BIC
        bic_dolp[i] = DoLP_BIC
        bic_aop[i] = AoP_BIC

        # 计算PSNR
        total_S0_PSNR[i] = psnr(S0_true, S0_hat_test, 2)
        total_DoLP_PSNR[i] = psnr(DoLP_true, DoLP_hat_test, 1)
        total_AoP_PSNR[i] = psnr(AoP_true, AoP_hat_test, math.pi / 2)

        total_S0_PSNR_BIC[i] = psnr(S0_true, S0_BIC, 2)
        total_DoLP_PSNR_BIC[i] = psnr(DoLP_true, DoLP_BIC, 1)
        total_AoP_PSNR_BIC[i] = psnr(AoP_true, AoP_BIC, math.pi / 2)

        # Show progress
        view_bar(i, IMG_NUM)

    print('\n========================================Testing=======================================' +
          '\n ————————————————————————————————————————————————————————————————————————————————' +
          '\n| PSNR of S_0 using ForkNet: %.5f    |   PSNR of S_0 using BICUBIC: %.5f   |' % (np.mean(total_S0_PSNR), np.mean(total_S0_PSNR_BIC)) +
          '\n| PSNR of DoLP using ForkNet: %.5f   |   PSNR of DoLP using BICUBIC: %.5f  |' % (np.mean(total_DoLP_PSNR), np.mean(total_DoLP_PSNR_BIC)) +
          '\n| PSNR of AoP using ForkNet: %.5f    |   PSNR of AoP using BICUBIC: %.5f   |' % (np.mean(total_AoP_PSNR), np.mean(total_AoP_PSNR_BIC)) +
          '\n ————————————————————————————————————————————————————————————————————————————————')

    print('\nForkNet time: {} sec'.format(total_time / IMG_NUM))

    for j in output_images:
        imgio.imsave(f"./images/bic_s0_{j}_{total_S0_PSNR_BIC[j-1]}.jpg", bic_s0[j-1, 390:-470, 690:-490])
        imgio.imsave(f"./images/pred_s0_3_path_{j}_{total_S0_PSNR[j-1]}.jpg", pred_s0[j-1, 200:-80, 325:-275])
        imgio.imsave(f"./images/org_s0_{j}.jpg", origin_s0[j-1, 390:-470, 690:-490])

        imgio.imsave(f"./images/bic_dolp_{j}_{total_DoLP_PSNR_BIC[j-1]}.jpg", bic_dolp[j-1, 285:-575, 865:-315])
        imgio.imsave(f"./images/pred_dolp_3_path_{j}_{total_DoLP_PSNR[j-1]}.jpg", pred_dolp[j-1, 200:-80, 325:-275])
        imgio.imsave(f"./images/org_dolp_{j}.jpg", origin_dolp[j-1, 285:-575, 865:-315])

        imgio.imsave(f"./images/bic_aop_{j}_{total_AoP_PSNR_BIC[j-1]}.jpg", bic_aop[j-1, 200:-80, 325:-275])
        imgio.imsave(f"./images/pred_aop_3_path_{j}_{total_AoP_PSNR[j-1]}.jpg", pred_aop[j-1, 200:-80, 325:-275])
        imgio.imsave(f"./images/org_aop_{j}.jpg", origin_aop[j-1, 200:-80, 325:-275])

        if hot_map:
            plt.axis('off')
            fig = plt.figure()
            fig = plt.gcf()
            height, width = bic_aop[j-1, 200:-80, 325:-275].shape
            fig.set_size_inches(width/300, height/300)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.imshow(bic_aop[j-1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/bic_aop.jpg', dpi=300)

            plt.imshow(pred_aop[j-1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/pred_aop_0.05_0.0180_log.jpg', dpi=300)

            plt.imshow(origin_aop[j-1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/org_aop.jpg', dpi=300)

        if vis_feature_map:
            visualize_layers = ['x_1', 'x_2', 'x_3_1', 'x_3_2', 'x_3_3']
            for name, layer in model.named_children():
                if name in visualize_layers:
                    layer_output = layer(input_tensor).squeeze().cpu().numpy()
                    if not os.path.exists(os.path.join(plot_dir, name)):
                        os.mkdir(os.path.join(plot_dir, name))
                    plot_feature_map(layer_output, os.path.join(plot_dir, name), maps_all=True)
