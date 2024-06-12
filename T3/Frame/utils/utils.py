import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
import os
import imageio as imgio
import torch
import math
import cv2
import torchmetrics as metrics

def view_bar(step, total):
    num = step + 1
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    arrow = 0 if num==total else 1
    r = '\r[%s%s%s]%d%%' % ('■'*rate_num, '▶'*arrow, '-'*(100-rate_num-arrow), rate * 100)
    sys.stdout.write(r)
    sys.stdout.flush()


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


def psnr(ground_truth, ref, mx):
    '''
    Calculate PSNR
    '''
    diff = ref - ground_truth   
    diff = diff.flatten('C')
    rmse = np.sqrt(np.mean(diff ** 2.))
    PSNR = 20 * np.log10(mx / rmse)

    return PSNR


def ssim(ground_truth, ref):
    '''
    Calculate SSIM
    '''
    ssim = metrics.SSIM(data_range=1.0, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03, eps=1e-8, reduction='mean')
    ssim = ssim(ground_truth, ref)
    return ssim


def count_para(model):
    total_params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', total_params)

