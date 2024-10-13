import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
import os
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

'''
Calculate the AoP
'''
def aop(x_0, x_45, x_90, x_135, normalization = False):
    AoP = 0.5 * np.arctan2((x_45 - x_135), (x_0 - x_90 + 1e-8))
    AoP = np.mod(AoP, np.pi)
    
    if normalization:
        AoP = AoP / np.pi

    return AoP

'''
Calculate the DoLP
'''
def dolp(x_0, x_45, x_90, x_135, normalization = False):
    Int = 0.5*(x_0 + x_45 + x_90 + x_135)   
    DoLP = np.sqrt(np.square(x_0-x_90) + np.square(x_45-x_135))/(Int+1e-8)
    DoLP[np.where(Int==0)] = 0   #if Int==0, set the DoLP to 0
    if normalization:
        DoLP = normalize(DoLP,0,1)
    
    return DoLP
