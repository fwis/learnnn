import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ForkNet(nn.Module):
    def __init__(self, padding='valid'):
        super(ForkNet, self).__init__()
        if padding.lower() == 'valid':
            self.padding = 0
        else:
            self.padding = 1

        self.conv1 = nn.Conv2d(4, 96, kernel_size=4, padding=self.padding)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=self.padding)
        
        self.conv3_1 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=5, padding=self.padding)
        
        self.conv3_2 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_2 = nn.Conv2d(32, 1, kernel_size=5, padding=self.padding)
        
        self.conv3_3 = nn.Conv2d(48, 32, kernel_size=3, padding=self.padding)
        self.conv4_3 = nn.Conv2d(32, 1, kernel_size=4, padding=self.padding)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        
        x3_1 = F.relu(self.conv3_1(x2))
        s0 = self.conv4_1(x3_1)
        
        x3_2 = F.relu(self.conv3_2(x2))
        dolp = self.conv4_2(x3_2)
        
        x3_3 = F.relu(self.conv3_3(x2))
        aop = self.conv4_3(x3_3)
        aop = torch.atan(aop) / 2. + math.pi / 4
        
        return s0, dolp, aop


def mae_loss(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    loss = torch.mean(0.1 * torch.abs(s0_true - s0_pred) + 
                      torch.abs(dolp_true - dolp_pred) + 
                      0.05 * torch.abs(aop_true - aop_pred))
    return loss


def smooth_l1_loss(reg_true, reg_pred, sigma):
    sigma2 = sigma ** 2
    diff = reg_true - reg_pred
    abs_diff = torch.abs(diff)
    mask = abs_diff < (1. / sigma2)
    
    loss = torch.where(mask, 0.5 * sigma2 * diff ** 2, abs_diff - 0.5 / sigma2)
    return loss.mean()

def smooth_loss(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    loss = (0.1 * smooth_l1_loss(s0_true, s0_pred, 2) + 
            smooth_l1_loss(dolp_true, dolp_pred, 2) + 
            0.01 * smooth_l1_loss(aop_true, aop_pred, 2))
    return loss


def ssim(ground_truth, ref, mx, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim_map(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * mx) ** 2
        C2 = (0.03 * mx) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    channel = ground_truth.size(1)
    window = create_window(window_size, channel)

    if ground_truth.is_cuda:
        window = window.cuda(ground_truth.get_device())
    window = window.type_as(ground_truth)

    return ssim_map(ground_truth, ref, window, window_size, channel, size_average)


def LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2):
    L, C, S, ssim_loss_value = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = torch.mean(0.1 * torch.abs(s0_true - s0_pred) + 
                      torch.abs(dolp_true - dolp_pred) + 
                      0.05 * torch.abs(aop_true - aop_pred)) - 0.02 * torch.log(C)
    return loss

def MSE_LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2):
    L, C, S, ssim_loss_value = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = torch.mean(0.1 * torch.square(s0_true - s0_pred) + 
                      torch.square(dolp_true - dolp_pred) + 
                      0.032 * torch.square(aop_true - aop_pred)) - 0.02 * torch.log(C)
    return loss


def std_variance(x):
    x_mean = torch.mean(x)
    variance = torch.sqrt(torch.sum((x - x_mean) ** 2) / (x.numel() - 1))
    return variance

def covariance(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    covariance = torch.sum((x - x_mean) * (y - y_mean)) / (x.numel() - 1)
    return covariance

def ssim_loss(x, y, mv, k1=0.01, k2=0.03):
    c1 = (k1 * mv) ** 2
    c2 = (k2 * mv) ** 2
    c3 = c2 / 2

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_std_var = std_variance(x)
    y_std_var = std_variance(y)
    xy_covar = covariance(x, y)

    L = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
    C = (2 * x_std_var * y_std_var + c2) / (x_std_var ** 2 + y_std_var ** 2 + c2)
    S = (xy_covar + c3) / (x_std_var * y_std_var + c3)
    SSIM = L * C * S

    loss = 1 - SSIM

    return L, C, S, loss

