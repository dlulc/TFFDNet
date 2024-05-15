import sys
import time

import cv2
import torchvision
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os
import torch
import torch.nn.functional as F
from math import log10
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import deltaE_ciede2000 as compare_ciede
from utils import pytorch_ssim



def print_log_dehaze(epoch, num_epochs, one_epoch_time, predict_psnr, predict_ssim, exp_name):
    print('dehaze_net：({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:4f}'
          .format(one_epoch_time, epoch, num_epochs, predict_psnr, predict_ssim ))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print(
            'dehaze_net：Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, predict_psnr, predict_ssim), file=f)




def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 400

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))



def ssim1(img1, img2):
    ssim = pytorch_ssim.ssim(img1, img2)

    return ssim

def psnr1(img1, img2):
    criterion = torch.nn.MSELoss(size_average=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion.to(device)
    mse = criterion(img1, img2)
    psnr = 10 * log10(1 / mse)

    return psnr
