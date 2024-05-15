import os
import argparse
import time
import random
from tqdm import tqdm
import torch
import numpy as np
from datasets.datasets import Datasets
from torch.utils import data
from models.DehazeNet_dpconv2_no_down2 import DehazeNet
import torch.nn as nn
from loss import MSSSIM
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from utils.utils import psnr1, print_log_dehaze, ssim1, adjust_learning_rate
from loss.perceptual import Perceptual_loss


parser =argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default='./data', help="path to Dataset")
parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 4)')# 验证batch_size
parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str, default='checkpoints')
parser.add_argument('-datasets', help='', type=str, default='OTS')

opts = parser.parse_args()

data_root = opts.data_root
random_seed = opts.random_seed
exp_name = opts.exp_name
datasets = opts.datasets
val_batch_size = opts.val_batch_size

torch.manual_seed(opts.random_seed)
torch.cuda.manual_seed(opts.random_seed)
np.random.seed(opts.random_seed)
random.seed(opts.random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def predict():

    test_dst = Datasets(root=opts.data_root, split='test', dataset=datasets, crop=False)
    test_loader = data.DataLoader(test_dst, batch_size=val_batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Train set: %d" % (len(test_dst)))

    dehaze_net = DehazeNet().to(device)
    dehaze_net.load_state_dict(torch.load('./OTS_net_epoch30.pth', map_location=device))





    old_psnr = 0
    total_train_step = 0  # 记录训练的次数

    avg_psnr = 0
    avg_ssim = 0
    iteration = 0

    dehaze_net.eval()

    print("---------- 开始验证 ----------- ")

    with torch.no_grad():
        for foggy, images, file in tqdm(test_loader):
            foggy = foggy.to(device)
            images = images.to(device)

            dehazed_image_1, dehazed_images = dehaze_net(foggy)
            psnr = psnr1(dehazed_images, images)
            avg_psnr += psnr
            ssim = ssim1(dehazed_images, images)
            avg_ssim += ssim
            iteration += 1

        predict_psnr = avg_psnr / iteration
        predict_ssim = avg_ssim / iteration
        print("predict_psnr:{}".format(predict_psnr))
        print("predict_ssim:{}".format(predict_ssim))

















if __name__ == '__main__':
    predict()