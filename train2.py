import os
import argparse
import time
import random
from tqdm import tqdm
import torch
import numpy as np
from datasets.datasets import Datasets
from torch.utils import data
from models.DehazeNet2 import DehazeNet
import torch.nn as nn
from loss import MSSSIM
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from utils.utils import to_psnr, print_log_dehaze, calc_ssim, calc_psnr, adjust_learning_rate
from loss.perceptual import Perceptual_loss


parser =argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default='./data', help="path to Dataset")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help='batch size (default: 16)')    # 训练batch_size
parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 4)')# 验证batch_size
parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--num_epochs", type=int, default=200, help='num_epochs')
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str, default='checkpoints')

opts = parser.parse_args()

data_root = opts.data_root
batch_size = opts.batch_size
lr = opts.lr
random_seed = opts.random_seed
num_epochs = opts.num_epochs
exp_name = opts.exp_name

torch.manual_seed(opts.random_seed)
torch.cuda.manual_seed(opts.random_seed)
np.random.seed(opts.random_seed)
random.seed(opts.random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def train():

    train_dst = Datasets(root=opts.data_root, split='train')
    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Train set: %d" % (len(train_dst)))

    dehaze_net = DehazeNet().to(device)
    total = sum(torch.numel(parameter) for parameter in dehaze_net.parameters())
    print("Number of parameter:%.2fM" % (total / 1e6))
    l1_loss = nn.L1Loss()
    msssim_loss = MSSSIM()
    mse_loss = nn.MSELoss().to(device)
    smoothl1_loss = nn.SmoothL1Loss()
    perceptual_loss = Perceptual_loss()


    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=lr)

    #   添加tensorboard
    writer = SummaryWriter("logs_train")

    old_psnr = 0
    total_train_step = 0  # 记录训练的次数

    for epoch in range(num_epochs):
        psnr_list = []
        ssim_list = []
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        dehaze_net.train()
        for foggy, images, file in tqdm(train_loader):

            foggy = foggy.to(device)
            images = images.to(device)

            dehazed_image = dehaze_net(foggy)

            optimizer.zero_grad()
            SmoothL1 = smoothl1_loss(dehazed_image, images)
            # Mse = mse_loss(dehazed_image, images)
            # L1_loss = l1_loss(dehazed_image, images)
            Perceptual = perceptual_loss(dehazed_image, images)
            # ms_ssim = 1 - msssim_loss(dehazed_image, images)

            loss_total = SmoothL1 + 0.04 * Perceptual
            # print("SmoothL1:{},Perceptual:{}".format(SmoothL1, Perceptual))


            loss_total.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:  # 逢百打
                writer.add_scalar("train_loss", loss_total.item(), total_train_step)
                print("SmoothL1:{},Perceptual:{}".format(SmoothL1, Perceptual))

            psnr_list.extend(calc_psnr(dehazed_image, images))
            ssim_list.extend(calc_ssim(dehazed_image, images))

        train_psnr = sum(psnr_list) / len(psnr_list)
        train_ssim = sum(ssim_list) / len(ssim_list)



        if os.path.exists('./{}/dehaze/'.format(exp_name)) == False:
            os.makedirs('./{}/dehaze'.format(exp_name))
        torch.save(dehaze_net.state_dict(), './{}/dehaze/latest'.format(exp_name))
        torch.save(dehaze_net.state_dict(), './{}/dehaze/dehaze_net_epoch{}.pth'.format(exp_name, str(epoch + 1)))
        print("epoch:{}:train_psnr:{},train_ssim:{}".format(str(epoch + 1), train_psnr,train_ssim))
        if train_psnr >= old_psnr:

            torch.save(dehaze_net.state_dict(), './{}/dehaze/best'.format(exp_name))
            print('best model saved')
            old_psnr = train_psnr


        one_epoch_time = time.time() - start_time

        print_log_dehaze(epoch + 1, opts.num_epochs, one_epoch_time, train_psnr, train_ssim, 0, 0, opts.exp_name)




    writer.close()






if __name__ == '__main__':
    train()