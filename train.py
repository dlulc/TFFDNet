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
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2, help='batch size (default: 16)')    # 训练batch_size
parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 4)')# 验证batch_size
parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--num_epochs", type=int, default=200, help='num_epochs')
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str, default='checkpoints')
parser.add_argument('-datasets', help='', type=str, default='OTS')

opts = parser.parse_args()

data_root = opts.data_root
batch_size = opts.batch_size
lr = opts.lr
random_seed = opts.random_seed
num_epochs = opts.num_epochs
exp_name = opts.exp_name
datasets = opts.datasets
val_batch_size = opts.val_batch_size

torch.manual_seed(opts.random_seed)
torch.cuda.manual_seed(opts.random_seed)
np.random.seed(opts.random_seed)
random.seed(opts.random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():

    train_dst = Datasets(root=opts.data_root, split='train',dataset=datasets, crop=True)
    test_dst = Datasets(root=opts.data_root, split='test', dataset=datasets, crop=False)
    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = data.DataLoader(test_dst, batch_size=val_batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Train set: %d" % (len(train_dst)))

    dehaze_net = DehazeNet().to(device)

    total_parameters = count_parameters(dehaze_net)
    total_parameters_m = total_parameters / 1e6

    print(f"Total trainable parameters: {total_parameters_m} M")
    # dehaze_net = torch.nn.DataParallel(dehaze_net)
    # dehaze_net.load_state_dict(torch.load('./checkpoints/city2/latest', map_location=device))
    total = sum(torch.numel(parameter) for parameter in dehaze_net.parameters())
    print("Number of parameter:%.2fM" % (total / 1e6))
    l1_loss = nn.L1Loss()
    msssim_loss = MSSSIM()
    mse_loss = nn.MSELoss().to(device)
    smoothl1_loss = nn.SmoothL1Loss().to(device)
    perceptual_loss = Perceptual_loss()


    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=lr)

    #   添加tensorboard
    writer = SummaryWriter("logs_train")

    old_psnr = 0
    total_train_step = 0  # 记录训练的次数

    for epoch in range(num_epochs):
        avg_psnr = 0
        avg_ssim = 0
        iteration = 0
        max_psnr = 0
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        dehaze_net.train()
        for foggy, images, file in tqdm(train_loader):

            foggy = foggy.to(device)
            images = images.to(device)

            dehazed_image_1, dehazed_image = dehaze_net(foggy)

            optimizer.zero_grad()
            SmoothL1 = smoothl1_loss(dehazed_image, images)
            Mse = mse_loss(dehazed_image, images)
            L1_loss = l1_loss(dehazed_image, images)
            Perceptual = perceptual_loss(dehazed_image, images)
            ms_ssim = 1 - msssim_loss(dehazed_image, images)

            loss_total = SmoothL1 + 0.04 * Perceptual



            loss_total.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 200 == 0:  # 逢百打
                writer.add_scalar("train_loss", loss_total.item(), total_train_step)
                print("SmoothL1:{},Perceptual:{}".format(SmoothL1, Perceptual))

        #     psnr_list.extend(calc_psnr(dehazed_image, images))
        #     ssim_list.extend(calc_ssim(dehazed_image, images))
        #
        # train_psnr = sum(psnr_list) / len(psnr_list)
        # train_ssim = sum(ssim_list) / len(ssim_list)

        print("---------- 开始验证 ----------- ")
        dehaze_net.eval()
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

            print("Epoch:{}:predict_psnr:{}".format(epoch+1, predict_psnr))
            print("Epoch:{}:predict_ssim:{}".format(epoch+1, predict_ssim))




        if os.path.exists('./{}/{}/'.format(exp_name, datasets)) == False:
            os.makedirs('./{}/{}/'.format(exp_name,datasets))
        torch.save(dehaze_net.state_dict(), './{}/{}/latest'.format(exp_name, datasets))
        torch.save(dehaze_net.state_dict(), './{}/{}/{}_net_epoch{}.pth'.format(exp_name, datasets, datasets, str(epoch + 1)))
        print("epoch:{}:predict_psnr:{},predict_ssim:{}".format(str(epoch + 1), predict_psnr, predict_ssim))
        if predict_psnr >= old_psnr:

            torch.save(dehaze_net.state_dict(), './{}/{}/best'.format(exp_name, datasets))
            print('best model saved')
            old_psnr = predict_psnr
            max_psnr = predict_psnr
            print("epoch:{}-----------------------max_psnr:{}".format(epoch,max_psnr))


        one_epoch_time = time.time() - start_time

        print_log_dehaze(epoch + 1, opts.num_epochs, one_epoch_time, predict_psnr, predict_ssim, opts.exp_name)




    writer.close()






if __name__ == '__main__':
    train()