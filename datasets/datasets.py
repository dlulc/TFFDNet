import json
import os


import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange

class Datasets(data.Dataset):


    def __init__(self, root, split='train', dataset='OTS', crop=True):
        super().__init__()
        self.root = os.path.expanduser(root)  # ./data
        self.mode = 'gtFine'
        self.split = split
        self.images_dir = os.path.join(self.root, split)  # ./data/train/

        self.gt_dir = os.path.join(self.root, split)  # ./data/train/

        # self.transform = Compose([ToTensor(), Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])])
        self.transform1 = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5, 0.5])])
        self.transform2 = Compose([ToTensor()])

        self.resize = crop######################################### 裁剪

        self.foggy_images = []
        self.gt_images = []

        self.foggy_dir = os.path.join(self.root, split)    # ./data/train

        foggy_dir = os.path.join(self.foggy_dir, dataset, 'hazy')  # ./data/train/OTS/hazy
        gt_dir = os.path.join(self.images_dir, dataset, 'clear')  # ./data/train/OTS/clear

        for file_name in os.listdir(foggy_dir):


            self.foggy_images.append(os.path.join(foggy_dir, file_name))

            if split == 'train':
                if dataset == 'ITS':
                    gt_name = file_name.split('_')[0] + '.png'
                elif dataset == 'OTS':
                    gt_name = file_name.split('_')[0] + '.jpg'
                elif dataset == 'O_haze':
                    gt_name = file_name.split('_')[0] + '_outdoor_GT.jpg'
                else:
                    gt_name = file_name.split('_')[0] + '_GT.png'
            elif split == 'test':
                if dataset == 'NH_haze' or dataset == 'Dense_haze':
                    gt_name = file_name.split('_')[0] + '_GT.png'
                elif dataset == 'O_haze':
                    gt_name = file_name.split('_')[0] + '_outdoor_GT.jpg'
                else:
                    gt_name = file_name.split('_')[0] + '.png'
            # print('file_name:', file_name)
            # print('gt_name:', gt_name)

            self.gt_images.append(os.path.join(gt_dir, gt_name))



        # for i in range(len(self.foggy_images)):
        #     foggy_image = Image.open(self.foggy_images[i]).convert('RGB')
        #     image = Image.open(self.gt_images[i]).convert('RGB')
        #
        #     foggy_image.show()
        #     image.show()



    def get_images(self, index):

        foggy_image = Image.open(self.foggy_images[index]).convert('RGB')
        gt_image = Image.open(self.gt_images[index]).convert('RGB')
        # w, h = foggy_image.size
        # gt_image.resize((w, h), Image.ANTIALIAS)

        if self.split == 'test':
            self.resize = False
        if self.resize:
            crop_width, crop_height = 240, 240
            width, height = foggy_image.size
            if width < crop_width and height < crop_height:
                foggy_image = foggy_image.resize((crop_width, crop_height), Image.ANTIALIAS)  # Image.ANTIALIAS是选择高质量缩放滤镜。
                gt_image = gt_image.resize((crop_width, crop_height), Image.ANTIALIAS)
            elif width < crop_width:
                foggy_image = foggy_image.resize((crop_width, height), Image.ANTIALIAS)
                gt_image = gt_image.resize((crop_width, height), Image.ANTIALIAS)
            elif height < crop_height:
                foggy_image = foggy_image.resize((width, crop_height), Image.ANTIALIAS)
                gt_image = gt_image.resize((width, crop_height), Image.ANTIALIAS)
            width, height = foggy_image.size
            # --- x,y coordinate坐标 of left-top corner角 --- #
            x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
            foggy_crop_img = foggy_image.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img = gt_image.crop((x, y, x + crop_width, y + crop_height))

            foggy_image_transform = self.transform1(foggy_crop_img)
            gt_image_transform = self.transform2(gt_crop_img)
        else:
            foggy_image_transform = self.transform1(foggy_image)
            gt_image_transform = self.transform2(gt_image)


        n = len(self.foggy_images)
        foggy_path = self.foggy_images[index % n]

        return foggy_image_transform, gt_image_transform, foggy_path

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.foggy_images)



