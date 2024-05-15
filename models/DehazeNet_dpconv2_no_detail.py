import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- 特征提取模块----------- #

class DRB(nn.Module):  # 尺寸没变
    def __init__(self, in_channels=64, growth_rate=32):   #in_channels=64, growth_rate=32
        super(DRB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)  # 64-》32
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)  # 96->32
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)  # 128->32
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, kernel_size=1)  # 160->32

        self.depthwise = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192)
        self.pointwise = nn.Conv2d(192, in_channels, 1)




    def forward(self, x):
        out1 = torch.relu(self.conv1(x))  # 32
        out2 = torch.relu(self.conv2(torch.cat([x, out1], 1)))  # cat:96   out2:32
        out3 = torch.relu(self.conv3(torch.cat([x, out1, out2], 1)))  # cat:128  out3:32
        out4 = torch.relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))  # cat:160  out4:32
        out5 = torch.relu(self.pointwise(self.depthwise(torch.cat([x, out1, out2, out3, out4], 1))))  # cat:192->32->64

        out = out5 + x


        return out



# ---------- 注意力模块 ---------- #
class CA(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(CA, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels*2, 1, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool_output = self.max_pool(x)
        avg_pool_output = self.avg_pool(x)

        pool_output = torch.cat([max_pool_output, avg_pool_output], 1)

        attention_weights = self.fc(pool_output)

        x = x * attention_weights
        return x

class DualAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualAttention, self).__init__()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        self.ca = CA(inchannel=64, ratio=16)

        self.sa = SpatialAttention(in_channels=64)

        self.conv1_1 = nn.Conv2d(in_channels=64*2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        res = torch.relu(self.conv1(self.bn(x)))

        ca_feature = self.ca(res)
        sa_feature = self.sa(res)

        cat = torch.cat([ca_feature, sa_feature], dim=1)

        out = self.conv1_1(cat)


        return out


# --------- 特征增强模块 ---------  #
class FE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FE, self).__init__()

        self.haze = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.dehaze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x1, x2):

        x1 = self.haze(x1)
        x2 = self.dehaze(x2)

        out = x2 + x1 * x2

        return out




# ---------纹理增强和细节细化------------#
class Detail(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Detail, self).__init__()

        # 细节细化
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)  # 3-16
        self.relu = nn.LeakyReLU()


        self.bn = nn.BatchNorm2d(in_channels)

        # 第一层多尺度卷积
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, dilation=1, stride=1, padding=1)  # 大小没变    32-64
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, dilation=3, stride=1, padding=3)  # 大小没变
        self.conv1_5 = nn.Conv2d(64, 64, kernel_size=3, dilation=5, stride=1, padding=5)  # 大小没变
        self.conv3 = nn.Conv2d(64 * 3, 64, kernel_size=1, stride=1)  # 大小没变

        # 第二层多尺度卷积
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, dilation=3, stride=1, padding=3)
        self.conv2_5 = nn.Conv2d(64, 64, kernel_size=3, dilation=5, stride=1, padding=5)
        self.conv4 = nn.Conv2d(64 * 3, 64, kernel_size=1, stride=1)

        self.conv5 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

        self.pixel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 64-64
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 64-1
            nn.Sigmoid()
        )


    def forward(self, x):

        x1 = self.relu(self.conv1(self.bn(x)))


        # x2 = x1 + x

        x1_1 = self.conv1_1(x1)
        x1_3 = self.conv1_3(x1)
        x1_5 = self.conv1_5(x1)

        x3 = torch.cat((x1_1, x1_3, x1_5), dim=1)  # 192
        x4 = self.relu(self.conv3(x3))  # 192->64

        x2_1 = self.conv2_1(x4)
        x2_3 = self.conv2_3(x4)
        x2_5 = self.conv2_5(x4)

        x5 = torch.cat((x2_1, x2_3, x2_5), dim=1)

        x6 = self.relu(self.conv4(x5)) + x1#64

        x7 = self.pixel_attention(x6) * x6

        out = self.relu(self.conv5(x7))

        return x + out


# -------- 融合模块 ---------- #
class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class SDFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(SDFM, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_C)
        self.bn2 = nn.BatchNorm2d(in_C)
        self.denselayer1 = DenseLayer(in_C=in_C, out_C=out_C)
        self.denselayer2 = DenseLayer(in_C=in_C, out_C=out_C)

        self.conv = nn.Conv2d(in_C * 2, out_C, kernel_size=1, padding=0)

    def forward(self, x1, x2):

        dense1 = self.denselayer1(x1)
        dense2 = self.denselayer2(x2)

        att1 = dense1 * dense2

        output2 = att1 * x1 + x2
        output1 = att1 * x2 + x1

        out = torch.relu(self.conv(torch.cat([output1, output2], 1)))


        return out






class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )



    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input  ## 先对特征进行一步自校正wD
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim=1)

        return x1, x2





# --------- 颜色校正模块 -------- #
class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(in_C, out_C)
        self.Infobj = DenseLayer(in_C, out_C)

        self.obj_fuse = GEFM(cat_C, out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out







class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()

        # self.down1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # self.down2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        self.relu1 = nn.ReLU()

        self.drb1 = DRB(in_channels=64, growth_rate=32)
        self.drb2 = DRB(in_channels=64, growth_rate=32)
        self.drb3 = DRB(in_channels=64, growth_rate=32)
        self.drb4 = DRB(in_channels=64, growth_rate=32)
        self.drb5 = DRB(in_channels=64, growth_rate=32)
        self.drb6 = DRB(in_channels=64, growth_rate=32)
        self.drb7 = DRB(in_channels=64, growth_rate=32)
        self.drb8 = DRB(in_channels=64, growth_rate=32)



        self.attention1 = DualAttention(in_channels=64, out_channels=64)
        self.attention2 = DualAttention(in_channels=64, out_channels=64)
        self.attention3 = DualAttention(in_channels=64, out_channels=64)
        self.attention4 = DualAttention(in_channels=64, out_channels=64)


        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        self.tanh1 = nn.Tanh()


        self.fe = FE(in_channels=3, out_channels=3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)

        self.detail1 = Detail(in_channels=64, out_channels=64)
        self.detail2 = Detail(in_channels=64, out_channels=64)
        self.detail3 = Detail(in_channels=64, out_channels=64)
        self.detail4 = Detail(in_channels=64, out_channels=64)
        self.detail5 = Detail(in_channels=64, out_channels=64)
        self.detail6 = Detail(in_channels=64, out_channels=64)
        self.detail7 = Detail(in_channels=64, out_channels=64)
        self.detail8 = Detail(in_channels=64, out_channels=64)

        self.fusion1 = SDFM(in_C=64, out_C=64)
        self.fusion2 = SDFM(in_C=64, out_C=64)
        self.fusion3 = SDFM(in_C=64, out_C=64)



        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        self.tanh2 = nn.Tanh()

        self.block_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block_relu1 = nn.ReLU()
        self.block_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block_relu2 = nn.ReLU()
        self.block_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block_relu3 = nn.ReLU()
        self.block_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block_relu4 = nn.ReLU()

    def forward(self, x):
        pre = self.relu1(self.conv1(x))

        x1 = self.drb1(pre)
        x2 = self.drb2(x1)
        x3 = self.attention1(x2) + x2
        block1_out = self.block_relu1(self.block_conv1(x3)) + pre

        x4 = self.drb3(block1_out)
        x5 = self.drb4(x4)
        x6 = self.attention2(x5) + x5
        block2_out = self.block_relu2(self.block_conv2(x6)) + block1_out

        x7 = self.drb5(block2_out)
        x6 = self.drb6(x7)
        x9 = self.attention3(x6) + x6
        block3_out = self.block_relu3(self.block_conv3(x9)) + block2_out

        x10 = self.drb7(block3_out)
        x11 = self.drb8(x10)
        x12 = self.attention4(x11) + x11
        block4_out = self.block_relu4(self.block_conv4(x12)) + block3_out

        block4_out = block4_out
        out1 = self.conv2(block4_out)
        # out1_up = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out1 = self.tanh1(out1) + x

        # -----第一阶段完成----- #

        fe = self.fe(x, out1)

        fe1 = torch.relu(self.conv3(fe))

        # details_block1 = self.detail2(self.detail1(fe1))
        # fusion1 = self.fusion1(block1_out, details_block1)
        #
        # details_block2 = self.detail4(self.detail3(fusion1))
        # fusion2 = self.fusion2(block2_out, details_block2)
        #
        # details_block3 = self.detail6(self.detail5(fusion2))
        # fusion3 = self.fusion3(block3_out, details_block3)
        #
        # details_block4 = self.detail8(self.detail7(fusion3))
        #
        # details_out = details_block4

        out2 = self.conv4(fe1)
        # out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.tanh2(out2) + fe

        return out1, out














