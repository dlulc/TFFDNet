# TFFD-Net: An effective two-stage mixed feature fusion and detail recovery dehazing network
Authors：Chen Li, Weiqi Yan, Hongwei Zhao, Shihua Zhou, Yuping Wang
### Abstract
Image dehazing is an effective means of improving the image quality captured in hazy weather. Although many dehazing models have produced excellent results, most of them ignore the accuracy of recovering details in haze-free images and lose some detail information during the dehazing process. To address this issue, we propose a two-stage dehazing network, TFFD-Net, dividing dehazing and detail recovery into two stages. Specifically, our model consists of four main components: haze removal sub network (HRSN), detail recovery sub network (DRSN), haze image guided feature correction module (FCM), and cross stage feature fusion module (CSFFM). After the basic haze is removed from the input image by using HRSN, the haze image as the second mode is fed into the FCM along with the dehaze feature. The information-rich character of the input image is utilized to guide the adjustment and feature enhancement of the dehazing feature, and the adjusted feature is finally input into the DRSN for multi scale detail reconstruction. During this period, in order to balance the two stages of the task, we also attentively fuse the feature of the two stages through CSFFM. Preventing information loss during dehazing in the first phase limits the detail recovery performance in the second stage. Our experiments on real and synthetic haze datasets indicate that our proposed TFFD-Net attains remarkable results in both evaluation metrics and visualization in a variety of scenarios. The source code and dataset are available from https://github.com/dlulc/TFFDNet/tree/master.
## Prerequisites:
Python 3.9

Pytorch 2.0.0

CUDA 11.8
## Dataset:
Our dataset can be obtained from the following link https://pan.quark.cn/s/e4084344283b, and the extraction code is eS9N.
