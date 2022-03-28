import math
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as udata
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
from skimage.measure import compare_psnr,compare_ssim,compare_mse
####零散的辅助函数
from speckle_noise import add_noise

def weights_init_kaiming(m): #初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])

def train_transform(imagesize): #图像大小  针对训练集图片变化 imagesize
    transform = transforms.Compose([
        #            transforms.Resize((size,size)),
        transforms.RandomCrop(imagesize), #裁剪
        transforms.RandomHorizontalFlip(),#水平翻转
        transforms.RandomVerticalFlip(),#垂直翻转
        transforms.ToTensor() #变成tensor（pytroch中需要采用的向量格式）格式  归一化 0-1
    ])
    return transform
def valid_transform(imagesize): #针对测试图片变化
    transform = transforms.Compose([
        transforms.CenterCrop(imagesize), #中间裁剪
        transforms.ToTensor()
    ])
    return transform
def Image_to_tensor():
    transform = transforms.Compose([
          transforms.ToTensor(),
    ])
    return transform

# class ImageDataFlow(udata.Dataset):
#     def __init__(self, labelDir, imagesize=None, is_training=False):
# #初始化参数 计算出感受野
#         self.is_training = is_training
#         self.train_transforms = train_transform(imagesize)#图片处理 图片裁剪
#         self.valid_transforms = valid_transform(imagesize)
#
#         img_paths = os.listdir(labelDir)  #读取路径
#         self.img_paths = [os.path.join(labelDir, k) for k in img_paths] #遍历路径中所有文件
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, index):
#         img_path = self.img_paths[index]
#         image = Image.open(img_path).convert('L')#绝对路径，转化灰度图像
#
#         if self.is_training:
#             # image  = data_augmentation(image,np.random.randint(1,8))
#             label = self.train_transforms(image)
#         else:
#             label = self.valid_transforms(image)
#         # noise_data = add_noise(label, self.shape, self.sigma, self.noise_model)
#         # noise_data = add_noise(label.numpy(),self.shape,self.scale,self.sigma,self.noise_model)
#         # noise_data = add_noise_rayleigh_np(label.numpy(), self.sigma)
#         # noise_data = np.expand_dims(noise_data,0)
#         # noise_data = torch.Tensor(noise_data)
#       #  return label

class ImageDataFlow(udata.Dataset):
    def __init__(self, labelDir, noise_mode=None, shape=None, sigma=None, imageSize=None, is_training=False):
        #self参数   label是干净图片    噪声类型   噪声模型 高斯噪声 图片尺寸
        self.noise_mode = noise_mode
        self.shape = shape
        self.sigma = sigma
        self.is_training = is_training
        self.train_transforms = train_transform(imageSize)#尺寸变大 原图256取96大小块 取出图像固定
        self.valid_transforms = valid_transform(imageSize)
        img_paths = os.listdir(labelDir)
        self.img_paths = [os.path.join(labelDir, k) for k in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('L') #读取图像 转换为灰度图像
        if self.is_training:
            # image  = data_augmentation(image,np.random.randint(1,8))
            label = self.train_transforms(image)
        else:
            label = self.valid_transforms(image)
        noise_data = add_noise(label, self.shape, self.sigma, self.noise_mode)
        return noise_data, label

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))#数组变换  扩大数据集
    if mode == 0:
        # original 原始
        out = out
    elif mode == 1:
        # flip up and down 上下翻转
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree 逆时针旋转90°
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down 旋转90°并上下翻转
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree 旋转180°
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip  旋转180度并翻转
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree 旋转270°
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip 旋转270度并翻转
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))#转置
