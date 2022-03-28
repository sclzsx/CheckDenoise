import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim #优化模块  可视化模块  辅助函数模块
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter #训练数据可视化模块
from models import  MSUNet  #自己定义模块
from dataset import prepare_data, Dataset
from utils import *
#训练模型+测试  导入库
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#超参数定义 数据预处理指示符、训练数据集测试数据的路径、文件格式
parser = argparse.ArgumentParser(description="MSUNet")
parser.add_argument("--label_dir", type=str,default='./data/train_aug', help='path of train set') #训练集路径
parser.add_argument("--val_dir", type=str, default='./data/Set12', help='path of val set')#验证集路径
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')#false表示不进行h5文件生成
parser.add_argument("--imageSize", type=int, default=64,help="image size to train")  #修改图片尺寸
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")  #维度修改
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs") #学习率设置
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
#噪声类型：mode；s为单噪声 一种噪声水平；B为盲噪声。其中S需要指定，B不需要
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--noise_mode",type=str,default='Gamma',help='') #当定义噪声类型为gamma时 当为B噪声时定义default为B
parser.add_argument("--sigma", type=float, default=5, help='Gauss noise level used on validation set') #设定噪声级别为5
parser.add_argument("--shape", type=float, default=10, help='Gamma  noise level used on validation set') #乘法
parser.add_argument("--test_file", type=str, default='./r.txt', help="Training batch size")

opt = parser.parse_args() #默认定义参数别名

def main():
    train_psnr_box=[]
    test_psnr_box = []
    train_loss_box=[]
    test_loss_box= []
    test_ssim_box=[]
    train_ssim_box=[]##ssim指标

    # Load dataset  数据载入 h5或者直接读取数据
    print('Loading dataset ...\n')
    #dataset_train = Dataset(train=True)
    #dataset_val = Dataset(train=False)
    dataset_train = ImageDataFlow(opt.label_dir, opt.noise_mode, opt.shape, opt.sigma, opt.imageSize,
                                  is_training=True)  #训练集
    dataset_val = ImageDataFlow(opt.val_dir, opt.noise_mode, opt.shape, opt.sigma, opt.imageSize,
                                is_training=False) #测试数据
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_val=DataLoader(dataset=dataset_val,batch_size=1, shuffle=False)
    #载入数据迭代器
   # loader_valid = DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False)
#训练集 两种导入数据的方式
    #dataset_train = ImageDataFlow(opt.label_dir, imagesize=opt.imagesize, is_training=True)
#测试数据
    #dataset_val = ImageDataFlow(opt.val_dir,imagesize=128, is_training=False)
# 载入数据迭代器生成
    #loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model 构建模型 指定优化函数和loss函数
    net = MSUNet() #conv_dim=1, num_of_layers=opt.num_of_layers
    # net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # 移动到GPU  从cpu移至gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = net.to(device)
    criterion = criterion.to(device)
    # Optimizer  权重、偏置进行优化
    optimizer = optim.Adam(model.parameters(), lr=opt.lr) #Adam优化算法  指定参数、指定学习率
    # training
    writer = SummaryWriter(opt.outf) #写入器  给定输出路径
    step = 0
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'  B为盲噪声  S为高斯噪声
    for epoch in range(opt.epochs):   #进行50轮学习   epoch是轮数
        if epoch < opt.milestone:   #30组后衰减学习率
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10. #衰减 学习率降低十倍
        # set learning rate 建立学习率
        for param_group in optimizer.param_groups: #遍历列表 元素自定 每一轮进行学习率更新
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        # for i, data in enumerate(loader_train, 0): #载入数据，更新模型参数   loss更新模型
        #     # training step
        #     model.train()
        #     model.zero_grad()
        #     optimizer.zero_grad() #模型、梯度清零
        #     img_train = data.to(device)
        for i, (imgn_train, img_train) in enumerate(loader_train, 0):
            # if i+1 == 10:   讲含噪图片和干净图片载入
            # break      imgn_train含噪图片  img_train干净图片
            imgn_train = imgn_train.to(device)
            img_train = img_train.to(device)
            print(imgn_train.shape)

       #高斯噪声
            # if opt.mode == 'S':
            #     noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            # if opt.mode == 'B':
            #     noise = torch.zeros(img_train.size())
            #     stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            #     for n in range(noise.size()[0]):
            #         sizeN = noise[0, :, :, :].size()
            #         noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

            # noise = noise.to(device)
            # imgn_train = img_train + noise #得到含噪图片
            # imgn_train = imgn_train.to(device)

            out_train = model(imgn_train) #输入到模型中
            loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
            #img_train与out_train相减得到干净图片做loss
            #label干净图片+给定噪声得到imgn_train,根据模型去除噪声后得到out_train，经过imgn_train-out_train得到输出的图片与label
            #干净图片做对比分析
            optimizer.zero_grad()
            loss.backward()   #计算出loss反向传播 根据loss大小进行参数更新
            optimizer.step()   #训练结束

            train_loss_box.append(loss.item())#################

            # results
            model.eval()  #模型评估，不进行参数更新
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)  #psnr越大去噪效果越好


            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]

            train_psnr_box.append(psnr_train)

            #psnr曲线
            if step % 10 == 0: #每10步观测一下
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch

        model.eval() #测试过程 验证集
        # validate
        # validate
        psnr_val = 0
        loss_val = 0
        ssim_val=0
        with torch.no_grad():
            for k, (imgn_val, img_val) in enumerate(loader_val, 0):
                img_val = img_val.to(device)
                imgn_val = imgn_val.to(device)
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
                loss_val += criterion(out_val, img_val) / (img_val.size()[0] * 2)
        psnr_val /= len(dataset_val)
        test_psnr_box.append(psnr_val)
        loss_val/=len(dataset_val) ##############################
        test_loss_box.append(loss_val.item())

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images  验证过程
        out_train = torch.clamp(model(imgn_train), 0., 1.) #减噪
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth')) #保存参数
    with open(opt.test_file, 'w', encoding='utf-8') as f:
        f.writelines(str(train_psnr_box[:]) + '\n')  # 训练的psnr 0
        f.writelines(str(test_psnr_box[:]) + '\n')  # 1
        f.writelines(str(train_loss_box[:]) + '\n')#训练的loss值 2
        f.writelines(str(test_loss_box[:]) + '\n') # 3
    f.close()

if __name__ == "__main__": #图像预处理
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=50,aug_times=1) #斑块40*40  aug_times指扩充次数
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)#50*50
    main()
