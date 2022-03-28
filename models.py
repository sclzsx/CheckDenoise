#构建上采样和两个卷积层
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSUnet_down_Block(nn.Module):
    """Downsampling conv Block with batch normalization."""
    #批量归一化的下采样conv块
    def __init__(self, dim_in, dim_out):#获取模型参数
        super(MSUnet_down_Block, self).__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2, 2), #最大池化
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False), #kernel内核大小
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True), #加速神经网络训练
            # 数据拉回到均值为0，方差为1的正态分布上(归一化)，一方面使得数据分布一致，另一方面避免梯度消失、梯度爆炸
            nn.ReLU(inplace=True), #激活函数
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True), )
    def forward(self, x):
        return self.main(x)

class MSUnet_sk(nn.Module):
    """conv Block with batch normalization for Skip connection ."""
#用于Skip连接的批量归一化块
    def __init__(self, dim_in, dim_out):
        super(MSUnet_sk, self).__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(dim_in, dim_out, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.main(x)
        return x

class MSUnet_up_Block(nn.Module):
    """Upsampling conv Block with batch normalization."""
#带批量归一化的上采样转换块
    def __init__(self, dim_in, dim_out):
        super(MSUnet_up_Block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3 * dim_in // 2, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.convTrans = nn.ConvTranspose2d(dim_in, dim_in // 2, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, prev_map, prev_sk_map, x):
        x1 = self.convTrans(x)
        x2 = torch.cat((prev_map, prev_sk_map, x1), dim=1)
        x3 = self.main(x2)
        return x3

class MSUNet(nn.Module):
    """MSUnet network."""
    #MSUnet网络
    def __init__(self, input_dim=1, output_dim=1, conv_dim=64):  # 这里的几个参数为默认值，可以在初始化的时候自己设置
        super(MSUNet, self).__init__()
        layers = []
        # 卷积时的采样间隔为stride，为减少输入数目的参数 减少计算量
        # padding是填充，在输入特征图的每一边添加一定数目的行列，使得输出和输入的特征图的尺寸相同
        # 输出特征图尺寸  a-b+2d/c+1 其中a*a的特征图，经过b*b的卷积，步幅stride为c,填充为d
       #########****stride多1，步长就是间隔跳动； padding多1，就是外部表格填充多一层   max pool是卷积核内最大数
        layers.append(nn.Conv2d(input_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)) #input_dim代表张量的维度
        # output_dim是输出维度，是kears常见参数   dim指定的维度是可变的，其他都是固定不变的   nn.Conv2d作为二维卷积实现
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        self.down_Block1 = nn.Sequential(*layers)

        #对应的卷积核变成7*7之后填充变化为得到相同的特征值
        self.down_Block1_sk = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True))

        # Downsampling Bolcks with four times 4次下采样Bolcks
        self.down_Block2 = MSUnet_down_Block(conv_dim * 1, conv_dim * 2) #64变成128
        self.down_Block3 = MSUnet_down_Block(conv_dim * 2, conv_dim * 4) #128变成256
        self.down_Block4 = MSUnet_down_Block(conv_dim * 4, conv_dim * 8) #256变成512
        self.down_Block5 = MSUnet_down_Block(conv_dim * 8, conv_dim * 16) #512变成1024
        #卷积核变成7*7
        self.down_Block2_sk = MSUnet_sk(conv_dim * 1, conv_dim * 2)   #对应紫色的线
        self.down_Block3_sk = MSUnet_sk(conv_dim * 2, conv_dim * 4)
        self.down_Block4_sk = MSUnet_sk(conv_dim * 4, conv_dim * 8)

        # Up-sampling layers.上采样层 反卷积反池化
        #####******反卷积需要，边长增加一倍，channel通道减少一倍
        self.up_Block4 = MSUnet_up_Block(conv_dim * 16, conv_dim * 8) #对应墨绿色的线 从1024个特征值变成64个特征值
        self.up_Block3 = MSUnet_up_Block(conv_dim * 8, conv_dim * 4)
        self.up_Block2 = MSUnet_up_Block(conv_dim * 4, conv_dim * 2)
        self.up_Block1 = MSUnet_up_Block(conv_dim * 2, conv_dim * 1)

        # Full connection layer 完整连接层
        self.lastconv = nn.Conv2d(conv_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
    #  self.FeatureExtrator = VGG_FeatureExtractor() 自我特征提取器= VGG
    def forward(self, x):
        self.conv1 = self.down_Block1(x)
        self.conv2 = self.down_Block2(self.conv1)
        self.conv3 = self.down_Block3(self.conv2)
        self.conv4 = self.down_Block4(self.conv3)
        self.conv5 = self.down_Block5(self.conv4)

        self.conv1_sk = self.down_Block1_sk(x)
        self.conv2_sk = self.down_Block2_sk(self.conv1)
        self.conv3_sk = self.down_Block3_sk(self.conv2)
        self.conv4_sk = self.down_Block4_sk(self.conv3)

        y4 = self.up_Block4(self.conv4, self.conv4_sk, self.conv5)
        y3 = self.up_Block3(self.conv3, self.conv3_sk, y4)
        y2 = self.up_Block2(self.conv2, self.conv2_sk, y3)
        y1 = self.up_Block1(self.conv1, self.conv1_sk, y2)
        y = self.lastconv(y1)
        # rec_img = x - y
        rec_img = y
        #        out_img = self.FeatureExtrator(rec_img)
        return rec_img  # ,out_img #x为干净图片，n为散斑噪声，z值接近于x    f（y)为残差学习

