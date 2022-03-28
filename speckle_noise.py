import torch
import numpy as np
import torch.distributions.gamma as G

#添加噪声
def add_noise(label, shape=None, sigma=None, noise_type=None):#添加噪声 label为标签位置  sigma是高斯噪声影响
    # noise——type判断是否有gamma或瑞利噪声
    if sigma is None:
        sigma = np.random.randint(16)
        sigma_ = np.random.randint(16)
    else:
        sigma_ = sigma#激活函数应用于高斯噪声
   #判断是否有模型
    if shape is None:
        shape = np.random.choice(np.arange(1, 11, 1))  # why is the val? val是验证是否过拟合，调节训练参数
    noise_gauss = torch.randn(label.size()).mul_(sigma / 255.0).to(label.device)  #gauss算法 高斯噪声拟合
   #判断是否有噪声类型
    if noise_type is None:
        val = np.random.randint(2)
        if val == 0: #如果给定的任意值为0，则为伽马噪声模型
            m = G.Gamma(shape, rate=shape)
            noise_gamma = m.sample(label.size()).to(label.device)
            noise_data = label * torch.sqrt(noise_gamma) + noise_gauss
        elif val == 1: #若为1，则为瑞利噪声
            data_rayleigh = np.random.rayleigh(label.cpu().numpy(), label.size())
            noise_data = torch.Tensor(data_rayleigh.to(label.device)) + noise_gauss
    elif noise_type == 'Gamma': #伽马噪声
        m = G.Gamma(shape, rate=shape)
        noise_gamma = m.sample(label.size()).to(label.device)
        noise_data = label * torch.sqrt(noise_gamma) + noise_gauss
    elif noise_type == 'Rayleigh':#瑞利噪声
        data_rayleigh = np.random.rayleigh(label.cpu().numpy(), label.shape)
        # data_rayleigh = np.random.rayleigh(label.cpu().numpy() * np.sqrt(2 / np.pi), label.shape)
        noise_data = torch.Tensor(data_rayleigh).to(label.device) + noise_gauss
    else:
        print('!Error:The noise model is false...')
    return noise_data
