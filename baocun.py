import os
import argparse
import torch
from models import *
import glob
from utils import *
from torchvision.utils import *
import speckle_noise
import torch.distributions.gamma as G

parser = argparse.ArgumentParser(description="Save_pic")

parser.add_argument("--dir_path",type=str,default="Ot_wBn_res",help="")
parser.add_argument("--val_dir", type=str, default='./data/Set12', help='path of val set')
parser.add_argument("--test_save_dir", type=str, default='Rayleigh/gamma', help='save dir')
#parser.add_argument("--test_save_dir", type=str, default='Rayleigh/ray', help='save dir')
parser.add_argument("--txt_file",type=str, default='record_result.txt', help='')
parser.add_argument("--test_data", type=str, default='./data/Set12', help='test on Set12 or Set68')
parser.add_argument("--train_mode",type=str, default='S', help='for recording')
parser.add_argument("--test_noise_type",type=str,default='Gauss', help='for recording') #Gaussi高斯
parser.add_argument("--train_noise_mode",type=str,default='Gamma', help='for recording')
parser.add_argument("--add_noise",type=str,default='Gamma', help='for recording')
parser.add_argument("--test_noiseL", type=float, default=5, help='测试集上使用的高斯噪声级') #25
#parser.add_argument("--test_shape", type=float, default=1, help='gamma shape parameters used on test set') #测试集上使用的伽马形状参数
parser.add_argument("--rand_seed", type=int, default=11, help='random seed for random generator')
parser.add_argument("--logs_dir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_epoch", type=float, default=50, help='net model parameters for test')

opt = parser.parse_args()

def main():
    file_path = os.path.join(opt.dir_path,opt.test_save_dir)  ##测试模型路径: Ot_wBn_res/coslrnoRelu\Rayleigh/test_sig0_L1
    if not os.path.exists(file_path):  # not none   如果文件不存在
        os.makedirs(file_path)  #建一个

    file = open(opt.txt_file, 'a')
    write_head(file, file_path, opt.test_data, opt.train_noise_mode, opt.test_noise_type, opt.test_noiseL)
    file.close()

    with open(opt.txt_file, 'a') as file:
        file.writelines('X2_' + "\tImage\tpsnr_in\t\tpsnr_out\tssim_out\t\t\n")
        file.writelines("********************************************************************\n")


    torch.random.manual_seed(opt.rand_seed)
    print('Loading model ...\n')
    net =MSUNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    # save_dict = torch.load(os.path.join(opt.dir_path, opt.log_dir, 'net_{}.pth'.format(opt.test_epoch)),
    #                        map_location=lambda storage, loc: storage)
    #save_dict = torch.load(os.path.join(opt.logdir, 'net.pth'), map_location='cpu')

    model.load_state_dict(torch.load(os.path.join(opt.logs_dir, 'net.pth'), map_location='cpu'))
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_source.sort()

    # process datas
    print('Starting test ...\n')
    model.eval()

    result_list = []
    input_list = []
    psnr_input = 0
    psnr_test = 0
    count = 0
    size = 483
    Trans = transforms.ToTensor()
    for f in files_source:
        count += 1
        Img = Image.open(f).convert("L")
        # ISource = Image_to_Tensor(Img).to(device)ra
        Isource = Trans(Img)
        Isource  = torch.unsqueeze(Isource , 0)
        # noise
        #noise = torch.FloatTensor(Isource .size()).normal_(mean=0, std=opt.test_noiseL / 255.).to(device)
        # noisy image
        noise_data=add_noise(label=Isource,shape=10,sigma=5,noise_type='Gamma')
        print(noise_data.shape)
################################
        with torch.no_grad():  # this can save much memory
            # Out = torch.clamp(model(INoisy), 0., 1.).to(device)
            Out = torch.clamp(model(noise_data), 0., 1.).to(device)

            psnr_in = batch_PSNR(noise_data, Isource, 1.)  # 含噪图片和原图
            psnr = batch_PSNR(Out, Isource, 1.)  # 去噪图片和原图
            psnr_input += psnr_in
            psnr_test += psnr
            print("%s PSNR %f" % (f, psnr))

            result_list.append([count, psnr])
            input_list.append([count, psnr_in])

            with open(opt.txt_file,'a') as f:
                f.writelines('\t\t\t' + str(count)+'\t\t'+str(psnr_in)[:7] + '\t\t'+str(psnr)[:7]+'\t\n')

            # save fixed image with rectangle
            if count in range(13) :
                # save together
                # INoisy = normalize1(INoisy)
                filenames = os.path.join(file_path, 'net%s_N%s_psnr_%.2f_%.2f.png' % (opt.test_epoch, count, psnr_in, psnr))
                save_image(torch.cat((Isource, noise_data, Out), 3), filenames, normalize=False)
                index = 'net%s_I%s_' % (opt.test_epoch, count)
                save_image(noise_data, os.path.join(file_path, index + 'INoisy.png'), normalize=False)
                save_image(Out, os.path.join(file_path, index + 'Out.png'), normalize=False)


    psnr_test /= len(files_source)
    psnr_input /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print("PSNR_in on test data %f " % psnr_input)

    result_list.append([0, np.mean(np.array(input_list)[:, 1])])

    result_list.append([count, psnr_test])
    np.savetxt(os.path.join(file_path, 'result.txt'), np.vstack(result_list), fmt='%2.4f')

    with open(opt.txt_file, 'a') as f:
        f.writelines('\n\t\t\t' + 'average' + '\t\t\t\t' + str(psnr_test)[:7] + '\t\n')

    print("Finished test of the test image! (#`O′) ")


def write_head(f, model_path, test_data,train_n_mode, test_n_type, sigma):
    f.writelines("\n=======================================================================\n")
    f.writelines(" \t\t\t\t测试模型路径:\n"+model_path)
    f.writelines("\n-----  ---------    ----------    ------------    --------    ----------\n")
    f.writelines("\t" + test_data[-5:] + train_n_mode + "\t" + test_n_type + "\t"+"sigma="+str(sigma)+"\t"+"noise_shape=" )
    f.writelines("\n------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()