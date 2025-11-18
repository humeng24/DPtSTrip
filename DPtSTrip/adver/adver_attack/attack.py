from module import *
from torchvision import datasets, transforms
#import configuration.configuration as cfg
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from PIL import Image
import pylab

from torch.utils.tensorboard import SummaryWriter


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into files
    with open(log_path, 'a+') as f:
        f.write(log_info + '\n')


def attack(adver_method, model, data_loader, device, fp=None, flag_adv=True, draw_adv=False):
    """
    对抗攻击
    :param draw_adv: 显示对抗样本
    :param adver_method: advertorch中的对抗攻击方法
    :param model: 训练好的模型
    :param data_loader: 数据迭代器
    :param flag_adv: 是否进行对抗攻击，True时攻击
    :return:
    """
    H = 0
    H_adv = 0
    test_clnloss = 0
    clncorrect = 0
    robustness = 0

    if flag_adv:
        test_advloss = 0
        advcorrect = 0

    model.eval()

    for clndata, target in data_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)

        # 计算输出熵
        H_i = - F.softmax(output, 1) * F.log_softmax(output, 1)
        H_i = H_i.detach().cpu().numpy()
        H_batch = np.sum(np.sum(H_i, 1))
        H += H_batch

        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        clnpred = output.max(1, keepdim=True)[1]
        clncorrect += clnpred.eq(target.view_as(clnpred)).sum().item()

        if flag_adv:
            advdata = adver_method.perturb(clndata)

            if draw_adv:
                advdata_n = advdata / 2 + 0.5
                grid = make_grid(advdata_n)
                tb.add_image('adv images', grid)

            with torch.no_grad():
                output_adv = model(advdata)
            H_adv_i = - F.softmax(output_adv, 1) * F.log_softmax(output_adv, 1)
            H_adv_i = H_adv_i.detach().cpu().numpy()
            H_adv_batch = np.sum(np.sum(H_adv_i, 1))
            H_adv += H_adv_batch

            test_advloss += F.cross_entropy(
                output_adv, target, reduction='sum').item()
            advpred = output_adv.max(1, keepdim=True)[1]
            advcorrect += advpred.eq(target.view_as(advpred)).sum().item()
            robustness += advpred.eq(clnpred).sum().item()

    if draw_adv:
        plt.figure(1)
        clndata = clndata / 2 + 0.5
        img_clean = clndata[0].cpu().detach().numpy()
        img_clean = img_clean.transpose(1, 2, 0)
        plt.title('clean image')
        plt.imshow(img_clean, cmap='gray')

        plt.figure(2)
        advdata = advdata / 2 + 0.5
        img_adv = advdata[0].cpu().detach().numpy()
        img_adv = img_adv.transpose(1, 2, 0)
        plt.title('adv image')
        plt.imshow(img_adv, cmap='gray')

        plt.show()

    test_clnloss /= len(data_loader.dataset)

    H /= len(data_loader.dataset)
    H_adv /= len(data_loader.dataset)

    if fp:
        print_log('\nTest set: avg cln loss: {:.4f},'
                      ' cln acc: {}/{} ({:.0f}%)\n'.format(
            test_clnloss, clncorrect, len(data_loader.dataset),
            100. * clncorrect / len(data_loader.dataset)), fp)

        print_log('Test set: avg adv loss: {:.4f},'
                      ' adv acc: {}/{} ({:.0f}%)\n'.format(
            test_advloss, advcorrect, len(data_loader.dataset),
            100. * advcorrect / len(data_loader.dataset)), fp)

        print_log('Test set: avg adv loss: {:.4f},'
                  ' adv robustness: {}/{} ({:.0f}%)\n'.format(
            test_advloss, robustness, len(data_loader.dataset),
            100. * robustness / len(data_loader.dataset)), fp)

        print_log(f'\nClean test entropy = {H}\n', fp)

    return advcorrect / len(data_loader.dataset)
