from tqdm import tqdm
from module import *

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import LinfPGDAttack

import utils.utils as utils
import configuration.configuration as cfg

from datasets.load_dataset import data_loader
from torch.utils.tensorboard import SummaryWriter


seed = 6666
lr = 0.0001
momentum = 0.09
batch_size = 128
nb_epoch = 200
dataset = 'fashion_mnist'  # mnist | fashion_mnist | cifar10 | cifar100
model_s = 'resnet'
path = f'/home/suxw/save_model/{dataset}/'

embedding_num = 84


class net(nn.Module):
    def __init__(self, backbone):
        super(net, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embedding_num, 10)

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x
        x = self.fc(x)
        return x


def train(epoch):
    model.train()  # 设置为训练模式
    train_loss = 0  # 初始化训练损失为0
    correct = 0  # 初始化预测正确个数为0
    for data, target in tqdm(train_loader, desc=f"epoch_{epoch}"):
        data = data.to(device)
        target = target.to(device)

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零

        with ctx_noparamgrad_and_eval(model):
            advdata = adversary.perturb(data, target)  # 生成对抗样本

        output = model(advdata)  # 把数据输入网络并得到输出，即进行前向传播
        loss = F.cross_entropy(output, target)  # 交叉熵损失函数
        train_loss += loss.item() * batch_size  # 计算训练误差
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数
        pred = output.data.max(1, keepdim=True)[1]  # 获取预测值
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 计算预测正确数

    print("Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))


def test(draw_embedding=False):
    model.eval()

    test_clnloss = 0
    clncorrect = 0

    test_advloss = 0
    advcorrect = 0

    for i, (clndata, target) in enumerate(test_loader):

        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        advdata = adversary.perturb(clndata, target)
        with torch.no_grad():
            output = model(advdata)

        if draw_embedding:
            for key, value in model.backbone.feature_dict.items():
                writer.add_embedding(value, metadata=target, tag=key, global_step=i)

        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(target.view_as(pred)).sum().item()

    test_clnloss /= len(test_loader.dataset)

    # 干净样本准确率
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)'.format(
          test_clnloss, clncorrect, len(test_loader.dataset),
          100. * clncorrect / len(test_loader.dataset)))
    test_advloss /= len(test_loader.dataset)

    # 对抗样本准确率
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        test_advloss, advcorrect, len(test_loader.dataset),
        100. * advcorrect / len(test_loader.dataset)))


if __name__ == '__main__':

    writer = SummaryWriter(f'./{dataset}_adver_train')
    data_selection = data_loader()

    # 数据集选择
    if dataset == 'mnist':
        _, test_loader = data_selection('mnist', size=size)
    elif dataset == 'fashion_mnist':
        _, test_loader = data_selection('fashion_mnist', size=size)
    elif dataset == 'cifar10':
        _, test_loader = data_selection('cifar10', size=size)
    elif dataset == 'cifar100':
        _, test_loader = data_selection('cifar100', size=size)
    else:
        raise ValueError("没有此数据集")

    utils.setup_seed(seed)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model_s = 'resnet_fashion_mnist.pth'
    model_path = f'/home/suxw/save_model/{dataset}/{model_s}'

    model = torch.load(model_path)  # 读取模型
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=0.55,
        nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    test()

    for epoch in range(nb_epoch):
        train(epoch)
        test()

        # 参数可视化
        for name, param in model.named_parameters():
            if 'bn' not in name:
                # writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name, param, epoch)

    # test(draw_embedding=True)
    writer.close()

    model_name = f"{model_s[:-4]}_advtrain.pth"
    save_path = path + model_name

    # torch.save(obj=model.state_dict(), f=f'{save_path}')
    torch.save(model, save_path)  # 保存模型
