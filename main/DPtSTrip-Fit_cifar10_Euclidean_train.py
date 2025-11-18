import os
import advertorch.attacks as attacks
from module import *
from calculation.complexity import *
import utils.utils as utils
from tqdm import tqdm
from adver.adver_attack.attack import attack
from datasets.svhn_load_dataset import data_loader
#from datasets.sxw_load_dataset import data_loader
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch_examples.utils import TRAINED_MODEL_PATH
from sxwresnet import *
from torch.utils.tensorboard import SummaryWriter


# 超参数
lr = 0.01
momentum = 0.9

epochs = 10
seed = 6666
start_epoch = 1

# 正则化系数
use_norm = False
lamtha_1 = 1e-4
lamtha_2 = 1e-4

dataset = 'cifar10'               # mnist | fashion_mnist | emnist | cifar10 | cifar100
model_s = 'resnet'               # lenet | resnet | vgg11
img_complexity = 'g0'           # g0 | g1 | g2
attack_function = 'pgd'        # fgsm | pgd
use_normalization = False
cuda = 3

save_path=f"/data/hm/save_model_new/{dataset}/Euclidean/"
log_path = f"/data/hm/log_new/{dataset}/Euclidean/"
if(not os.path.exists(save_path)):
        os.makedirs(save_path)
if(not os.path.exists(log_path)):
        os.makedirs(log_path)
base_model_path = f"/data/hm/save_model/{dataset}/Euclidean/resnet_cifar10_512nodenum_0.01lr_32size_200epoch_6666seed_without_normalization_SGD1.pth"
# 网络设置
if 'mnist' in dataset:
    input_cl = 1
    img_size = 28
elif 'cifar' in dataset:
    input_cl = 3
    img_size = 32
#img_size = 64
node_num = 512

#---------------------用于计算给定特征向量与其他所有特征向量之间的相似度分数--------------------------------
#---------------------------------不同类别样本的相似度得分--------------------------------------------
def compute_vector_dist_toall_except_own(embed_dict_class, data_dict):
    epsilon = 1e-7 #创建一个名为"episilon"的变量，赋值为1e-7。该值用于对计算中的除零错误进行处理。
    fea =data_dict['x4'] #从"data_dict"中获取名为"x4"的特征向量，并将其赋值给变量"fea"
    embed =embed_dict_class['x4'] #从"embed_dict_class"中获取名为"x4"的特征向量，并将其赋值给变量"embed"
    fea = fea / (torch.norm(fea, dim=1, keepdim=True) + epsilon)#对特征向量"fea"进行归一化处理，除以每个向量的模长（即欧氏距离）
                                                                # 此步骤是为了将特征向量转换为单位向量，便于计算余弦相似度
    fea = fea.to(device)

    embed = embed / (torch.norm(embed, dim=1, keepdim=True) + epsilon)
    embed = torch.transpose(embed , 0, 1) #将"embed"转置，变为列向量的形式（fealen * total_size）
    embed = embed.to(device)

    fea_label = data_dict['label'] #batch * 1        #从"data_dict"中获取名为"label"的标签，并将其赋值给变量"fea_label"
    #fea_label = torch.unsqueeze(torch.tensor(fea_label), 1)  #对"fea_label"进行扩展，使得其维度变为(batch_size, 1)
    fea_label = torch.unsqueeze(fea_label, 1)
    emb_label =embed_dict_class['label']
    #emb_label = torch.unsqueeze(torch.tensor(emb_label), 1)
    emb_label = torch.unsqueeze(emb_label, 1)
    emb_label = torch.transpose(emb_label, 0, 1)  # 1* batch  #对"emb_label"进行扩展，使得其维度变为(1, batch_size)
    #print("emb_label", emb_label.shape, fea_label.shape)


    #similar_score = np.dot(fea, embed)  ##batch * total_size
    similar_score = torch.mm(fea, embed)#计算"fea"和"embed"之间的内积，得到一个相似度矩阵"similar_score"，大小为(batch_size * total_size)

    fea_label_replicate = fea_label.repeat(1, emb_label.shape[0])
    embed_label_replicate = emb_label.repeat(fea_label.shape[0], 1)



    #创建一个名为"mask_same"的变量，赋值为"fea_label_replicate"与 "embed_label_replicate"是否相等的布尔矩阵
    mask_same = fea_label_replicate == embed_label_replicate


#将"mask_same"取反，得到一个布尔矩阵，该矩阵表示不同类别的特征向量对
#将"similar_score"与"mask_same"进行元素级别的乘法操作，得到不同类别的特征向量对的相似度矩阵"similar_score_ofdif_class"

    similar_score_of_diff_class = torch.logical_not(mask_same) * similar_score
    return similar_score_of_diff_class

#----------------------------------相同类别相似度得分---------------------------------------------
def compute_same_vector_dist_toall_except_own(embed_dict_class, data_dict):
    epsilon = 1e-7 #创建一个名为"episilon"的变量，赋值为1e-7。该值用于对计算中的除零错误进行处理。
    fea_pos =data_dict['x4'] #从"data_dict"中获取名为"x4"的特征向量，并将其赋值给变量"fea"
    embed_pos =embed_dict_class['x4'] #从"embed_dict_class"中获取名为"x4"的特征向量，并将其赋值给变量"embed"
    fea_pos = fea_pos / (torch.norm(fea_pos, dim=1, keepdim=True) + epsilon)#对特征向量"fea"进行归一化处理，除以每个向量的模长（即欧氏距离）
                                                                # 此步骤是为了将特征向量转换为单位向量，便于计算余弦相似度
    fea_pos = fea_pos.to(device)

    embed_pos = embed_pos / (torch.norm(embed_pos, dim=1, keepdim=True) + epsilon)
    embed_pos = torch.transpose(embed_pos , 0, 1) #将"embed"转置，变为列向量的形式（fealen * total_size）
    embed_pos = embed_pos.to(device)

    fea_label_pos = data_dict['label'] #batch * 1        #从"data_dict"中获取名为"label"的标签，并将其赋值给变量"fea_label"
    #fea_label = torch.unsqueeze(torch.tensor(fea_label), 1)  #对"fea_label"进行扩展，使得其维度变为(batch_size, 1)
    fea_label_pos = torch.unsqueeze(fea_label_pos, 1)
    emb_label_pos =embed_dict_class['label']
    #emb_label = torch.unsqueeze(torch.tensor(emb_label), 1)
    emb_label_pos = torch.unsqueeze(emb_label_pos, 1)
    emb_label_pos = torch.transpose(emb_label_pos, 0, 1)  # 1* batch  #对"emb_label"进行扩展，使得其维度变为(1, batch_size)
    #print("emb_label", emb_label.shape, fea_label.shape)


    #similar_score = np.dot(fea, embed)  ##batch * total_size
    similar_score_pos = torch.mm(fea_pos, embed_pos)#计算"fea"和"embed"之间的内积，得到一个相似度矩阵"similar_score"，大小为(batch_size * total_size)
    #similar_score = torch.matmul(fea, embed)
   #通过将"fea_label"进行复制，使其维度变为(batch_size * total_size)，用于与"embed_label"比较是否相同。这是为了剔除特征向量与自身的比较
    #fea_label_replicate = fea_label.repeat(1, emb_label.shape[0])
    #embed_label_replicate = emb_label.repeat(1, fea_label.shape[0])
    #fea_label_replicate = fea_label.unsqueeze(1).repeat(1, emb_label.shape[0], 1)
    #embed_label_replicate = emb_label.repeat(fea_label.shape[0], 1)
    fea_label_replicate_pos = fea_label_pos.repeat(1, emb_label_pos.shape[0])
    embed_label_replicate_pos = emb_label_pos.repeat(fea_label_pos.shape[0], 1)



    #创建一个名为"mask_same"的变量，赋值为"fea_label_replicate"与 "embed_label_replicate"是否相等的布尔矩阵
    mask_same_pos = fea_label_replicate_pos == embed_label_replicate_pos


#将"mask_same"取反，得到一个布尔矩阵，该矩阵表示不同类别的特征向量对
#将"similar_score"与"mask_same"进行元素级别的乘法操作，得到不同类别的特征向量对的相似度矩阵"similar_score_ofdif_class"

    similar_score_of_diff_class_pos = mask_same_pos * similar_score_pos
    return similar_score_of_diff_class_pos

#-----------------------------------根据索引寻找样本-----------------------------------------------
def get_k_mask(input, num_k):
    ones = torch.ones(input.shape[1])#创建一个长度为 input 矩阵的列数的一维数组，数组中的元素都为 1
    diag_matrix = torch.diag(ones) #使用 np.diag() 函数将数组 ones 转换为对角矩阵

    ratio = num_k * 1.0 / input.shape[1] #计算 num_k 占矩阵列数的比例

#创建一个与 input 矩阵形状相同的随机矩阵。该随机矩阵中的元素是在0到1之间均匀分布的随机值，然后与 ratio 进行比较，生成布尔值矩阵
    mask = torch.rand(input.shape[0], input.shape[1]) < ratio
    mask = mask.float() #将布尔值矩阵转换为整数型矩阵
    mask = mask + diag_matrix #将整数型的 mask 矩阵与对角矩阵 diag_matrix 相加，实现保留部分对角元素的目的

    mask = mask > 0  #将矩阵中元素大于0的位置设置为True，其余位置设置为False
    mask = mask.float() #将布尔值矩阵转换为浮点型矩阵
    mask = mask.to(device)
    return mask


#-----------------------------------欧式距离计算----------------------------------

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    # 转换为单位向量
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    ## xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    ## torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    #dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    ## clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #dist = dist.clamp(min=1e-12,max=1e+12)
    return dist

#--------------------------triplet_loss-----------------------------
import torch
import torch.nn.functional as F

def triplet_loss(a, p, n,  regularize=0, margin=0.9):
    positive_dist, negative_dist = None, None

    norm_a = F.normalize(a, dim=1) # l2_norm
    norm_p = F.normalize(p, dim=1)
    norm_n = F.normalize( n, dim=1)

    positive_dist = 1 - torch.abs(torch.sum(norm_a * norm_p, dim=1))
    negative_dist = 1 - torch.abs(torch.sum(norm_a * norm_n, dim=1))

    triplet_loss= torch.mean(torch.max(positive_dist - negative_dist + margin, torch.tensor(0.0)))
    positive_dist = torch.mean(positive_dist)
    negative_dist = torch.mean(negative_dist)
    norm = torch.mean(a * a + p* p+ n * n)
    triplet_loss= triplet_loss + regularize * norm
    #return triplet_loss,positive_dist, negative_dist
    return triplet_loss

#----------------------------余弦距离-----------------------------------
def cosine_similarity(x1, x2):
    # 转换为单位向量
    normalized_vector1 = torch.nn.functional.normalize(x1,  dim=-1)
    normalized_vector2 = torch.nn.functional.normalize(x2, dim=-1)
    #similarity = F.cosine_similarity(normalized_vector1.unsqueeze(1), normalized_vector2.unsqueeze(0), dim=2)
    cose=torch.mm(normalized_vector1,normalized_vector2.t())

    return 1-cose

#---------------------------------标签平滑-----------------------
def softmax_crossentropy_labelsmooth(pred, targets, lb_smooth=None):
    if lb_smooth:
        eps = lb_smooth
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, targets.unsqueeze(1), 1)
        one_hot = one_hot*(1-eps)+(1-one_hot)*eps/(n_class - 1)
        log_prb = F.log_softmax(pred, dim = 1)
        loss = -(one_hot*log_prb).sum(dim=1)
        loss = loss.mean()
    else:
        loss = F.cross_entropy(pred, targets)
    return loss
#--------------------------------L2正则化------------------------
def L2_norm(logit):
    l2norm = logit.mul(logit) #.sum(1)
    return l2norm.mean() #.sqrt().mean()

#

#定义Resnet18
# 定义残差块ResBlock
class ResBlock(nn.Module):
    #  初始化中定义了网络的两种组成部分left、shortcut
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层.left是正常的传播通路，shortcut是短路连接
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),  # padding=1:上下左右各填充一层0元素
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),  # 第一个out_channel为上一层的输出通道数
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构 F(x)+shortcut(x)
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet_18(nn.Module):  # 18：17个卷积层，1个全连接层
    def __init__(self, ResBlock, input_cl):
        super(ResNet_18, self).__init__()

        self.in_channel = 16  # 可更改
        # ResNet18结构
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_cl, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, 2)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)


    def forward(self, x):
        # 数据流流动（前向传播）
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)  # view:将张量铺平，（行，列），-1表示不对这一维度（列）的数量做限定，算出来是多少就是多少，注意在所有维度中只能有一个维度指定为-1
        # x = self.fc(x)  # 输出层 在训练模型时再添加
        return x

#--------------------------------自定义网络----------------------------
# 自定义网络
class net(nn.Module):
    def __init__(self, backbone):
        super(net, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features=node_num, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x
        x = self.fc(x)
        return x



#------------------------------------训练-------------------------------
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

#----------------改-----------------------------------------------------------
        # 获取导数第二层特征
        clndata_feature=backbone_model(data)
        clndata_output = model(data)

        #获取导数第二层特征

        adv_feature=backbone_model(advdata)
        adv_output= model(advdata)  # 把数据输入网络并得到输出，即进行前向传播

        xent_loss = F.cross_entropy(adv_output, target)  # 交叉熵损失函数
        xent_loss1 = F.cross_entropy(clndata_output, target)  # 交叉熵损失函数


        #余弦距离
        #similarity = cosine_similarity(adv_feature, clndata_feature)
        # p_p = cosine_similarity(clndata_feature, clndata_feature)
        # positive_similarity = similarity[target.eq(target.view(-1, 1))].sum()
        # negative_similarity = similarity[target.ne(target.view(-1, 1))].sum()

        #欧式距离
        similarity =euclidean_dist(adv_feature, clndata_feature)

        # 根据标签计算正样本和负样本的欧式距离
        positive_similarity = similarity[target.eq(target.view(-1, 1))] #对抗样本与正样本之间的距离
        negative_similarity = similarity[target.ne(target.view(-1, 1))] #对抗样本与负样本之间的距离

        #采样
        ##只有那些与anchor点相似度大于正样本点最小相似度的负样本才应该包含在训练中
        #negative_similarity = negative_similarity1[negative_similarity1 + 0.00 < max(positive_similarity1)]

        # 只有那些与anchor点相似度小于具有最大相似度(最接近anchor点)的负样本点的正样本点才应该被包括在训练中
        #positive_similarity = positive_similarity1[positive_similarity1 - 0.00 > min(negative_similarity1)]

        #干净样本的欧式距离
        p_p = euclidean_dist(clndata_feature, clndata_feature)
        p_p_score = p_p[target.eq(target.view(-1, 1))]


        weights_pp = (positive_similarity + 1.0) ** 4
        weighted_distances_pp = positive_similarity * weights_pp

        pp_dist = torch.div(torch.sum(weighted_distances_pp, dim=0),
                            (torch.sum(weights_pp, dim=0) + 1e-6))
        pos_weight = torch.div(weights_pp, torch.sum(weights_pp, dim=0) + 1e-6)


        weights_nn = (negative_similarity + 1.0) ** (-8)
        weighted_distances_nn = negative_similarity * weights_nn

        neg_dist = torch.div(torch.sum(weighted_distances_nn, dim=0),
                                 (torch.sum(weights_nn, dim=0) + 1e-6))

        neg_weight = torch.div(weights_nn, torch.sum(weights_nn, dim=0) + 1e-6)


        margin = torch.tensor([2.5]).to('cuda:3')
        losses_vector_op = torch.sum(pp_dist - neg_dist + margin)

        my_loss = torch.mean(losses_vector_op )

        loss = xent_loss + 1.9*my_loss + 1.3*torch.mean(torch.max(p_p_score)) + xent_loss1
        train_loss += loss.item() * train_batch_size  # 计算训练误差

        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数

        pred = adv_output.data.max(1, keepdim=True)[1]  # 获取预测值
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 计算预测正确数

    train_loss /= len(train_loader.dataset)
    scheduler.step()

    print("Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        train_loss , correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))
    return train_loss, correct


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
            output= model(advdata)

        if draw_embedding:
            for key, value in model.backbone.feature_dict.items():
                writer.add_embedding(value, metadata=target, tag=key, global_step=i)

        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(target.view_as(pred)).sum().item()

    test_clnloss /= len(test_loader.dataset)
    test_advloss /= len(test_loader.dataset)

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
    return test_clnloss, clncorrect, test_advloss, advcorrect


def compute_fuzz(model, f=None, if_to_f=False):
    """
    :param model: 模型
    :param writer: tensorboard日志文件
    :param f: 本地日志文件
    :param if_to_f: 是否写进本地日志文件（默认为否）
    :return:
    """
    fuzz_dict = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            param = param.detach().cpu().numpy()
            param = param.reshape(1, -1)
            fuzz = utils.fuzziness(param)

            fuzz_dict[name] = fuzz

            if if_to_f:
                print_log("\nThe fuzziness of {} = {:.4f}".format(name, fuzz), f)
    return fuzz_dict


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into files
    with open(log_path, 'a+') as f:
        f.write(log_info + '\n')


if __name__ == '__main__':

    # my_mongo = Mongo()
    # my_db = f"{model_s}_{dataset}_clean"
    writer = SummaryWriter(f'/data/hm/drawing/{model_s}_{dataset}_SGD')       # 记录损失精度信息
    g = image_complexity(img_complexity)

    # 日志设置
    file_ls = os.walk(log_path).__next__()[2]
    if not file_ls:
        log_name = f'{model_s}_{dataset}_SGD_1.txt'
    else:
        pattern = re.compile("[_.]")
        file_ls = [int(pattern.split(file)[-2]) for file in file_ls]
        idx = max(file_ls) + 1
        log_name = f"{model_s}_{dataset}_SGD_{idx}.txt"

    log = os.path.join(log_path, log_name)
    os.mknod(log)

    # 数据集选择
    data_selection = data_loader()

    train_loader, test_loader = data_selection(dataset, normalize=use_normalization,size=img_size)
    print_log("-"*20 + " hyperparameters" + '-'*20, log)
    print_log(f"\nSeed: {seed}", log)
    print_log(f"\nEpochs: {epochs}", log)
    print_log("\n" + data_selection.type, log)
    print_log(f"\nLearning rate: {lr}", log)
    print_log(f"\nTrain batch size: {data_selection.train_batch}", log)
    print_log(f"\nTest batch size: {data_selection.test_batch}", log)
    print_log(f"\nSelection {img_complexity} for image complexity calculation", log)
    print_log(f"\nSelection {attack_function} attack function for adversarial train", log)

    train_batch_size = data_selection.train_batch

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')       # 启动GPU

    print_log(f"\nUsing {str(device)} for training\n", log)

    utils.setup_seed(seed)                                                      # 设置随机数种子

    print_log("-"*20 + " model params " + '-'*20, log)

    # 选择模型


    if model_s == 'resnet':
        backbone_model = ResNet_18(ResBlock, input_cl=input_cl)
        node_num = 512

    else:
        raise ValueError("没有此模型")


    model = net(backbone_model)

    print_log("\n" + str(model) + "\n", log)
    model.load_state_dict(torch.load(base_model_path))

    model = model.to(device)




    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7], gamma=0.1)  # 学习率衰减  # learning rate decay

    if 'mnist' in dataset:
        eps = 0.3
    elif 'cifar' in dataset:
        eps = 12/255

    if attack_function == 'fgsm':
        adversary = attacks.FGSM(model, eps=eps)
    elif attack_function == 'pgd':
        adversary = attacks.LinfPGDAttack(model, eps=eps,nb_iter=50)

    print_log("-" * 20 + " Training process" + '-' * 20, log)

    print_log('\nTraining start!\n', log)

    accuracy = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        my_col = f"epoch_{epoch}"
        # feature_collection = my_mongo.get_collection(my_db, my_col)
        train_loss, train_acc = train(epoch)
        test_clnloss, test_clnacc, test_advloss, test_advacc = test()

        # 参数可视化
        for name, param in model.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, epoch)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_clnloss', test_clnloss, epoch)
        writer.add_scalar('test_clnacc', test_clnacc, epoch)
        writer.add_scalar('test_advloss', test_advloss, epoch)
        writer.add_scalar('test_advacc', test_advacc, epoch)



        print_log("-"*5 + f" epoch_{epoch} " + "-"*5, log)
        print_log(f"\nTrain loss = {train_loss}, train accuracy = {train_acc}", log)
        print_log(f"\nTest clean loss = {test_clnloss}, test clean accuracy = {test_clnacc}", log)
        print_log(f"\nTest adver loss = {test_advloss}, test adver accuracy = {test_advacc}\n", log)



        fuzz_dict = compute_fuzz(model)
        for k, v in fuzz_dict.items():
            writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)
            print(f"{k} : {v}")

        if test_advacc > accuracy:
            #模型的名字
            model_name = f"{model_s}_{dataset}_{node_num}nodenum_{lr}lr_{img_size}size_{seed}seed_{data_selection.type.split(' ')[1]}_normalization_SGD_wei_12_50_115.pth"
            accuracy = test_advacc
            torch.save(model.state_dict(), save_path + model_name)
            print_log(f"\nSaving model state dict in {save_path + model_name}\n", log)

    print_log(f"\nTraining done!\n", log)

    writer.close()
    torch.cuda.empty_cache()

    print_log("-" * 20 + " Result " + '-' * 20, log)
    # 记录精度最高的模型的各层模糊度
    print_log('\n'+ '-'*5 + f"Compute fuzziness of each layer" + '-'*5 + '\n', log)
    model.load_state_dict(torch.load(save_path + model_name))
    compute_fuzz(model, f=log, if_to_f=True)

    # 记录对抗样本准确率
    print_log('\n'+ '-'*5 + f"Compute adversarial accuracy of trained model" + '-'*5 + '\n', log)
    if 'mnist' in dataset:
        eps = 0.3
    elif 'cifar' in dataset:
        eps = 8/255

    print_log(f'\nEpsilon of attack: {eps}', log)

    # FGSM攻击
    print_log("-"*5 + f" Using FGSM attack " + "-"*5, log)
    fgsm = attacks.FGSM(model, eps=eps)
    attack(adver_method=fgsm, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using PGD attack " + "-" * 5, log)
    pgd = attacks.LinfPGDAttack(model, eps=eps)
    attack(adver_method=pgd, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using BIM attack " + "-" * 5, log)
    bim = attacks.LinfBasicIterativeAttack(model, eps=eps)
    attack(adver_method=bim, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using MIM attack " + "-" * 5, log)
    mim = attacks.MomentumIterativeAttack(model, eps=eps)
    attack(adver_method=mim, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using CW attack " + "-" * 5, log)
    cw = attacks.CarliniWagnerL2Attack(
        model, num_classes=10, confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9,
        max_iterations=1000, abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0, )
    attack(adver_method=cw, model=model, data_loader=test_loader, device=device, fp=log)