import utils.utils as utils
import pandas as pd
import numpy as np

from datasets.load_data import data_loader
#from model import *
from module import *


__all__ = ['complexity', 'fish_ratio', 'image_complexity']

# 连接mongodb并且获取数据
# my_mongo = utils.Mongo()
# data_frame = my_mongo.get_collection("resnet_fashion_mnist_clean", 'epoch_2')
# data_frame = my_mongo.get_df(data_frame)


# data_frame = data_frame.drop(["_id", "cross_entropy", "predict"], axis=1)       # pandas删除某一列，1.del df["columns"]，改变原始数据，2.df.drop('columns', axis=1)，不改变原始数据
# feature_ls = data_frame.columns.values
# feature_ls = feature_ls[:-1]            # 存放特征名的list

# class_n = 10               # 分类问题的类别数
class_dict = dict()  # 创建空字典分别存放各个类


# 使用ovo策略计算，若有n个类别，则有n*(n-1)/2个子问题，其中参数data表示数据框，class_num表示类别数
def complexity(data_frame, class_num):
    data_frame = data_frame.drop(["_id", "cross_entropy", "predict"],
                                 axis=1)  # pandas删除某一列，1.del df["columns"]，改变原始数据，2.df.drop('columns', axis=1)，不改变原始数据
    feature_ls = data_frame.columns.values
    feature_ls = feature_ls[:-1]  # 存放特征名的list

    for i in range(class_num):
        cs = f"class_{i}"
        class_dict[cs] = data_frame.loc[data_frame.ground_truth == i]

    class_ls = list(class_dict.keys())
    complexity_num = class_num * (class_num - 1) / 2
    complexity_ls = np.zeros(int(complexity_num))
    index = 0

    # 遍历类
    for ith in range(class_num):
        compare_1 = class_ls[ith]
        csf_0 = class_dict[compare_1]
        mean_0 = csf_0.mean(axis=0)
        std_0 = csf_0.std(axis=0)
        # 遍历除了该类的其他类
        for jth in range(ith + 1, class_num):
            compare_2 = class_ls[jth]
            csf_1 = class_dict[compare_2]
            mean_1 = csf_1.mean(axis=0)
            std_1 = csf_1.std(axis=0)
            # 计算fisher判别率
            fish_dict = dict()
            for kth in feature_ls:
                fish_dict[kth] = fish_ratio(mean_0[kth], mean_1[kth], std_0[kth], std_1[kth])
            fish_arr = np.array(list(fish_dict.values()))
            complexity_ls[index] = 1 / np.max(fish_arr[:-1])
            index += 1
    # print(complexity_ls)
    return np.mean(complexity_ls)


# fisher判别率计算公式
def fish_ratio(mean_1, mean_2, std_1, std_2):
    up = np.square(mean_1 - mean_2)
    down = std_1 + std_2
    return up / (down + 1e-14)


# Computing image complexity
class image_complexity:

    def __init__(self, ft):
        self.fun = ft

    def G0(self, x):
        fuzzy_pixel = x * np.log(x + 1e-12) + (1 - x) * np.log(1 - x + 1e-12)
        return -np.sum(fuzzy_pixel, axis=1) / (x.shape[1] * np.log(2))

    def G1(self, x):
        def low(xi):
            return xi * (np.exp(1 - xi) - np.exp(xi - 1))

        def high(xi):
            return (1 - xi) * (np.exp(xi) - np.exp(-xi))

        yita = np.where(x < 0.5, low(x), high(x))
        return np.sum(yita, axis=1) * 2 * np.sqrt(np.e) / (x.shape[1] * (np.e - 1))

    def G2(self, x):
        return np.sum(x * (1 - x), axis=1) * 4 / x.shape[1]

    def __call__(self, x):
        x = x.cpu().detach().numpy()
        x = x.reshape(x.shape[0], -1)
        if self.fun.lower() == 'g0':
            return self.G0(x)
        elif self.fun.lower() == 'g1':
            return self.G1(x)
        elif self.fun.lower() == 'g2':
            return self.G2(x)


if __name__ == '__main__':
    g0 = image_complexity('g0')
    data_selection = data_loader()

    # Print the information of dataset
    train_dataloader, test_dataloader = data_selection('mnist')
    data, label = next(iter(test_dataloader))
    for data, label in test_dataloader:
        img_fuzz = g0(data)
        print(max(img_fuzz))

    # Print the information of model
    # backbone = Lenet(input_cl=1)
    # for name, params in backbone.named_parameters():
    #     print(f"Shape of {name} : {params.shape}")

    # embedding = backbone(data)
    # print(embedding.shape)
    #
    # backbone.fc = nn.Linear(84, 10)
    # output = backbone.fc(embedding)
    # print(output.shape)
    # entropy = - F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
    # entropy = torch.sum(entropy, dim=1)
    # entropy = entropy.cpu().detach().numpy()
    # print(entropy)

