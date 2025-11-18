import numpy as np
import pandas as pd
from calculation.complexity import *
import utils.utils as utils
import matplotlib.pyplot as plt


my_mongo = utils.Mongo()
# complexity_ls = np.zeros(40)
#
# for i, j in enumerate(range(1, 41)):
#     data_frame = my_mongo.get_collection("resnet_fashion_mnist_clean", f'epoch_{j}')
#     data_frame = my_mongo.get_df(data_frame)
#
#     class_n = 10               # 分类问题的类别数
#     class_dict = dict()        # 创建空字典分别存放各个类
#
#     result = complexity(data_frame, class_n)
#     complexity_ls[i] = result
#     print(f"{j} complexity = {result}")
#
# plt.figure(1)
# plt.plot(range(1, 41), complexity_ls)
# plt.show()

db_name = 'fashion_mnist_Embedding'
my_db = my_mongo.get_database(db_name)
collections = my_db.list_collection_names()
for c in collections:
    data_frame = my_db[c]
    data_frame = my_mongo.get_df(data_frame)

    class_n = 10
    class_dict = dict()

    result = complexity(data_frame, class_n)
    print(f"The complexity of {c} = {result}")
