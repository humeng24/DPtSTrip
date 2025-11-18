from module import *
from tqdm import tqdm
from datasets.load_dataset import data_loader
from calculation.complexity import *


save_path = '/home/suxw/save_data/mnist/'


class new_dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.label[index]

    def __len__(self):
        return len(self.label)


# 过滤测试集数据
def filter(loader_1, loader_2):

    def calculte_sum(loader):
        data_sum = []
        labels = []
        for data, target in loader:
            data_sum.append(torch.sum(torch.flatten(data, start_dim=1), axis=1))
            labels.append(target)
        data_sum = np.concatenate(data_sum)
        labels = np.concatenate(labels)
        return pd.DataFrame({'img_sum': data_sum, 'label': labels})
    
    df_1 = calculte_sum(loader_1)
    df_2 = calculte_sum(loader_2)
    return pd.concat([df_1, df_2, df_2]).drop_duplicates(keep=False)


def save_data(index, dataset, name, path):
    datas = []
    targets = []
    for idx in index:
        data, target = dataset[idx]
        datas.append(data.cpu().numpy())
        targets.append(target)
    datas = np.array(datas)
    targets = np.array(targets)
    np.save(os.path.join(path, name + '_data.npy'), datas)
    np.save(os.path.join(path, name + '_label.npy'), targets)


def main():
    """
    将qmnist中测试集多出的数据也加进训练集做分析,先计算每张图片的图像复杂度
    :return: 
    """
    data_selection = data_loader()
    g = image_complexity('g1')
    
    # Dataloader of qmnist without normalization
    qmnist_trainloader, qminst_testloader = data_selection('qmnist')
    
    # Dataloader of mnist without normalization
    mnist_trainloader, mnist_testloader = data_selection('mnist')

    # ------------------------------------------------------------- #
    # --------------------- make dataset -------------------------- #
    # ------------------------------------------------------------- #

    df_filter = filter(qminst_testloader, mnist_testloader)
    print(f"Total number of train data in qmnist: {len(df_filter)}")
    print(list(df_filter.index)[:10])

    mnist_train_dataset = mnist_trainloader.dataset
    qmnist_test_dataset = qminst_testloader.dataset

    add_data = []
    add_labels = []
    for idx in tqdm(list(df_filter.index)):
        data, label = qmnist_test_dataset[idx]
        add_data.append(data)
        add_labels.append(label)
    add_data = torch.cat(add_data)
    s = add_data.shape
    add_data = torch.reshape(add_data, (s[0], 1, s[1], s[2]))
    print(add_data.shape)
    print(add_labels[:10])

    add_dataset = new_dataset(add_data, add_labels)
    mnist_train_pro = mnist_train_dataset + add_dataset
    print(f"Total number of train data in mnist: {len(mnist_train_pro)}")

    # --------------------------------------------------------------- #
    # ---------------------analysis dataset-------------------------- #
    # --------------------------------------------------------------- #

    gs = []
    labels = []
    mnist_train_dataloader_pro = torch.utils.data.DataLoader(mnist_train_pro, batch_size=128)
    for data, target in mnist_train_dataloader_pro:
        gs.append(g(data))
        labels.append(target)
    gs = np.concatenate(gs)
    labels = np.concatenate(labels)

    high_index = []
    low_index = []
    df = pd.DataFrame({'img_fuzz': gs, 'label': labels})
    df = df.sort_values(['img_fuzz'])
    df_group = df.groupby(by=['label'])
    for i, dataframe in df_group:
        low_index.append(np.array(dataframe.iloc[:5000, :].index))
        high_index.append(np.array(dataframe.iloc[len(dataframe)-5000:len(dataframe), :].index))
    # --------------------------------------------------------------- #
    # ------------------------save dataset--------------------------- #
    # --------------------------------------------------------------- #
    high_index = np.concatenate(high_index)
    low_index = np.concatenate(low_index)
    save_data(high_index, mnist_train_pro, 'high', save_path)
    save_data(low_index, mnist_train_pro, 'low', save_path)


if __name__ == '__main__':
    main()

