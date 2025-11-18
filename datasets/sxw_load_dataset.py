from tqdm import tqdm
from module import *
import matplotlib.pyplot as plt
from calculation.complexity import *


"""
      transforms.Resize((size, size)),
      transforms.RandomHorizontalFlip(),            # 水平翻转
      transforms.RandomVerticalFlip(),              # 竖直翻转
      transforms.RandomRotation(15),  # 旋转，范围-15-15
      transforms.RandomRotation([90, 180]),         # 从三个角度中挑选一个旋转
      transforms.Resize([28, 28]),                  # 缩放
      transforms.CenterCrop([28, 28]),              # 随机裁剪
      transforms.RandomCrop(size, padding=4),
"""


class data_loader(object):

    def __init__(self):
        self.exist_dataset = ['mnist', 'fashion_mnist', 'qmnist', 'cifar10','cifar100']
        self.path = {
            'mnist_path': '/data/suxw/sxuhome/image_data/mnist/',
            'fashion_path': '/data/suxw/sxuhome/image_data/fashion_mnist/',
            'qmnist': '/data/suxw/sxuhome/image_data/qmnist/',
            'cifar10_path': '/data/suxw/sxuhome/image_data/Cifar10/',
            'cifar100_path': '/data/suxw/sxuhome/image_data/Cifar100/'
        }

    def __call__(self, data_name=None, data_dir=None, download=False, size=None, train_batch=None, test_batch=None, normalize=False):

        """
        :param data_name: 选择数据集
        :param data_dir: 外部数据集
        :param download: 是否下载（默认为否）
        :param size: 图像大小
        :param train_batch: 训练集batch大小
        :param test_batch: 测试集batch大小
        :param normalize: 是否对输入数据进行标准化（默认为否）
        """

        self.normalize = normalize

        if data_name is None and data_dir is None:
            raise ValueError

        if data_name not in self.exist_dataset:
            raise ValueError

        if data_name:
            train_loader, test_loader = self.__load(data_name, download, size, train_batch, test_batch)

        if data_dir:
            self.data_dir = data_dir

        if self.normalize:
            self.type = f"{data_name} with normalization"
        else:
            self.type = f"{data_name} without normalization"

        return train_loader, test_loader

    def __load(self, name, download, size, train_batch, test_batch):

        if train_batch is None:
            train_batch = 128

        if test_batch is None:
            test_batch = 1000

        self.train_batch = train_batch
        self.test_batch = test_batch

        if name == 'mnist':
            if size is None:
                size = 28
            train, test = self.__load_mnist(download, size, train_batch, test_batch)
        elif name == 'fashion_mnist':
            if size is None:
                size = 28
            train, test = self.__load_fashion_mnist(download, size, train_batch, test_batch)
        elif name == 'qmnist':
            if size is None:
                size = 28
            train, test = self.__load_qmnist(download, size, train_batch, test_batch)
        elif name == 'cifar10':
            if size is None:
                size = 32
            train, test = self.__load_cifar10(download, size, train_batch, test_batch)
        elif name =='cifar100':
            if size is None:
                size = 32
            train, test = self.__load_cifar100(download, size, train_batch, test_batch)

        return train, test

    def __load_mnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                               transforms.Resize((size, size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                               transforms.Resize((size, size)),
                               transforms.ToTensor(),
                           ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.path['mnist_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.path['mnist_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    def __load_fashion_mnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                      transforms.Resize((size, size)),
                                      transforms.RandomHorizontalFlip(),            # 水平翻转
                                      transforms.RandomRotation(15),  # 旋转，范围-15-15
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                  ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                                      transforms.Resize((size, size)),
                                      transforms.RandomHorizontalFlip(),            # 水平翻转
                                      transforms.RandomRotation(15),  # 旋转，范围-15-15
                                      transforms.ToTensor(),
                                  ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    def __load_qmnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.QMNIST(self.path['mnist_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.QMNIST(self.path['mnist_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    def __load_cifar10(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader


    def __load_cifar100(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader


if __name__ == '__main__':

    data_path = '/data/suxw/sxuhome/save_data/mnist/'

    g = image_complexity('g2')
    gs_all = []
    labels_all = []

    gs_test = []
    labels_test = []

    data_selection = data_loader()
    qtrainloader, qtestloader = data_selection('qmnist')
    trainloader, testloader = data_selection('mnist')

    qtrain_dataset = qtrainloader.dataset
    qtest_dataset = qtestloader.dataset
    all_dataset = qtrain_dataset + qtest_dataset
    all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=128, shuffle=False)

    for data, target in qtestloader:
        gs_all.append(torch.sum(torch.flatten(data, start_dim=1), axis=1))
        labels_all.append(target)

    for data, target in testloader:
        gs_test.append(torch.sum(torch.flatten(data, start_dim=1), axis=1))
        labels_test.append(target)

    gs_all = np.concatenate(gs_all)
    labels_all = np.concatenate(labels_all)

    gs_test = np.concatenate(gs_test)
    labels_test = np.concatenate(labels_test)

    df_all = pd.DataFrame({'img_fuzz': gs_all, 'label': labels_all})
    df_test = pd.DataFrame({'img_fuzz': gs_test, 'label': labels_test})

    set_diff_df = pd.concat([df_all, df_test, df_test]).drop_duplicates(keep=False)
    df = set_diff_df.sort_values(by=['img_fuzz'])
    df_label = df.groupby(['label'])
    for i, d in df_label:
        print(d)
    data = []
    labels = []

    # for i, (x, y) in tqdm(enumerate(qtest_dataset), total=len(qtest_dataset)):
    #     if i in list(set_diff_df.index):
    #         data.append(np.array(x))
    #         labels.append(y)
    #
    # for i, (x, y) in tqdm(enumerate(trainloader.dataset), total=len(trainloader.dataset)):
    #     data.append(np.array(x))
    #     labels.append(y)
    #
    # data = np.array(data)
    # labels = np.array(labels)
    #
    # print(f"Total number of data: {len(data)}")
    # np.save(os.path.join(data_path, 'mnist.npy'), data)
    # np.save(os.path.join(data_path, 'label.npy'), labels)

    # trans_to_pil = transforms.ToPILImage()
    # print(np.array(trans_to_pil(data_first[0][0])))

    # train_dataset = datasets.QMNIST('/data/khp/data/image_data/mnist/', train=True, download=True,
    #                                 transform=transforms.Compose([transforms.ToTensor()]))
    # test_dataset = datasets.QMNIST('/data/khp/data/image_data/mnist/', train=False, download=True,
    #                                transform=transforms.Compose([transforms.ToTensor()]))
