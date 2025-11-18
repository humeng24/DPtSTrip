from module import *
import matplotlib.pyplot as plt

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
        self.exist_dataset = ['mnist', 'fashion_mnist', 'cifar10']
        self.path = {
            'mnist_path': '/home/suxw/data/image_data/mnist/',
            'fashion_path': '/home/suxw/data/image_data/fashion_mnist/',
            'cifar10_path': '/home/suxw/data/image_data/Cifar10/'
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
        elif name == 'cifar10':
            if size is None:
                size = 32
            train, test = self.__load_cifar10(download, size, train_batch, test_batch)

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


if __name__ == '__main__':
    data_selection = data_loader()
    trainloader, testloader = data_selection('cifar10')

    data_first = next(iter(trainloader))
    data_numpy = data_first[0][0].cpu().detach().numpy()

    print(data_numpy)
    data_numpy = data_numpy.transpose(1, 2, 0)
    plt.imshow(data_numpy)
    plt.show()

    # train_dataset = datasets.QMNIST('/data/khp/data/image_data/mnist/', train=True, download=True,
    #                                 transform=transforms.Compose([transforms.ToTensor()]))
    # test_dataset = datasets.QMNIST('/data/khp/data/image_data/mnist/', train=False, download=True,
    #                                transform=transforms.Compose([transforms.ToTensor()]))
