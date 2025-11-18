import random
from tqdm import tqdm
from module import *
import matplotlib.pyplot as plt
# from calculation.complexity import *  # 不能和complexity.py相互import
#autoaugument
""" This code is taken from https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py """
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class Get_Dataset_C10(torchvision.datasets.CIFAR10):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        image_auto2 = self.transform[1](image)
        return image_clean, image_auto1, image_auto2, target


class Get_Dataset_C100(torchvision.datasets.CIFAR100):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        image_auto2 = self.transform[1](image)
        return image_clean, image_auto1, image_auto2, target







#-------------------------------------------------


"""
      transforms.Resize((size, size)),              # (h,w) 调整图像尺寸，这样会改变图片的长宽比，但本身并没有发生裁切。注意img.size返回的是(w,h)
      transforms.RandomHorizontalFlip(),            # 水平翻转
      transforms.RandomVerticalFlip(),              # 竖直翻转
      transforms.RandomRotation(15),                # 旋转，范围-15-15
      transforms.RandomRotation([90, 180]),         # 从两个角度中挑选一个旋转
      transforms.Resize([28, 28]),                  # 缩放
      transforms.CenterCrop([28, 28]),              # 中心裁剪
      transforms.RandomCrop(size, padding=4),       # 随机裁剪 size:sequence or int，若为sequence,则为(h,w)，若为int，则(size,size)。padding:sequence or int，当为int时，上下左右均填充int个像素
"""




class data_loader(object):

    def __init__(self):
        self.exist_dataset = ['mnist', 'fashion_mnist', 'qmnist', 'cifar10', 'cifar100', 'svhn']
        self.path = {
            'mnist_path': '/data/suxw/sxuhome/image_data/mnist/',
            'fashion_path': '/data/suxw/sxuhome/image_data/fashion_mnist/',
            'qmnist_path': '/data/suxw/sxuhome/image_data/qmnist/',
            'cifar10_path': '/data/suxw/sxuhome/image_data/Cifar10/',
            'cifar100_path': '/data/suxw/sxuhome/image_data/Cifar100/',
            'svhn_path': '/data/suxw/sxuhome/image_data/svhn/'

        }

    def __call__(self, data_name=None, data_dir=None, download=False, size=None, train_batch=None, test_batch=None, normalize=False):

        """
        :param data_name: 选择数据集
        :param data_dir: 外部数据集
        :param download: 是否下载（默认为否）
        :param size: 图像大小（根据网络输入图像大小设置）
        :param train_batch: 训练集每个batch所含样本数目
        :param test_batch: 测试集每个batch所含样本数目
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
        elif name == 'cifar100':
            if size is None:
                size = 32
            train, test = self.__load_cifar100(download, size, train_batch, test_batch)
        elif name == 'svhn':
            if size is None:
                size = 32
            train, test = self.__load_svhn(download, size, train_batch, test_batch)

        return train, test

    def __load_mnist(self, download, size, train_batch_size, test_batch_size):  # train_batch_size:训练集中每个batch所含的样本数目

        if self.normalize:
            train_transform = transforms.Compose([
                               transforms.Resize((size, size)),  # 根据网络输入图像大小设置 得到的图像像素值的取值范围为0——1
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),  # 均值，标准差
                           ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL(Python Image Library).Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # 标准化，使模型更容易收敛
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


        train_loader = torch.utils.data.DataLoader(  # 若首次运行base_train，下一行的download应等于True
            datasets.MNIST(self.path['mnist_path'], train=True, download=False, transform=train_transform),
            batch_size=train_batch_size, shuffle=True  # shuffle:是否在每个epoch重新排列数据 download=True表示如果root目录下没有数据则从网上下载数据（原来为False)
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.path['mnist_path'], train=False, download=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    def __load_fashion_mnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                      transforms.Resize((size, size)),
                                      # transforms.RandomHorizontalFlip(),            # 水平翻转
                                      # transforms.RandomRotation(15),                # 旋转，范围-15-15
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
                                      # transforms.RandomHorizontalFlip(),            # 水平翻转
                                      # transforms.RandomRotation(15),  # 旋转，范围-15-15
                                      transforms.ToTensor(),
                                  ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=True, download=False, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=False, download=False, transform=test_transform),
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
            datasets.QMNIST(self.path['qmnist_path'], train=True, download=True, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.QMNIST(self.path['qmnist_path'], train=False, download=True, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    def __load_cifar10(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 CIFAR10Policy(),
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
                                 #CIFAR10Policy(),
                                 transforms.Resize((size, size)),
                                 #训练模型时，以下3行不能删，否则会出现严重过拟合
                                 #transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            auto_transform = transforms.Compose([
                                 CIFAR10Policy(),
                                 transforms.Resize((size, size)),
                                 # 训练模型时，以下3行不能删，否则会出现严重过拟合
                                 #transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])
        trainset = Get_Dataset_C10(root=self.path['cifar10_path'], train=True, transform=[train_transform, auto_transform],
                                   download=False)
        """
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=True, download=False, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )
        """
        train_size = 49000
        valid_size = 1000
        test_size = 10000
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(100)
        for index in range(len(trainset)):
            _, _, _, target = trainset[index]
            if (np.all(count == 10)):
                break
            if (count[target] < 10):
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=train_batch_size, sampler=SubsetRandomSampler(train_indices)
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=False, download=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader

    def __load_cifar100(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 CIFAR10Policy(),
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
                                 #CIFAR10Policy(),
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])
            auto_transform = transforms.Compose([
                                 CIFAR10Policy(),
                                 transforms.Resize((size, size)),
                                 #transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])                   

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])
        """
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=True, download=False, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )
        """
        trainset = Get_Dataset_C100(root=self.path['cifar100_path'], train=True,transform=[train_transform, auto_transform], download=False)
        
        train_size = 49000
        valid_size = 1000
        test_size = 10000
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(100)
        for index in range(len(trainset)):
            _, _, _, target = trainset[index]
            if (np.all(count == 10)):
                break
            if (count[target] < 10):
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=train_batch_size, sampler=SubsetRandomSampler(train_indices)
        )               
        

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=False, download=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader

    def __load_svhn(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomRotation(15),
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
                SVHNPolicy(),
                transforms.Resize((size, size)),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(self.path['svhn_path'], split='train', download=False, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(self.path['svhn_path'], split='test', download=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader


if __name__ == '__main__':

    data_path = '/home/suxw/save_data/mnist/'

#    g = image_complexity('g2')  #

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
        gs_test.append(torch.sum(torch.flatten(data, start_dim=1), axis=1))  # flatten推平，start_dim=1(end_dim=-1):把第1个维度到最后一个维度全部推平合并
        labels_test.append(target)

    gs_all = np.concatenate(gs_all)  # 数组拼接
    labels_all = np.concatenate(labels_all)

    gs_test = np.concatenate(gs_test)
    labels_test = np.concatenate(labels_test)

    df_all = pd.DataFrame({'img_fuzz': gs_all, 'label': labels_all})
    df_test = pd.DataFrame({'img_fuzz': gs_test, 'label': labels_test})

    set_diff_df = pd.concat([df_all, df_test, df_test]).drop_duplicates(keep=False)  # pd.concat数据拼接。drop_duplicates:删除重复项，keep=False:删除所有重复项
    df = set_diff_df.sort_values(by=['img_fuzz'])  # 根据img_fuzz从小到大排序
    df_label = df.groupby(['label'])  # 根据label对数据进行分组
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
    #                                 transform=transforms.Compose([transforms.ToTensor()]))



