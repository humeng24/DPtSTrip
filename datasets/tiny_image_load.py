import os
import glob
from torch.utils.data import Dataset
from PIL import Image

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = '/data/suxw/sxuhome/image_data/tiny-imagenet-200/wnids.txt'
VAL_ANNOTATION_FILE = '/data/suxw/sxuhome/image_data/tiny-imagenet-200/val/val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
#from TinyImageNet import TinyImageNet
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.utils.data as data
#from TinyImageNet import TinyImageNet




def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def ImageNet_get_loaders(dir_, batch_size):
    num_workers = {'train' : 0,'val'   : 0,'test'  : 0}
    data_transforms = {'train': transforms.Compose([transforms.ToTensor(),]),\
                       'val': transforms.Compose([transforms.ToTensor(),]),\
                       'test': transforms.Compose([transforms.ToTensor(),])}
    num_workers = 0
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x])
                  for x in ['train', 'val','test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_workers)
                  for x in ['train', 'val', 'test']}
    return dataloaders

def ImageNet_get_loaders_32(dir_, batch_size):
    num_workers = {'train' : 0,'val'   : 0,'test'  : 0}
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])}
    num_workers = 0
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x])
                  for x in ['train', 'val','test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_workers)
                  for x in ['train', 'val', 'test']}
    return dataloaders


def New_ImageNet_get_loaders_32(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(dir_, 'train', transform=transform_train, in_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)



    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader

def New_ImageNet_get_loaders_32_testloader(dir_, batch_size):


    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])




    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return  testloader

def New_ImageNet_get_loaders_64(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(dir_, 'train', transform=transform_train, in_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)



    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader



def New_ImageNet_get_loaders_64_testloader(dir_, batch_size):


    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])

    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return testloader








