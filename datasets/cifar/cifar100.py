from module import *

import configuration.configuration as cfg

__all__ = ['train_loader', 'test_loader', 'train_batch_size']


train_batch_size = 256
test_batch_size = 1000
download = True
path = cfg.dataset_path['cifar100_path']


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(path, train=True, download=download,
                      transform=transforms.Compose([
                          transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ])),
    batch_size=train_batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(path, train=False, download=download,
                      transform=transforms.Compose([
                          # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ])),
    batch_size=test_batch_size, shuffle=True
)

if __name__ == '__main__':
    train_loader
    test_loader
