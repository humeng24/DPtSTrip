from module import *
import configuration.configuration as cfg

__all__ = ['train_loader', 'test_loader', 'train_batch_size']

mnist_normalize = {"mean": 0.1307, "std": 0.3081}
train_batch_size = 256
test_batch_size = 1000
path = cfg.dataset_path["mnist_path"]
# 若download=True，则表示下载数据集
download = False

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(path, train=True, download=download,
                          transform=transforms.Compose([
                              # transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize((mnist_normalize["mean"],), (mnist_normalize["std"])),
                          ])),
    batch_size=train_batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(path, train=False, download=download, transform=transforms.Compose([
        # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((mnist_normalize["mean"],), (mnist_normalize["std"])),
    ])),
    batch_size=test_batch_size, shuffle=True
)

if __name__ == '__main__':
    train_loader
    test_loader
