from advertorch.attacks import LinfPGDAttack
from model import *
import configuration.configuration as cfg
from adver.adver_attack.attack import attack
from module import *
from datasets.load_dataset import data_loader

embedding_node = 512
size = 224
# size = 32
base_path = '/home/suxw/save_model/'
data_selection = data_loader()


class net(nn.Module):
    def __init__(self, backbone):
        super(net, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embedding_node, 10)

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x
        x = self.fc(x)
        return x


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_p = 'fashion_mnist'           # mnist | fashion_mnist | cifar10 | cifar100
model_path = base_path + model_p

model_list = os.walk(model_path).__next__()[2]


for model_i in model_list:

    if 'vgg' in model_i:
        size = 224
    elif 'resnet' in model_i and 'cifar' in model_i:
        size = 32
    elif 'resnet' in model_i and 'mnist' in model_i:
        size = 28
    elif 'lenet' in model_i:
        size = 28

    if model_p == 'mnist':
        _, test_loader = data_selection('mnist', size=size)
    elif model_p == 'fashion_mnist':
        _, test_loader = data_selection('fashion_mnist', size=size)
    elif model_p == 'cifar10':
        _, test_loader = data_selection('cifar10', size=size)
    elif model_p == 'cifar100':
        _, test_loader = data_selection('cifar100', size=size)

    model_pt = os.path.join(model_path, model_i)
    print(f"The result of {model_i}")
    print('-'*50)
    model = torch.load(model_pt)
    model.to(device)

    if 'mnist' in model_p:
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=0.55,
            nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
    elif 'cifar' in model_p:
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=20 / 255,
            nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)

    attack(adversary, model, test_loader, device)
    print('-'*50)

    del test_loader
    torch.cuda.empty_cache()


