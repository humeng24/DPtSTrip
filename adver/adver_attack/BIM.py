from advertorch.attacks import LinfBasicIterativeAttack
from model import *
import configuration.configuration as cfg
from adver.adver_attack.attack import attack
from module import *

embedding_node = 84
base_path = '/home/suxw/pycharm_project/adver_project/save_model/'


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


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_p = 'cifar10'           # mnist | fashion_mnist | cifar10 | cifar100
model_path = base_path + model_p

model_list = os.walk(model_path).__next__()[2]


if model_p == 'mnist':
    from datasets.mnist.mnist import *
elif model_p == 'fashion_mnist':
    from datasets.mnist.fashion_mnist import *
elif model_p == 'cifar10':
    from datasets.cifar.cifar10 import *
elif model_p == 'cifar100':
    from datasets.cifar.cifar100 import *


for model_i in model_list:
    model_pt = os.path.join(model_path, model_i)
    print(f"The result of {model_i}")
    print('-'*50)
    model = torch.load(model_pt)
    model.to(device)
    if 'mnist' in model_p:
        adversary = LinfBasicIterativeAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.55,
            nb_iter=40, eps_iter=0.1, clip_min=0.0,
            clip_max=1.0, targeted=False)
    elif 'cifar' in model_p:
        adversary = LinfBasicIterativeAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=20/255,
            nb_iter=40, eps_iter=0.1, clip_min=0.0,
            clip_max=1.0, targeted=False)
    attack(adversary, model, test_loader, device)
    print('-'*50)
