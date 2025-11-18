from module import *
from calculation.complexity import *
from datasets.load_data import data_loader
from advertorch.attacks import FGSM
from advertorch.attacks import LinfPGDAttack


data_selection = data_loader()
dataset = 'fashion_mnist'               # mnist | fashion_mnist | emnist | cifar10 | cifar100
model_name = 'resnet_fashion_mnist.pth'
save_path = f"/home/suxw/project_model/image_complexity/{dataset}/"


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


if __name__ == '__main__':

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(threshold=np.inf)

    train_loader, test_loader = data_selection(dataset)
    g0 = image_complexity('G0')             # 图像复杂度计算方式

    # 读取模型
    model = torch.load(os.path.join(save_path, model_name))
    model.to(device)

    # 攻击方式
    fgsm_adversary = FGSM(model)
    pgd_adversary = LinfPGDAttack(model)

    clean_complexity_ls = []
    fgsm_complexity_ls = []
    pgd_complexity_ls = []

    for data, target in train_loader:

        data = data.to(device)

        # 干净样本
        clean_complexity = g0(data)
        clean_complexity_ls.append(clean_complexity)

        # 对抗样本
        fgsm_data = fgsm_adversary.perturb(data)
        fgsm_complexity = g0(fgsm_data)
        fgsm_complexity_ls.append(fgsm_complexity)

        pgd_data = pgd_adversary.perturb(data)
        pgd_complexity = g0(pgd_data)
        pgd_complexity_ls.append(pgd_complexity)

        print(f"clean : {clean_complexity[:5]}")
        print(f"fgsm : {fgsm_complexity[:5]}")
        print(f"pgd : {pgd_complexity[:5]}")
        print("--"*40)

    clean_complexity_ls = np.concatenate(clean_complexity_ls)
    fgsm_complexity_ls = np.concatenate(fgsm_complexity_ls)
    pgd_complexity_ls = np.concatenate(pgd_complexity_ls)
    print(f"complexity of clean : {np.mean(clean_complexity_ls)}")
    print(f"complexity of fgsm : {np.mean(fgsm_complexity_ls)}")
    print(f"complexity of pgd : {np.mean(pgd_complexity_ls)}")

