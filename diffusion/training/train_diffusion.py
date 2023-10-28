import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../chip-project')))
from argparse import ArgumentParser

import torch
from torchvision import datasets, transforms
from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss
from diffusion.models.diffusion_model import UNetDiffusionModel
from diffusion.datasets.diffusion_dataset import DiffusionDS
from diffusion.utils.diffusion_process import DiffusionProcess
import random


class CIFAR10_Dataset():
    def __init__(self, root_path):
        self.data_mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.data_stddev = torch.tensor([0.5, 0.5, 0.5])

        self.train_dataset = datasets.CIFAR10(root=root_path, train=True, download=True,
                                              transform=self.train_transform())
        self.test_dataset = datasets.CIFAR10(root=root_path, train=False, download=True,
                                             transform=self.test_transform())

    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_stddev),
            ]
        )

    def test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(self.data_mean, self.data_stddev),
        ])

    def renormalize(self, x):
        x = x.clone()
        if x.shape[0] == 3:
            sh = x.shape[1:]
            x = x.reshape(3, -1).T
            x = x.reshape(*sh, 3)
        x *= self.data_stddev
        x += self.data_mean
        return x


class MNIST_Dataset():
    def __init__(self, root_path):
        self.data_mean = torch.tensor([0.1307])
        self.data_stddev = torch.tensor([0.5])


        self.train_dataset = datasets.MNIST(root=root_path, train=True, download=True,
                                            transform=self.train_transform())
        self.test_dataset = datasets.MNIST(root=root_path, train=False, download=True,
                                           transform=self.test_transform())

    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_stddev),
            ]
        )

    def test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(self.data_mean, self.data_stddev),
        ])

    def renormalize(self, x):
        x = x.clone()
        if x.shape[0] == 1:
            sh = x.shape[1:]
            x = x.reshape(1, -1).T
            x = x.reshape(*sh, 1)
        x *= self.data_stddev
        x += self.data_mean
        return x


class diffusion_loss(BaseLoss):

    def __init__(self):
        stats_names = ["loss"]
        super(diffusion_loss, self).__init__(stats_names)

    def compute_loss(self, instance, model):
        mse = torch.nn.MSELoss()
        x_t, epsilon, t = instance
        x_t, epsilon, t = x_t.to(device), epsilon.to(device), t.to(device)

        model.zero_grad()
        output = model.forward(x_t, t)
        loss = mse(output, epsilon)
        return loss, {"loss": loss}


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"models loaded from checkpoint {model_path}")


if __name__ == '__main__':
    import lovely_tensors as lt

    lt.monkey_patch()

    parser = ArgumentParser(description="PyTorch experiments")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size of every process")
    parser.add_argument("--epochs", default=1001, type=int, help="number of epochs to train")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="weight decay")
    parser.add_argument("--scheduler", default="[500]", type=str, help="scheduler decrease after epochs given")
    parser.add_argument("--lr_decay", default=0.1, type=float, help="Learning rate decay")
    parser.add_argument("--wandb_exp_name", default='random_experiments', type=str, help="Experiment name in wandb")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument("--load_checkpoint", default='', type=str, help="name of models in folder checkpoints to load")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    args = vars(parser.parse_args())
    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp]
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    device = torch.device('cpu')

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../chip-project'))
    model_path = f"{project_path}/checkpoints/diffusion_model_mnist.pt"

    model = UNetDiffusionModel(
        image_channels=1,
        n_channels=64,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True],
    ).to(device)

    if args['load_checkpoint'] != "":
        load_model(model, f"{project_path}/checkpoints/{args['load_checkpoint']}")

    dp = DiffusionProcess(T=1000)

    root_path = "/Users/lfbarba/GitHub/data"
    # root_path = "/myhome/chip-project/data"
    cifar = MNIST_Dataset(root_path)
    trainSet = DiffusionDS(cifar.train_dataset, diffusion_process=dp)
    testSet = DiffusionDS(cifar.test_dataset, diffusion_process=dp)

    exp = PyTorchExperiment(
        train_dataset=trainSet,
        test_dataset=testSet,
        batch_size=args['batch_size'],
        model=model,
        loss_fn=diffusion_loss(),
        checkpoint_path=model_path,
        experiment_name=args['wandb_exp_name'],
        with_wandb=args['wandb'],
        num_workers=0,
        seed=args["seed"],
        args=args
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args['learning_rate'],
        weight_decay=args['weight_decay'])

    exp.train(args['epochs'], optimizer, milestones=args['scheduler'], gamma=args['lr_decay'])








