import torch
import random
from diffusion.utils.diffusion_process import DiffusionProcess
from diffusers import DDPMScheduler

class DiffusionDS(torch.utils.data.Dataset):
    def __init__(self, dataset, num_train_timesteps=1000):
        self.dataset = dataset
        self.diffusion_process = DDPMScheduler(num_train_timesteps=1000)

    def __getitem__(self, idx, t=None):
        x_0, _ = self.dataset[idx]
        epsilon = torch.randn(x_0.shape)
        timesteps = torch.LongTensor([t])
        x_t = self.diffusion_process.add_noise(x_0, epsilon, timesteps)

        return x_t, epsilon, torch.tensor(t).int()

    def __len__(self):
        return len(self.dataset)