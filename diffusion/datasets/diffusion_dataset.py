import torch
import random
from diffusion.utils.diffusion_process import DiffusionProcess

class DiffusionDS(torch.utils.data.Dataset):
    def __init__(self, dataset, diffusion_process:DiffusionProcess):
        self.dataset = dataset
        self.diffusion_process = diffusion_process

    def __getitem__(self, idx, t=None):
        x_0, _ = self.dataset[idx]
        if t is None:
            # We sample later steps more often
            c = random.randint(0, 2)
            if c == 0:
                t = random.randint(0, self.diffusion_process.T - 1)
            elif c == 1:
                t = random.randint(self.diffusion_process.T // 2, self.diffusion_process.T - 1)
            else:
                t = random.randint(3 * self.diffusion_process.T // 4, self.diffusion_process.T - 1)

        x_t, epsilon = self.diffusion_process.x_t(x_0, t)
        # concatenate the sqrt(alpha_tilde_t) and (1- alpha_tilde_t) as additional channels
        return x_t, epsilon, torch.tensor(t).int()

    def __len__(self):
        return len(self.dataset)