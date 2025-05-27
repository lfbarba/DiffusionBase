import torch
import random
from diffusion.utils.diffusion_process import DiffusionProcess
from diffusers import DDPMScheduler


class DiffusionDS(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for diffusion models.
    This dataset takes an existing dataset of clean images and applies
    the diffusion process to generate noisy images at various timesteps.
    """
    def __init__(self, dataset, num_train_timesteps=1000):
        """
        Initializes the DiffusionDS dataset.

        Args:
            dataset: The underlying dataset of clean images.
            num_train_timesteps (int): The number of timesteps in the
                                     diffusion process.
        """
        self.dataset = dataset
        # Initialize the DDPM scheduler for the diffusion process
        self.diffusion_process = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    def __getitem__(self, idx, t=None):
        """
        Retrieves a noisy image sample and the corresponding noise.

        Args:
            idx (int): The index of the clean image in the underlying dataset.
            t (int, optional): The timestep for the diffusion process.
                               If None, a random timestep is chosen.
                               Defaults to None.

        Returns:
            tuple: A tuple containing:
                - x_t (torch.Tensor): The noisy image at timestep t.
                - epsilon (torch.Tensor): The noise added to the clean image.
                - t (torch.Tensor): The timestep used for the diffusion process.
        """
        # Get the clean image from the underlying dataset
        x_0, _ = self.dataset[idx]
        # Generate random noise with the same shape as the clean image
        epsilon = torch.randn(x_0.shape)
        # Convert the timestep to a LongTensor
        timesteps = torch.LongTensor([t])
        # Add noise to the clean image using the DDPM scheduler
        x_t = self.diffusion_process.add_noise(x_0, epsilon, timesteps)

        return x_t, epsilon, torch.tensor(t).int()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)