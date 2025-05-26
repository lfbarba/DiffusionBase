"""
This script trains a diffusion model.
It includes:
- Dataset handling for CIFAR10 and MNIST, including transformations.
- A custom loss function for the diffusion process.
- The main training loop, orchestrated by PyTorchExperiment.
- Argument parsing for configuring the training process.
- Model loading functionality.
"""
import sys
import os

# Add the chip-project directory to the Python path to import project modules
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

# Diffusers library imports for UNet model, scheduler, and optimizer
from diffusers import UNet2DConditionModel # TODO: Check if this is used, otherwise remove
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


class CIFAR10_Dataset():
    """
    Handles loading and transforming the CIFAR-10 dataset.
    """
    def __init__(self, root_path):
        """
        Initializes the CIFAR10_Dataset.

        Args:
            root_path (str): The root directory where the CIFAR-10 dataset
                             is stored or will be downloaded.
        """
        self.data_mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.data_stddev = torch.tensor([0.5, 0.5, 0.5])

        self.train_dataset = datasets.CIFAR10(root=root_path, train=True, download=True,
                                              transform=self.train_transform())
        self.test_dataset = datasets.CIFAR10(root=root_path, train=False, download=True,
                                             transform=self.test_transform())

    def train_transform(self):
        """
        Defines the transformations applied to the training dataset.
        Includes random cropping, horizontal flipping, resizing, conversion to tensor,
        and normalization.

        Returns:
            transforms.Compose: A composition of image transformations.
        """
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
                transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                transforms.Resize((64, 64)),          # Resize the image to 64x64 pixels
                transforms.ToTensor(),                # Convert the image to a PyTorch tensor
                transforms.Normalize(self.data_mean, self.data_stddev), # Normalize the image
            ]
        )

    def test_transform(self):
        """
        Defines the transformations applied to the test dataset.
        Includes conversion to tensor, resizing, and normalization.

        Returns:
            transforms.Compose: A composition of image transformations.
        """
        return transforms.Compose([
            transforms.ToTensor(),                # Convert the image to a PyTorch tensor
            transforms.Resize((64, 64)),          # Resize the image to 64x64 pixels
            transforms.Normalize(self.data_mean, self.data_stddev), # Normalize the image
        ])

    def renormalize(self, x):
        """
        Reverses the normalization applied to an image tensor.

        Args:
            x (torch.Tensor): The normalized image tensor.

        Returns:
            torch.Tensor: The denormalized image tensor.
        """
        x = x.clone()
        # Reshape if the tensor is for a single image (C, H, W) vs batch (N, C, H, W)
        if x.shape[0] == 3: # Assuming 3 channels for CIFAR-10
            sh = x.shape[1:]
            x = x.reshape(3, -1).T
            x = x.reshape(*sh, 3)
        x *= self.data_stddev # Multiply by standard deviation
        x += self.data_mean   # Add mean
        return x


class MNIST_Dataset():
    """
    Handles loading and transforming the MNIST dataset.
    """
    def __init__(self, root_path):
        """
        Initializes the MNIST_Dataset.

        Args:
            root_path (str): The root directory where the MNIST dataset
                             is stored or will be downloaded.
        """
        self.data_mean = torch.tensor([0.1307]) # Mean for MNIST (grayscale)
        self.data_stddev = torch.tensor([0.5])  # Standard deviation for MNIST

        self.train_dataset = datasets.MNIST(root=root_path, train=True, download=True,
                                            transform=self.train_transform())
        self.test_dataset = datasets.MNIST(root=root_path, train=False, download=True,
                                           transform=self.test_transform())

    def train_transform(self):
        """
        Defines the transformations applied to the training dataset.
        Includes random cropping, resizing, conversion to tensor, and normalization.

        Returns:
            transforms.Compose: A composition of image transformations.
        """
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
                transforms.Resize((64, 64)),          # Resize the image to 64x64 pixels
                transforms.ToTensor(),                # Convert the image to a PyTorch tensor
                transforms.Normalize(self.data_mean, self.data_stddev), # Normalize the image
            ]
        )

    def test_transform(self):
        """
        Defines the transformations applied to the test dataset.
        Includes conversion to tensor, resizing, and normalization.

        Returns:
            transforms.Compose: A composition of image transformations.
        """
        return transforms.Compose([
            transforms.ToTensor(),                # Convert the image to a PyTorch tensor
            transforms.Resize((64, 64)),          # Resize the image to 64x64 pixels
            transforms.Normalize(self.data_mean, self.data_stddev), # Normalize the image
        ])

    def renormalize(self, x):
        """
        Reverses the normalization applied to an image tensor.

        Args:
            x (torch.Tensor): The normalized image tensor.

        Returns:
            torch.Tensor: The denormalized image tensor.
        """
        x = x.clone()
        # Reshape if the tensor is for a single image (C, H, W) vs batch (N, C, H, W)
        if x.shape[0] == 1: # Assuming 1 channel for MNIST
            sh = x.shape[1:]
            x = x.reshape(1, -1).T
            x = x.reshape(*sh, 1)
        x *= self.data_stddev # Multiply by standard deviation
        x += self.data_mean   # Add mean
        return x


class diffusion_loss(BaseLoss):
    """
    Custom loss function for the diffusion model.
    It calculates the Mean Squared Error (MSE) between the predicted noise
    and the actual noise added to the images.
    """
    def __init__(self):
        stats_names = ["loss"]
        super(diffusion_loss, self).__init__(stats_names)

    def compute_loss(self, instance, model):
        """
        Computes the diffusion loss.

        Args:
            instance: A batch of data from the DataLoader, expected to be a tuple
                      (x_t, y) where x_t is the noisy image and y is the original noise.
                      Note: y is named `noise` in the implementation below.
            model: The diffusion model being trained.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed MSE loss.
                - stats (dict): A dictionary containing the loss value.
        """
        mse = torch.nn.MSELoss()
        # x_t here is the original clean image from the dataset, not the noisy one.
        # The variable name `y` is misleading in the original code, it should be `noise_target` or similar.
        # However, the dataset returns (noisy_image, original_noise, timestep), so `x_t` from instance is already noisy.
        # Let's assume `instance` provides (clean_image, _), and we add noise here.
        # Based on the `DiffusionDS` __getitem__, `instance` is (x_t, epsilon, t_idx_for_batch)
        # So x_t is already noisy. The `noise` variable below is the target.
        
        # The current implementation of diffusion_loss takes (x_t, y) from the dataloader.
        # Where x_t is the *clean* image and y is unused.
        # Noise is then added inside this function. This is inefficient if DiffusionDS already adds noise.
        # For now, I will document based on the current code structure.
        # If DiffusionDS returns (noisy_image, noise_target, timestep), then this function should be simplified.
        # Re-evaluating based on the main script: exp.train_dataset is cifar.train_dataset, not DiffusionDS.
        # So this loss function DOES need to perform the noising.

        clean_images, _ = instance # x_t is the clean image, y is the label (unused in diffusion)
        clean_images = clean_images.to(device)
        # Generate random noise
        noise_target = torch.randn_like(clean_images).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image in the batch
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise_target, timesteps)

        model.zero_grad()
        # Predict the noise from the noisy image and timestep
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        # Calculate the MSE loss between the predicted noise and the actual noise
        loss = mse(noise_pred, noise_target)
        return loss, {"loss": loss}


def load_model(model, model_path):
    """
    Loads model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model instance to load the weights into.
        model_path (str): Path to the checkpoint file (.pt).
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Models loaded from checkpoint {model_path}")


if __name__ == '__main__':
    import lovely_tensors as lt

    lt.monkey_patch() # For pretty printing tensors, useful for debugging

    # --- Argument Parsing ---
    # Initialize ArgumentParser to handle command-line arguments for the experiment
    parser = ArgumentParser(description="PyTorch experiments for diffusion model training")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size for training and testing")
    parser.add_argument("--epochs", default=1001, type=int, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate for the optimizer")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay for the optimizer")
    parser.add_argument("--scheduler", default="[500]", type=str, help="Epoch milestones for learning rate scheduler (e.g., '[100,200]')")
    parser.add_argument("--lr_decay", default=0.1, type=float, help="Factor by which the learning rate is reduced by the scheduler")
    parser.add_argument("--wandb_exp_name", default='random_experiments', type=str, help="Name of the experiment in Weights & Biases")
    parser.add_argument('--wandb', action='store_true', help="Flag to enable Weights & Biases logging")
    parser.add_argument("--load_checkpoint", default='', type=str, help="Path to a model checkpoint to load (relative to project root)")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed for reproducibility (-1 for random seed)")
    args = vars(parser.parse_args()) # Parse arguments and convert to a dictionary

    # Process scheduler argument from string to list of integers
    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp if x] # Ensure empty strings are not converted
    # Set a random seed if not specified, for experiment reproducibility
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]

    # --- Device Selection ---
    # Select training device: CUDA (NVIDIA GPU) if available, otherwise MPS (Apple Silicon GPU), fallback to CPU
    device = torch.device('cuda') if torch.cuda.is_available() else \
             (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    # device = torch.device('cpu') # Uncomment to force CPU training
    print(f"Using device: {device}")

    # Define the path for saving/loading the model checkpoint
    model_path = f"../../diffusion_model_cifar.pt" # TODO: Make this configurable or dataset-dependent

    # --- Model Initialization ---
    # Initialize the UNet2DModel for the diffusion process
    model = UNet2DModel(
        sample_size=64,          # Target image resolution (height and width)
        in_channels=3,           # Number of input channels (3 for RGB CIFAR-10)
        out_channels=3,          # Number of output channels (predicting noise for 3 channels)
        layers_per_block=2,      # Number of ResNet layers per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512), # Output channels for each UNet block
        down_block_types=(       # Configuration of down-sampling blocks
            "DownBlock2D",       # Standard ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",   # ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(         # Configuration of up-sampling blocks
            "UpBlock2D",         # Standard ResNet upsampling block
            "AttnUpBlock2D",     # ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device) # Move the model to the selected device

    # Load model from checkpoint if specified
    if args['load_checkpoint']:
        load_model(model, f"../../{args['load_checkpoint']}")

    # --- Dataset Instantiation ---
    # Define the root path for datasets
    # TODO: Make this configurable via command-line arguments
    root_path = "/Users/lfbarba/GitHub/data"  # Example local path
    # root_path = "/myhome/chip-project/data" # Example server path
    # Initialize the CIFAR-10 dataset handler
    cifar = CIFAR10_Dataset(root_path)
    # TODO: Add logic to select between CIFAR10 and MNIST based on an argument

    # --- DDPMScheduler Initialization ---
    # Initialize the Denoising Diffusion Probabilistic Models (DDPM) scheduler
    # This scheduler manages the noise addition process (forward diffusion)
    # and is used during sampling (reverse diffusion).
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # --- PyTorchExperiment Setup ---
    # Initialize the PyTorchExperiment helper class to manage training and evaluation
    exp = PyTorchExperiment(
        train_dataset=cifar.train_dataset,   # Training dataset
        test_dataset=cifar.test_dataset,     # Test dataset
        batch_size=args['batch_size'],       # Batch size for DataLoaders
        model=model,                         # The UNet model
        loss_fn=diffusion_loss(),            # Custom diffusion loss function
        checkpoint_path=model_path,          # Path to save model checkpoints
        experiment_name=args['wandb_exp_name'], # Name for Weights & Biases experiment
        with_wandb=args['wandb'],            # Enable/disable Weights & Biases logging
        num_workers=0,                       # Number of workers for DataLoader (0 for main process)
        seed=args["seed"],                   # Random seed for reproducibility
        args=args                            # Pass all command-line arguments to the experiment
    )

    # --- Optimizer and Learning Rate Scheduler Setup ---
    # Initialize the AdamW optimizer with model parameters and learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    num_epochs = args['epochs'] # Use epochs from arguments
    # Initialize a cosine learning rate scheduler with warmup
    # This scheduler adjusts the learning rate during training for better convergence.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500, # Number of steps for the learning rate to warm up
        num_training_steps=(len(cifar.train_dataset) // args['batch_size'] * num_epochs), # Total training steps
    )

    # --- Start Training ---
    # Call the train method of the PyTorchExperiment class to start the training loop
    exp.train(
        epochs=num_epochs,
        optimizer=optimizer,
        milestones=args['scheduler'], # Epoch milestones for the MultiStepLR (if used by PyTorchExperiment)
        gamma=args['lr_decay'],       # Learning rate decay factor for MultiStepLR
        scheduler=lr_scheduler        # The cosine scheduler with warmup
    )








