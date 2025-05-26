import torch
import numpy as np

class DiffusionProcess():
    """
    Manages the noising and denoising schedules and operations for a diffusion model.

    This class is responsible for:
    - Computing the noise schedule (betas, alphas, alphas_tilde) based on
      different modes (e.g., "cosine", "linear").
    - Implementing the forward diffusion process (q_sampler): adding noise to
      an image x_0 to get x_t at a specific timestep t.
    - Implementing one step of the reverse diffusion process (p_sampler):
      denoising x_t to predict x_{t-1}, given the predicted noise.
    """
    def __init__(self, T, beta_0=0.02, beta_T=0.001, mode="cosine"):
        """
        Initializes the DiffusionProcess.

        Args:
            T (int): The total number of diffusion timesteps.
            beta_0 (float): The starting value of beta (noise variance)
                            for the schedule.
            beta_T (float): The ending value of beta (noise variance)
                            for the schedule.
            mode (str): The type of noise schedule to use.
                        Supported modes: "cosine", "linear".
        """
        self.T = T
        # Compute the noise schedule parameters upon initialization
        params = self.compute_noise_schedule(T, beta_0, beta_T, mode)
        self.alphas, self.alphas_tilde, self.betas = params

    def compute_noise_schedule(self, T, beta_0, beta_T, mode="cosine"):
        """
        Computes the noise schedule (variances and related parameters) for the
        diffusion process.

        The schedule determines how much noise is added at each timestep.
        - betas: Variance schedule (noise levels).
        - alphas: 1 - betas. Represents the signal rate.
        - alphas_tilde: Cumulative product of alphas. Represents the overall
                        signal rate from x_0 to x_t.

        Args:
            T (int): Number of timesteps.
            beta_0 (float): Initial beta value (start of the noise schedule).
            beta_T (float): Final beta value (end of the noise schedule).
            mode (str): Schedule mode. Supported: ["cosine", "linear"].
                        - "cosine": Follows a cosine-based schedule, often preferred
                                    for smoother transitions.
                        - "linear": Follows a linearly increasing schedule for betas.

        Returns:
            tuple: A tuple containing:
                - alphas (torch.Tensor): Tensor of alpha values for each timestep.
                - alphas_tilde (torch.Tensor): Tensor of cumulative alpha_tilde values.
                - betas (torch.Tensor): Tensor of beta values (noise variances).
        """
        if mode == "cosine":
            # Cosine schedule: betas are calculated based on a cosine function.
            # This schedule starts with small beta values and increases them,
            # often leading to better performance than linear schedules.
            betas = beta_0 + 0.5 * (beta_T - beta_0) * (1 + np.cos(np.pi * np.arange(T) / (T - 1)))
        elif mode == "linear":
            # Linear schedule: betas increase linearly from beta_0 to beta_T.
            betas = np.linspace(beta_0, beta_T, T)
        else:
            raise ValueError(f"Unsupported schedule mode: {mode}. Supported modes are 'cosine' and 'linear'.")

        # Compute alphas based on betas: alpha_t = 1 - beta_t
        alphas = 1 - betas
        # Compute alphas_tilde (cumulative product of alphas): alpha_tilde_t = product_{s=1 to t} alpha_s
        # This value is crucial for sampling x_t directly from x_0.
        alphas_tilde = np.cumprod(alphas)

        return (
            torch.tensor(alphas, dtype=torch.float32),
            torch.tensor(alphas_tilde, dtype=torch.float32),
            torch.tensor(betas, dtype=torch.float32)
        )

    def x_t(self, x_0, t):
        """
        Implements the forward diffusion process (q_sampler).
        Adds noise to an initial sample x_0 to produce a noisy sample x_t at timestep t.
        The formula is: x_t = sqrt(alpha_tilde_t) * x_0 + sqrt(1 - alpha_tilde_t) * epsilon,
        where epsilon is random noise ~ N(0, I).

        Args:
            x_0 (torch.Tensor): The initial clean data sample (e.g., an image).
            t (int or torch.Tensor): The timestep(s) at which to compute the noisy sample.
                                     If an int, it's applied to the whole batch.
                                     If a tensor, it specifies timesteps per item in the batch.

        Returns:
            tuple: A tuple containing:
                - x_t (torch.Tensor): The noisy sample at timestep t.
                - epsilon (torch.Tensor): The noise that was added to x_0.
        """
        # Generate random noise from a standard normal distribution, with the same shape as x_0
        epsilon = torch.randn_like(x_0)

        # Ensure t is a tensor for indexing alphas_tilde
        if isinstance(t, int):
            # Use t for all items in the batch
            sqrt_alpha_tilde_t = torch.sqrt(self.alphas_tilde[t])
            sqrt_one_minus_alpha_tilde_t = torch.sqrt(1 - self.alphas_tilde[t])
        elif isinstance(t, torch.Tensor):
            # t is a tensor, use its values to index alphas_tilde for each item in the batch
            # Reshape t to be (batch_size, 1, 1, 1) for broadcasting with x_0 (N, C, H, W)
            batch_size = t.shape[0]
            sqrt_alpha_tilde_t = torch.sqrt(self.alphas_tilde[t]).reshape(batch_size, 1, 1, 1).to(x_0.device)
            sqrt_one_minus_alpha_tilde_t = torch.sqrt(1 - self.alphas_tilde[t]).reshape(batch_size, 1, 1, 1).to(x_0.device)
        else:
            raise TypeError("t must be an int or a torch.Tensor")

        # Apply the forward diffusion formula
        x_t = sqrt_alpha_tilde_t * x_0 + sqrt_one_minus_alpha_tilde_t * epsilon
        return x_t, epsilon

    def reverse_process(self, x_t, t, noise):
        """
        Implements one step of the reverse diffusion process (p_sampler).
        Given a noisy sample x_t at timestep t and the predicted noise (from the model),
        this method estimates the sample at the previous timestep, x_{t-1}.

        The core idea is to predict the mean of p(x_{t-1} | x_t) and then add
        some variance (noise) to sample from this distribution.

        Args:
            x_t (torch.Tensor): The noisy sample at the current timestep t.
            t (int or torch.Tensor): The current timestep(s).
            noise (torch.Tensor): The noise predicted by the model for x_t at timestep t.

        Returns:
            torch.Tensor: The estimated sample at timestep t-1.
        """
        # Ensure t is usable for indexing and calculations
        current_t_idx = t if isinstance(t, int) else t.item() # Assuming t is scalar or single-element tensor for simplicity here
        if current_t_idx == 0:
            # If at timestep 0, no further reverse step is possible with this formulation.
            # Depending on the desired behavior, could return x_t or raise an error.
            # For DDPM, this function is usually called for t > 0.
            return x_t # Or handle as an error/special case

        # Gather required parameters for the current timestep t
        # These need to be correctly indexed if t can be a tensor for batch operations.
        # For simplicity in this docstring example, assuming t is an int or scalar tensor.
        beta_t = self.betas[current_t_idx]
        alpha_t = self.alphas[current_t_idx]
        alpha_tilde_t = self.alphas_tilde[current_t_idx]
        alpha_tilde_t_prev = self.alphas_tilde[current_t_idx - 1] if current_t_idx > 0 else torch.tensor(1.0) # alpha_tilde_0 is 1

        # Calculate the predicted mean of x_{t-1}
        # This formula comes from deriving p(x_{t-1} | x_t, x_0) and substituting x_0 with its estimate from x_t and noise.
        # pred_mean = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_tilde_t)) * noise)
        pred_mean = (x_t - beta_t * noise.to(x_t.device) / torch.sqrt(1 - alpha_tilde_t)) / torch.sqrt(alpha_t)

        # Calculate beta_tilde_t, which is the variance of q(x_{t-1} | x_t, x_0).
        # This is the variance of the "posterior" distribution used in the reverse step.
        # beta_tilde_t = beta_t * (1 - alpha_tilde_{t-1}) / (1 - alpha_tilde_t)
        beta_tilde = beta_t * (1 - alpha_tilde_t_prev) / (1 - alpha_tilde_t)

        # Add noise to the predicted mean to get the sample x_{t-1}
        # The noise added is scaled by sqrt(beta_tilde_t).
        # If t=0, beta_tilde is not used for noise, typically. (Handled by current_t_idx == 0 check)
        random_noise = torch.randn_like(pred_mean).to(x_t.device)
        return pred_mean + torch.sqrt(beta_tilde) * random_noise