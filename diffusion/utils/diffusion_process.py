import torch
import numpy as np

class DiffusionProcess():
    def __init__(self, T, beta_0=0.02, beta_T=0.001, mode="cosine"):
        self.T = T
        params = self.compute_noise_schedule(T, beta_0, beta_T, mode)
        self.alphas, self.alphas_tilde, self.betas = params

    def compute_noise_schedule(self, T, beta_0, beta_T, mode="cosine"):
        """
        Compute the noise schedule for diffusion models.

        Parameters:
        - T (int): Number of timesteps.
        - beta_0, beta_T (float): Initial and final beta values.
        - mode (str): Schedule mode. Supported: ["cosine", "linear"].

        Returns:
        - decay (numpy array): scale to decay x_0
        - noise_scales (numpy array): Noise scales for each timestep.
        """
        if mode == "cosine":
            # Cosine schedule
            betas = beta_0 + 0.5 * (beta_T - beta_0) * (1 + np.cos(np.pi * np.arange(T) / (T - 1)))
        elif mode == "linear":
            # Linear schedule
            betas = np.linspace(beta_0, beta_T, T)
        else:
            raise ValueError("Unsupported schedule mode.")
        # Compute alphas
        alphas = 1 - betas
        alphas_tilde = np.cumprod(alphas)

        return (
            torch.tensor(alphas),
            torch.tensor(alphas_tilde),
            torch.tensor(betas)
        )

    def x_t(self, x_0, t):
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(self.alphas_tilde[t]) * x_0 + torch.sqrt(1 - self.alphas_tilde[t]) * epsilon
        return x_t, epsilon

    def reverse_process(self, x_t, t, noise):
        pred_mean = (x_t - self.betas[t] * noise.to(x_t.device) / torch.sqrt(1 - self.alphas_tilde[t])) / torch.sqrt(self.alphas[t])
        beta_tilde = self.betas[t] * (1 - self.alphas_tilde[t - 1]) / (1 - self.alphas_tilde[t])

        return pred_mean + torch.sqrt(beta_tilde) * torch.randn_like(pred_mean).to(x_t.device)