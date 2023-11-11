import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffuser(nn.Module):
    """
    Implements the diffusion process for image tensors, progressively adding Gaussian noise.

    Attributes:
        num_timesteps (int): Number of timesteps in the diffusion process.
        alphas (torch.Tensor): Sequence of alpha values for the forward diffusion process.
        sigmas (torch.Tensor): Sequence of sigma values for the forward diffusion process.
    """

    def __init__(self, num_timesteps=1000, alpha_start=0.1, alpha_end=0.9):
        """
        Initializes the Diffuser with calculated alpha and sigma values over timesteps.

        Args:
            num_timesteps (int): Number of timesteps in the diffusion process.
            alpha_start (float): Starting value of alpha for the schedule.
            alpha_end (float): Ending value of alpha for the schedule.
        """
        super(Diffuser, self).__init__()
        self.num_timesteps = num_timesteps

        # Create a schedule for alpha values
        self.alphas = torch.linspace(alpha_start, alpha_end, num_timesteps)
        self.sigmas = torch.sqrt(1.0 - self.alphas**2)

    def forward(self, x, t):
        """
        Applies the diffusion process to the input tensor at a specific timestep.

        Args:
            x (torch.Tensor): The input tensor.
            t (int): The current timestep.

        Returns:
            torch.Tensor: The diffused tensor.
        """
        alpha_t = self.alphas[t]
        sigma_t = self.sigmas[t]

        noise = torch.randn_like(x)
        return alpha_t * x + sigma_t * noise

    # def apply_diffusion(self, x, alpha_t, sigma_t):
    #     """
    #     Adds noise to the input tensor based on alpha and sigma values at a timestep.

    #     Args:
    #         x (torch.Tensor): The input tensor.
    #         alpha_t (float): The alpha value for the current timestep.
    #         sigma_t (float): The sigma value for the current timestep.

    #     Returns:
    #         torch.Tensor: The noised tensor.
    #     """
    #     noise = torch.randn_like(x)
    #     return alpha_t * x + sigma_t * noise


# Example usage
diffuser = Diffuser(num_timesteps=1000, alpha_start=0.1, alpha_end=0.9)
x = torch.randn(1, 3, 256, 256)  # Example input tensor
t = torch.randint(0, 1000, (1,))  # Random diffusion timestep
noised_x = diffuser(x, t.item())
print(noised_x)
