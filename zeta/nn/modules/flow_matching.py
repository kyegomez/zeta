from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from torch.distributions import Categorical


def make_moons(
    n_samples: int, noise: float = 0.1, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D dataset with two interleaving half circles (moons).

    This is a custom implementation that replaces sklearn.datasets.make_moons.

    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        random_state: Random seed for reproducibility (optional)

    Returns:
        Tuple of (X, y) where:
        - X: ndarray of shape (n_samples, 2) with the generated samples
        - y: ndarray of shape (n_samples,) with binary labels (0 or 1)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate outer moon (semicircle)
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))

    # Generate inner moon (semicircle, flipped and shifted)
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, n_samples_in))

    # Combine the two moons
    X = np.vstack(
        [
            np.column_stack([outer_circ_x, outer_circ_y]),
            np.column_stack([inner_circ_x, inner_circ_y]),
        ]
    )

    # Create labels (0 for outer moon, 1 for inner moon)
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


@dataclass
class FlowConfig:
    """Configuration for Flow neural network.

    Attributes:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate for optimizer
        n_iterations: Number of training iterations
        batch_size: Batch size for training
        noise_level: Noise level for moon dataset
        device: torch device for computation
    """

    dim: int = 2
    hidden_dim: int = 64
    learning_rate: float = 1e-2
    n_iterations: int = 10000
    batch_size: int = 256
    noise_level: float = 0.05
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class Flow(nn.Module):
    """Neural network for modeling continuous normalizing flows.

    This class implements a neural network that learns the velocity field
    of a continuous normalizing flow, transforming a simple distribution
    (e.g., Gaussian) into a more complex target distribution.
    """

    def __init__(self, config: FlowConfig):
        """Initialize the Flow model.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        logger.info(f"Initializing Flow model with config: {config}")

        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.dim + 1, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.dim),
        ).to(config.device)

        logger.debug(f"Model architecture: {self.net}")

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        """Forward pass computing the velocity field at time t and position x_t.

        Args:
            t: Time tensor of shape (batch_size, 1)
            x_t: Position tensor of shape (batch_size, dim)

        Returns:
            Velocity field tensor of shape (batch_size, dim)
        """
        input_tensor = torch.cat((t, x_t), -1)
        return self.net(input_tensor)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        """Perform one step of numerical integration using the midpoint method.

        Args:
            x_t: Current position tensor of shape (batch_size, dim)
            t_start: Start time of the step
            t_end: End time of the step

        Returns:
            New position tensor after integration step
        """
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start

        # Midpoint method
        k1 = self(t=t_start, x_t=x_t)
        x_mid = x_t + k1 * dt / 2
        t_mid = t_start + dt / 2
        k2 = self(t=t_mid, x_t=x_mid)

        return x_t + k2 * dt

    def train_model(self, save_dir: Optional[Path] = None) -> List[float]:
        """Train the flow model using the moons dataset.

        Args:
            save_dir: Optional directory to save model checkpoints

        Returns:
            List of loss values during training
        """
        logger.info("Starting model training")
        optimizer = torch.optim.Adam(
            self.parameters(), self.config.learning_rate
        )
        loss_fn = nn.MSELoss()
        losses = []

        for iteration in range(self.config.n_iterations):
            # Generate data
            x_1 = torch.tensor(
                make_moons(
                    self.config.batch_size,
                    noise=self.config.noise_level,
                )[0],
                dtype=torch.float32,
            ).to(self.config.device)
            x_0 = torch.randn_like(x_1).to(self.config.device)
            t = torch.rand(len(x_1), 1).to(self.config.device)

            # Compute target points
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0

            # Training step
            optimizer.zero_grad()
            loss = loss_fn(self(t=t, x_t=x_t), dx_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if iteration % 1000 == 0:
                logger.info(f"Iteration {iteration}, Loss: {loss.item():.6f}")

            if save_dir and iteration % 1000 == 0:
                self.save_checkpoint(save_dir / f"checkpoint_{iteration}.pt")

        logger.info("Training completed")
        return losses

    def visualize_flow(
        self,
        n_points: int = 300,
        n_steps: int = 8,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize the learned flow transformation.

        Args:
            n_points: Number of points to visualize
            n_steps: Number of integration steps
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (figure, axes array)
        """
        logger.info("Generating flow visualization")
        x = torch.randn(n_points, 2).to(self.config.device)
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        fig, axes = plt.subplots(
            1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True
        )

        axes[0].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10)
        axes[0].set_title(f"t = {time_steps[0]:.2f}")
        axes[0].set_xlim(-3.0, 3.0)
        axes[0].set_ylim(-3.0, 3.0)

        for i in range(n_steps):
            x = self.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
            axes[i + 1].scatter(
                x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10
            )
            axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")

        return fig, axes

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "Flow":
        """Load model from checkpoint.

        Args:
            path: Path to the checkpoint file

        Returns:
            Loaded Flow model
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path}")
        return model


# # Example usage
# if __name__ == "__main__":
#     # Setup logging
#     logger.add("flow_model.log", rotation="500 MB")

#     # Initialize and train model
#     config = FlowConfig()
#     flow = Flow(config)
#     losses = flow.train_model(save_dir=Path("checkpoints"))

#     # Visualize results
#     flow.visualize_flow(save_path=Path("flow_visualization.png"))


@dataclass
class MixtureFlowConfig:
    """Configuration for Mixture of Flows neural network.

    Attributes:
        n_experts: Number of flow experts in the mixture
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension for each expert
        gating_hidden_dim: Hidden dimension for gating network
        learning_rate: Learning rate for optimizer
        n_iterations: Number of training iterations
        batch_size: Batch size for training
        noise_level: Noise level for moon dataset
        expert_dropout: Dropout rate for expert selection
        checkpoint_interval: Number of iterations between checkpoints
        device: torch device for computation
        logging_interval: Number of iterations between logging
    """

    n_experts: int = 4
    dim: int = 2
    hidden_dim: int = 64
    gating_hidden_dim: int = 32
    learning_rate: float = 1e-3
    n_iterations: int = 10000
    batch_size: int = 256
    noise_level: float = 0.05
    expert_dropout: float = 0.1
    checkpoint_interval: int = 1000
    logging_interval: int = 100
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class GatingNetwork(nn.Module):
    """Neural network for selecting which flow expert to use.

    The gating network takes the current state and time as input and outputs
    a probability distribution over experts.
    """

    def __init__(self, config: MixtureFlowConfig):
        """Initialize the gating network.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.dim + 1, config.gating_hidden_dim),
            nn.LayerNorm(config.gating_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.expert_dropout),
            nn.Linear(config.gating_hidden_dim, config.gating_hidden_dim),
            nn.LayerNorm(config.gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gating_hidden_dim, config.n_experts),
        ).to(config.device)

        logger.debug(f"Gating network architecture: {self.net}")

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        """Forward pass computing expert selection probabilities.

        Args:
            t: Time tensor of shape (batch_size, 1)
            x_t: Position tensor of shape (batch_size, dim)

        Returns:
            Expert selection probabilities tensor of shape (batch_size, n_experts)
        """
        input_tensor = torch.cat((t, x_t), -1)
        logits = self.net(input_tensor)
        return F.softmax(logits, dim=-1)


class ExpertFlow(nn.Module):
    """Individual flow expert network."""

    def __init__(self, config: MixtureFlowConfig, expert_id: int):
        """Initialize an expert flow network.

        Args:
            config: Configuration object containing model parameters
            expert_id: Unique identifier for this expert
        """
        super().__init__()
        self.config = config
        self.expert_id = expert_id

        self.net = nn.Sequential(
            nn.Linear(config.dim + 1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.dim),
        ).to(config.device)

        logger.debug(f"Expert {expert_id} architecture: {self.net}")

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        """Forward pass computing the velocity field for this expert.

        Args:
            t: Time tensor of shape (batch_size, 1)
            x_t: Position tensor of shape (batch_size, dim)

        Returns:
            Velocity field tensor of shape (batch_size, dim)
        """
        input_tensor = torch.cat((t, x_t), -1)
        return self.net(input_tensor)


class MixtureFlow(nn.Module):
    """Mixture of Flows model combining multiple flow experts with a gating network."""

    def __init__(self, config: MixtureFlowConfig):
        """Initialize the Mixture of Flows model.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        logger.info(
            f"Initializing Mixture of Flows model with config: {config}"
        )

        # Initialize gating network
        self.gating = GatingNetwork(config)

        # Initialize flow experts
        self.experts = nn.ModuleList(
            [ExpertFlow(config, i) for i in range(config.n_experts)]
        )

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"logs/mixture_flow_{timestamp}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="500 MB")

    def forward(self, t: Tensor, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass computing the combined velocity field.

        Args:
            t: Time tensor of shape (batch_size, 1)
            x_t: Position tensor of shape (batch_size, dim)

        Returns:
            Tuple of (combined velocity field, expert weights) tensors
        """
        # Get expert selection probabilities
        expert_weights = self.gating(t, x_t)

        # Compute velocity field for each expert
        expert_outputs = torch.stack(
            [expert(t, x_t) for expert in self.experts], dim=1
        )  # Shape: (batch_size, n_experts, dim)

        # Combine expert outputs using weights
        combined_output = torch.sum(
            expert_outputs * expert_weights.unsqueeze(-1), dim=1
        )

        return combined_output, expert_weights

    def step(
        self, x_t: Tensor, t_start: Tensor, t_end: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Perform one step of numerical integration using the midpoint method.

        Args:
            x_t: Current position tensor of shape (batch_size, dim)
            t_start: Start time of the step
            t_end: End time of the step

        Returns:
            Tuple of (new position tensor, expert weights)
        """
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start

        # Midpoint method
        k1, w1 = self(t=t_start, x_t=x_t)
        x_mid = x_t + k1 * dt / 2
        t_mid = t_start + dt / 2
        k2, w2 = self(t=t_mid, x_t=x_mid)

        # Average expert weights across the step
        avg_weights = (w1 + w2) / 2

        return x_t + k2 * dt, avg_weights

    def train_model(
        self, save_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """Train the mixture of flows model.

        Args:
            save_dir: Optional directory to save model checkpoints

        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting model training")
        save_dir = save_dir or Path("checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.Adam(
            self.parameters(), self.config.learning_rate
        )
        loss_fn = nn.MSELoss()

        metrics = {
            "total_loss": [],
            "mse_loss": [],
            "entropy_loss": [],
            "expert_usage": np.zeros(self.config.n_experts),
        }

        for iteration in range(self.config.n_iterations):
            # Generate data
            x_1 = torch.tensor(
                make_moons(
                    self.config.batch_size,
                    noise=self.config.noise_level,
                )[0],
                dtype=torch.float32,
            ).to(self.config.device)
            x_0 = torch.randn_like(x_1).to(self.config.device)
            t = torch.rand(len(x_1), 1).to(self.config.device)

            # Compute target points
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0

            # Training step
            optimizer.zero_grad()
            pred_dx_t, expert_weights = self(t=t, x_t=x_t)

            # Compute losses
            mse_loss = loss_fn(pred_dx_t, dx_t)
            entropy_loss = Categorical(probs=expert_weights).entropy().mean()
            total_loss = (
                mse_loss - 0.1 * entropy_loss
            )  # Encourage expert specialization

            total_loss.backward()
            optimizer.step()

            # Update metrics
            metrics["total_loss"].append(total_loss.item())
            metrics["mse_loss"].append(mse_loss.item())
            metrics["entropy_loss"].append(entropy_loss.item())
            metrics["expert_usage"] += (
                expert_weights.mean(0).cpu().detach().numpy()
            )

            # Logging
            if iteration % self.config.logging_interval == 0:
                self._log_training_status(iteration, metrics)

            # Checkpointing
            if save_dir and iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint(save_dir / f"checkpoint_{iteration}.pt")

        logger.info("Training completed")
        self._log_final_metrics(metrics)
        return metrics

    def _log_training_status(
        self, iteration: int, metrics: Dict[str, Any]
    ) -> None:
        """Log training status.

        Args:
            iteration: Current training iteration
            metrics: Dictionary of training metrics
        """
        logger.info(
            f"Iteration {iteration}: "
            f"Total Loss: {metrics['total_loss'][-1]:.6f}, "
            f"MSE Loss: {metrics['mse_loss'][-1]:.6f}, "
            f"Entropy Loss: {metrics['entropy_loss'][-1]:.6f}"
        )

    def _log_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log final training metrics.

        Args:
            metrics: Dictionary of training metrics
        """
        expert_usage = metrics["expert_usage"] / self.config.n_iterations
        logger.info("Final expert usage distribution:")
        for i, usage in enumerate(expert_usage):
            logger.info(f"Expert {i}: {usage:.3f}")

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "MixtureFlow":
        """Load model from checkpoint.

        Args:
            path: Path to the checkpoint file

        Returns:
            Loaded MixtureFlow model
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Model loaded from {path} (saved at {checkpoint['timestamp']})"
        )
        return model

    def visualize_flow(
        self,
        n_points: int = 300,
        n_steps: int = 8,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize the learned flow transformation with expert contributions.

        Args:
            n_points: Number of points to visualize
            n_steps: Number of integration steps
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (figure, axes array)
        """
        logger.info("Generating flow visualization")
        x = torch.randn(n_points, 2).to(self.config.device)
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        # Create subplot for each timestep plus expert usage plot
        fig, axes = plt.subplots(2, n_steps + 1, figsize=(30, 8))

        # Plot initial distribution
        axes[0, 0].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10)
        axes[0, 0].set_title(f"t = {time_steps[0]:.2f}")
        axes[0, 0].set_xlim(-3.0, 3.0)
        axes[0, 0].set_ylim(-3.0, 3.0)

        # Storage for expert weights over time
        expert_weights_over_time = []

        for i in range(n_steps):
            x, weights = self.step(
                x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1]
            )
            expert_weights_over_time.append(
                weights.mean(0).cpu().detach().numpy()
            )

            # Plot points
            axes[0, i + 1].scatter(
                x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10
            )
            axes[0, i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")

            # Plot expert usage
            expert_weights = np.array(expert_weights_over_time)
            for e in range(self.config.n_experts):
                axes[1, i + 1].bar(e, expert_weights[-1, e])
            axes[1, i + 1].set_title("Expert Usage")
            axes[1, i + 1].set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")

        return fig, axes

    def visualize_expert_specialization(
        self, grid_size: int = 20, save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Visualize how different experts specialize in different regions of the space.

        Args:
            grid_size: Number of points along each dimension in the visualization grid
            save_path: Optional path to save the visualization

        Returns:
            Matplotlib figure object
        """
        logger.info("Generating expert specialization visualization")

        # Create grid of points
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
        ).to(self.config.device)

        # Sample different time points
        time_points = torch.tensor([0.0, 0.33, 0.66, 1.0]).to(
            self.config.device
        )

        fig, axes = plt.subplots(
            len(time_points), 1, figsize=(10, 4 * len(time_points))
        )
        if len(time_points) == 1:
            axes = [axes]

        for t_idx, t in enumerate(time_points):
            t_expanded = t.view(1, 1).expand(len(grid_points), 1)
            expert_weights = self.gating(t_expanded, grid_points)

            # Get dominant expert at each point
            dominant_experts = torch.argmax(expert_weights, dim=1).cpu().numpy()

            # Plot
            scatter = axes[t_idx].scatter(
                grid_points[:, 0].cpu(),
                grid_points[:, 1].cpu(),
                c=dominant_experts,
                cmap="tab10",
                alpha=0.6,
            )
            axes[t_idx].set_title(f"t = {t.item():.2f}")
            axes[t_idx].set_xlim(-3, 3)
            axes[t_idx].set_ylim(-3, 3)

            # Add colorbar
            plt.colorbar(
                scatter,
                ax=axes[t_idx],
                label="Dominant Expert",
                ticks=range(self.config.n_experts),
            )

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(
                f"Expert specialization visualization saved to {save_path}"
            )

        return fig

    def get_expert_statistics(self) -> Dict[str, Any]:
        """Compute statistics about expert usage and specialization.

        Returns:
            Dictionary containing various expert statistics
        """
        logger.info("Computing expert statistics")

        # Generate grid of points for analysis
        x = torch.linspace(-3, 3, 20)
        y = torch.linspace(-3, 3, 20)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(
            self.config.device
        )

        # Sample different time points
        time_points = torch.linspace(0, 1, 10).to(self.config.device)

        expert_stats = {
            "total_usage": (
                torch.zeros(self.config.n_experts).to(self.config.device)
            ),
            "max_confidence": (
                torch.zeros(self.config.n_experts).to(self.config.device)
            ),
            "avg_confidence": (
                torch.zeros(self.config.n_experts).to(self.config.device)
            ),
            "territory_size": (
                torch.zeros(self.config.n_experts).to(self.config.device)
            ),
        }

        total_points = len(grid_points) * len(time_points)

        for t in time_points:
            t_expanded = t.view(1, 1).expand(len(grid_points), 1)
            expert_weights = self.gating(t_expanded, grid_points)

            # Update statistics
            expert_stats["total_usage"] += expert_weights.sum(dim=0)
            expert_stats["max_confidence"] = torch.maximum(
                expert_stats["max_confidence"],
                expert_weights.max(dim=0)[0],
            )
            expert_stats["avg_confidence"] += expert_weights.sum(dim=0)
            expert_stats["territory_size"] += (
                (expert_weights == expert_weights.max(dim=1, keepdim=True)[0])
                .float()
                .sum(dim=0)
            )

        # Normalize statistics
        expert_stats["total_usage"] /= total_points
        expert_stats["avg_confidence"] /= total_points
        expert_stats["territory_size"] /= total_points

        # Convert to numpy for easier handling
        return {k: v.cpu() for k, v in expert_stats.items()}


# # Example usage
# if __name__ == "__main__":
#     # Setup configuration
#     config = MixtureFlowConfig()

#     # Initialize and train model
#     model = MixtureFlow(config)
#     training_metrics = model.train_model(save_dir=Path("checkpoints"))

#     # Visualize results
#     model.visualize_flow(
#         save_path=Path("visualizations/flow_evolution.png")
#     )
#     model.visualize_expert_specialization(
#         save_path=Path("visualizations/expert_specialization.png")
#     )

#     # Get and log expert statistics
#     expert_stats = model.get_expert_statistics()
#     logger.info("\nExpert Statistics:")
#     for stat_name, values in expert_stats.items():
#         logger.info(f"\n{stat_name}:")
#         for expert_id, value in enumerate(values):
#             logger.info(f"Expert {expert_id}: {value:.3f}")
