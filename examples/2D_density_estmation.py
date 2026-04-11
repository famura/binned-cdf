from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import make_moons

from binned_cdf import BezierCDF, PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF

sns.set_theme()

NewDistributionType = type[PiecewiseLinearBinnedCDF] | type[PiecewiseConstantBinnedCDF] | type[BezierCDF]
NewDistribution = PiecewiseLinearBinnedCDF | PiecewiseConstantBinnedCDF | BezierCDF


class DensityNet(torch.nn.Module):
    """Neural network for 2D density estimation using PiecewiseLinearBinnedCDF."""

    def __init__(
        self,
        num_bins: int,
        distr_class: NewDistributionType,
        normalization_method: Literal["sigmoid", "softmax"],
        bound_low: float,
        bound_up: float,
    ) -> None:
        """Initialize the network.

        Args:
            num_bins: Number of bins for the CDF.
            distr_class: The distribution class to use for the output. One of the new distribution types introduced
                in this package.
            normalization_method: Normalization method for the distribution ("sigmoid" or "softmax").
            bound_low: Lower bound of the distribution support.
            bound_up: Upper bound of the distribution support.
        """
        super().__init__()
        self.num_bins = num_bins
        self.distr_class = distr_class
        self.normalization_method: Literal["sigmoid", "softmax"] = normalization_method
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(64, 2 * num_bins)

    def forward(self, x: torch.Tensor) -> NewDistribution:
        """Forward pass to create distribution.

        Args:
            x: Input coordinates of shape (batch_size, 2).

        Returns:
            One of the distributions introduced in this package with batch_shape (batch_size, 2).
        """
        features = self.shared(x)
        logits = self.head(features)
        logits = logits.reshape(*logits.shape[:-1], 2, self.num_bins)
        return self.distr_class(
            logits, bound_low=self.bound_low, bound_up=self.bound_up, normalization_method=self.normalization_method
        )


if __name__ == "__main__":
    """Main function to execute the density estimation example."""
    # Configure.
    bound_low_2moons = -1.5  # prior knowledge about the dataset
    bound_up_2moons = 2.5  # prior knowledge about the dataset
    distr_class: NewDistributionType = PiecewiseLinearBinnedCDF
    num_bins = 100
    normalization_method: Literal["softmax", "sigmoid"] = "softmax" if distr_class == BezierCDF else "sigmoid"

    # Create ground truth data.
    X, _ = make_moons(n_samples=1500, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32)

    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X = X.to(device)

    # Create the model and optimizer.
    model = DensityNet(
        num_bins=num_bins,
        distr_class=distr_class,
        normalization_method=normalization_method,
        bound_low=bound_low_2moons,
        bound_up=bound_up_2moons,
    )
    model = model.to(device)
    lr = 1e-3 if distr_class == BezierCDF else 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_iter = 6000
    num_grid_points = 200
    torch.manual_seed(0)

    print("Training started.")
    for epoch in range(num_iter):
        optimizer.zero_grad()
        distr = model(X)
        log_prob = distr.log_prob(X)
        loss = -log_prob.sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_iter}, Loss: {loss.item():.4f}")
    print("Training finished.")

    xx, yy = np.meshgrid(
        np.linspace(bound_low_2moons, bound_up_2moons, num_grid_points),
        np.linspace(bound_low_2moons, bound_up_2moons, num_grid_points),
    )
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32).to(device)

    print("Gird evaluation started.")
    with torch.no_grad():
        distr = model(grid)  # grid shape: (N, 2)
        probs = distr.prob(grid)  # probs shape: (N, 2)
        prob_x, prob_y = probs[:, 0], probs[:, 1]
        prob_joint = (prob_x * prob_y).cpu().numpy().reshape(xx.shape)
    print(f"Grid evaluation finished. Evaluation of the joint on the grid has shape {prob_joint.shape}.")

    sns.set_theme()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].contourf(xx, yy, prob_joint, levels=30, cmap="viridis")
    axes[0].scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=4, color="red", alpha=0.3)
    axes[0].set_title("Estimated Density (PiecewiseLinearBinnedCDF)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    print("Grid plotting finished.")

    print("Sampling started.")
    with torch.no_grad():
        distr = model(X)  # create distribution for all training data points
        samples = distr.sample()
    print(f"Sampling finished. Samples have shape {samples.shape}.")

    axes[1].scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), s=4, alpha=0.5, label="sampled")
    axes[1].scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=4, color="red", alpha=0.3, label="true data")
    axes[1].set_title("Samples from Learned Distribution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(loc="upper right")
    print("Samples plot plotting finished.")

    fig.tight_layout()
    fig.savefig("examples/2D_density_estimation_result.png", dpi=300, bbox_inches="tight")
    print("Plot saved to examples/2D_density_estimation_result.png")
