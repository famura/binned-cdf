import argparse
from pathlib import Path
from typing import Any, Literal

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
        extra_init_kwargs: dict[str, Any] = {}
        if self.distr_class is BezierCDF:
            # Split the bins between the vertical and horizontal control-point increments.
            logits, extra_init_kwargs["logits_x"] = logits.chunk(2, dim=-1)
        return self.distr_class(
            logits,
            bound_low=self.bound_low,
            bound_up=self.bound_up,
            normalization_method=self.normalization_method,
            **extra_init_kwargs,
        )


if __name__ == "__main__":
    """Main function to execute the density estimation example."""
    distr_classes: dict[str, NewDistributionType] = {
        cls.__name__: cls for cls in (PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF, BezierCDF)
    }
    parser = argparse.ArgumentParser(description="Fit a 2D density estimator to two moons.")
    parser.add_argument("--distr-class", choices=distr_classes, default=PiecewiseLinearBinnedCDF.__name__)
    parser.add_argument("--bound-low", type=float, default=-1.5)
    parser.add_argument("--bound-up", type=float, default=2.5)
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument("--num-iter", type=int, default=6000)
    parser.add_argument("--grid-points", type=int, default=200)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--normalization-method", choices=("softmax", "sigmoid"))
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/2D_density_estimation_result.png"
    )
    args = parser.parse_args()
    if args.num_iter < 1 or args.num_bins < 2 or args.grid_points < 2 or args.bound_low >= args.bound_up:
        parser.error("Require positive iterations, at least two bins/grid points, and bound_low < bound_up.")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    distr_class = distr_classes[args.distr_class]
    normalization_method: Literal["softmax", "sigmoid"] = args.normalization_method or (
        "softmax" if distr_class is BezierCDF else "sigmoid"
    )

    # Create ground truth data.
    X, _ = make_moons(n_samples=1500, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32)

    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X = X.to(device)

    # Create the model and optimizer.
    model = DensityNet(
        num_bins=args.num_bins,
        distr_class=distr_class,
        normalization_method=normalization_method,
        bound_low=args.bound_low,
        bound_up=args.bound_up,
    )
    model = model.to(device)
    lr = args.learning_rate if args.learning_rate is not None else (1e-3 if distr_class is BezierCDF else 5e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(0)

    print("Training started.")
    for epoch in range(args.num_iter):
        optimizer.zero_grad()
        distr = model(X)
        log_prob = distr.log_prob(X)
        loss = -log_prob.sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{args.num_iter}, Loss: {loss.item():.4f}")
    print("Training finished.")

    xx, yy = np.meshgrid(
        np.linspace(args.bound_low, args.bound_up, args.grid_points),
        np.linspace(args.bound_low, args.bound_up, args.grid_points),
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
    axes[0].set_title(f"Estimated Density ({distr_class.__name__})")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
