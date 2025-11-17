import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import make_moons

from binned_cdf.binned_logit_cdf import BinnedLogitCDF

sns.set_theme()


class DensityNet(torch.nn.Module):
    """Neural network for 2D density estimation using BinnedLogitCDF."""

    def __init__(self, num_bins: int) -> None:
        """Initialize the network.

        Args:
            num_bins: Number of bins for the CDF.
        """
        super().__init__()
        self.num_bins = num_bins
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(64, 2 * num_bins)

    def forward(self, x: torch.Tensor) -> BinnedLogitCDF:
        """Forward pass to create distribution.

        Args:
            x: Input coordinates of shape (batch_size, 2).

        Returns:
            BinnedLogitCDF distribution with batch_shape (batch_size, 2).
        """
        features = self.shared(x)
        logits = self.head(features)
        logits = logits.reshape(*logits.shape[:-1], 2, self.num_bins)
        dist = BinnedLogitCDF(logits, bound_low=-2.0, bound_up=3.0)
        return dist


# Create ground truth data.
X, _ = make_moons(n_samples=1000, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)

# Use CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
X = X.to(device)

# Create the model and optimizer.
model = DensityNet(num_bins=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
num_iter = 1500
num_grid_points = 150
torch.manual_seed(0)

print("Training started.")
for epoch in range(num_iter):
    optimizer.zero_grad()
    dist = model(X)
    log_prob = dist.log_prob(X)
    loss = -log_prob.sum(dim=-1).mean()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_iter}, Loss: {loss.item():.4f}")
print("Training finished.")

xx, yy = np.meshgrid(np.linspace(-2, 3, num_grid_points), np.linspace(-1.5, 2, num_grid_points))
grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32).to(device)

print("Gird evaluation started.")
with torch.no_grad():
    dist = model(grid)
    prob_x = dist.prob(grid[:, 0].unsqueeze(-1))[:, 0]
    prob_y = dist.prob(grid[:, 1].unsqueeze(-1))[:, 1]
    prob_joint = (prob_x * prob_y).cpu().numpy().reshape(xx.shape)
print(f"Grid evaluation finished. Evaluation of the joint on the grid has shape {prob_joint.shape}.")

sns.set_theme()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].contourf(xx, yy, prob_joint, levels=30, cmap="viridis")
axes[0].scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=4, color="red", alpha=0.3)
axes[0].set_title("Estimated Density (BinnedLogitCDF)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
print("Grid plotting finished.")

print("Sampling started.")
with torch.no_grad():
    dist = model(X)  # create distribution for all training data points
    samples = dist.sample()
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
