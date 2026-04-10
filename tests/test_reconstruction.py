import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import torch
from scipy import stats

from binned_cdf import BezierCDF, PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF


@pytest.mark.parametrize(
    "target_dist_params",
    [
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 3.0, "scale": 2.0},
                "bounds": (-10.0, 10.0),
                "tolerances": {"mean": 0.2, "std": 0.25},
            },
            id="normal",
        ),
        pytest.param(
            {
                "dist": torch.distributions.Exponential,
                "params": {"rate": 0.5},
                "bounds": (0.0, 15.0),
                "tolerances": {"mean": 0.5, "std": 0.15},
            },
            id="exponential",
        ),
    ],
)
@pytest.mark.parametrize(
    "plot",
    [
        pytest.param(True, marks=pytest.mark.plot, id="visual"),
        pytest.param(False, id="non-visual"),
    ],
)
@pytest.mark.parametrize("dim_logits", [40, 120], ids=["dim_logits_40", "dim_logits_120"])
@pytest.mark.parametrize("normalization_method", ["softmax", "sigmoid"], ids=["softmax", "sigmoid"])
def test_distribution_reconstruction_bezier(
    target_dist_params: dict,
    plot: bool,
    dim_logits: int,
    normalization_method: Literal["softmax", "sigmoid"],
):
    """Test reconstruction of different distributions using BezierCDF.

    The tolerances account for the slow O(1/n) convergence of Bernstein polynomials. For example, with dim_logits=40
    the std error for a Normal distribution is ~20%, roughly halving when the degree doubles (e.g., ~10% at 80, ~7%
    at 120).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract parameters from the parametrized input, and create target distribution.
    dist_class = target_dist_params["dist"]
    dist_params = target_dist_params["params"]
    bound_low, bound_up = target_dist_params["bounds"]
    tolerances = target_dist_params["tolerances"]
    target_distr = dist_class(**dist_params)

    # Get distribution properties for validation.
    target_mean = target_distr.mean.item()
    target_std = target_distr.stddev.item()

    # Use evenly-spaced bin centers (BezierCDF has no _create_bins).
    eval_points = torch.linspace(bound_low, bound_up, dim_logits)

    # Compute target probabilities at evaluation point.
    target_probs = torch.exp(target_distr.log_prob(eval_points))

    # With softmax normalization, logits = log(steps) up to a constant.
    logits = torch.log(target_probs + 1e-8)

    # Create BezierCDF distribution, and get mean and variance.
    distr = BezierCDF(logits=logits, bound_low=bound_low, bound_up=bound_up, normalization_method=normalization_method)
    reconstructed_mean = distr.mean.item()
    reconstructed_var = distr.variance.item()
    reconstructed_std = math.sqrt(reconstructed_var)

    # Check if mean and std are reasonably close (within specified tolerance).
    torch.testing.assert_close(
        reconstructed_mean,
        target_mean,
        rtol=tolerances["mean"],
        atol=0.0,
        msg=f"Mean mismatch: reconstructed={reconstructed_mean:.3f}, target={target_mean:.3f}",
    )
    torch.testing.assert_close(
        reconstructed_std,
        target_std,
        rtol=tolerances["std"],
        atol=0.0,
        msg=f"Std mismatch: reconstructed={reconstructed_std:.3f}, target={target_std:.3f}",
    )

    # Generate samples for statistical tests.
    n_samples = 10_000
    original_samples = target_distr.sample((n_samples,))
    reconstructed_samples = distr.sample((n_samples,)).squeeze()

    if plot:
        # Create comparison plot.
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(original_samples.numpy(), bins=50, alpha=0.7, label=f"Original {dist_class.__name__}")
        sns.histplot(reconstructed_samples.numpy(), bins=50, alpha=0.7, label="Reconstructed")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot CDFs.
        plt.subplot(1, 2, 2)
        x_range = torch.linspace(bound_low, bound_up, 1000)
        original_cdf = target_distr.cdf(x_range)
        reconstructed_cdf = distr.cdf(x_range)
        plt.plot(x_range.numpy(), original_cdf.numpy(), label="Original CDF", linewidth=2)
        plt.plot(x_range.numpy(), reconstructed_cdf.numpy(), label="Reconstructed CDF", linewidth=2, linestyle="--")
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title("CDF Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        results_dir = Path("tests/results/test_reconstruction")
        results_dir.mkdir(parents=True, exist_ok=True)
        dist_name = dist_class.__name__.lower()
        plt.savefig(
            results_dir / f"{dist_name}_bezier_dim-{dim_logits}_norm-{normalization_method}.png", bbox_inches="tight"
        )

    # Run the Kolmogorov-Smirnov test which looks at the maximum difference between CDFs.
    ks_statistic, ks_p_value = stats.ks_2samp(original_samples.numpy(), reconstructed_samples.numpy())

    if plot:
        mean_error = abs(reconstructed_mean - target_mean) / target_mean
        std_error = abs(reconstructed_std - target_std) / target_std
        print(f"Original: mean={target_mean:.3f}, std={target_std:.3f}")
        print(f"Reconstructed: mean={reconstructed_mean:.3f}, std={reconstructed_std:.3f}")
        print(f"Mean error: {mean_error:.3f}, Std error: {std_error:.3f}")
        print(f"KS test: statistic={ks_statistic:.4f}, p-value={ks_p_value:.4f}")


@pytest.mark.parametrize("distr_class", [PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF])
@pytest.mark.parametrize(
    "target_dist_params",
    [
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 3.0, "scale": 2.0},
                "bounds": (-10.0, 10.0),
                "tolerances": {"mean": 0.2, "std": 0.2},
            },
            id="normal",
        ),
        pytest.param(
            {
                "dist": torch.distributions.Exponential,
                "params": {"rate": 0.5},
                "bounds": (0.0, 15.0),
                "tolerances": {"mean": 0.5, "std": 0.15},
            },
            id="exponential",
        ),
    ],
)
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
@pytest.mark.parametrize(
    "plot",
    [
        pytest.param(True, marks=pytest.mark.plot, id="visual"),
        pytest.param(False, id="non-visual"),
    ],
)
def test_distribution_reconstruction_binned(
    distr_class: type[PiecewiseConstantBinnedCDF] | type[PiecewiseLinearBinnedCDF],
    target_dist_params: dict,
    log_spacing: bool,
    plot: bool,
    num_bins: int = 80,
):
    """Test reconstruction of different distributions using PiecewiseConstantBinnedCDF and PiecewiseLinearBinnedCDF."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract parameters from the parametrized input, and create target distribution.
    dist_class = target_dist_params["dist"]
    dist_params = target_dist_params["params"]
    bound_low, bound_up = target_dist_params["bounds"]
    tolerances = target_dist_params["tolerances"]
    target_distr = dist_class(**dist_params)

    # Skip log spacing tests for incompatible bounds.
    if log_spacing and not math.isclose(-bound_low, bound_up):
        pytest.skip("log_spacing requires symmetric bounds")
    if log_spacing and bound_up <= 0:
        pytest.skip("log_spacing requires positive upper bound")
    if log_spacing and num_bins % 2 != 0:
        pytest.skip("log_spacing requires even number of bins")

    # Get distribution properties for validation.
    target_mean = target_distr.mean.item()
    target_std = target_distr.stddev.item()

    # Use the distr_class's own bin construction to ensure matching shapes between distributions.
    _, bin_centers, bin_widths = distr_class._create_bins(
        num_bins=num_bins,
        bound_low=bound_low,
        bound_up=bound_up,
        log_spacing=log_spacing,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Compute target probabilities at bin centers.
    target_probs = torch.exp(target_distr.log_prob(bin_centers))

    # Normalize to get probability masses for each bin.
    target_prob_masses = target_probs * bin_widths
    target_prob_masses = target_prob_masses / target_prob_masses.sum()

    # Convert probabilities to logits (inverse sigmoid).
    eps = 1e-8
    target_prob_masses = torch.clamp(target_prob_masses, eps, 1 - eps)
    logits = torch.log(target_prob_masses / (1 - target_prob_masses))

    # Create the distribution, and get mean and variance.
    distr = distr_class(logits=logits, bound_low=bound_low, bound_up=bound_up, log_spacing=log_spacing)
    reconstructed_mean = distr.mean.item()
    reconstructed_var = distr.variance.item()
    reconstructed_std = math.sqrt(reconstructed_var)

    # Check if mean and std are reasonably close (within specified tolerance).
    torch.testing.assert_close(
        reconstructed_mean,
        target_mean,
        rtol=tolerances["mean"],
        atol=0.0,
        msg=f"Mean mismatch: reconstructed={reconstructed_mean:.3f}, target={target_mean:.3f}",
    )
    torch.testing.assert_close(
        reconstructed_std,
        target_std,
        rtol=tolerances["std"],
        atol=0.0,
        msg=f"Std mismatch: reconstructed={reconstructed_std:.3f}, target={target_std:.3f}",
    )

    # Generate samples for statistical tests.
    n_samples = 10_000
    original_samples = target_distr.sample((n_samples,))
    reconstructed_samples = distr.sample((n_samples,)).squeeze()

    if plot:
        # Create comparison plot.
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(original_samples.numpy(), bins=50, alpha=0.7, label=f"Original {dist_class.__name__}")
        sns.histplot(reconstructed_samples.numpy(), bins=50, alpha=0.7, label="Reconstructed")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot CDFs.
        plt.subplot(1, 2, 2)
        x_range = torch.linspace(bound_low, bound_up, 1000)
        original_cdf = target_distr.cdf(x_range)
        reconstructed_cdf = distr.cdf(x_range)
        plt.plot(x_range.numpy(), original_cdf.numpy(), label="Original CDF", linewidth=2)
        plt.plot(x_range.numpy(), reconstructed_cdf.numpy(), label="Reconstructed CDF", linewidth=2, linestyle="--")
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title("CDF Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        results_dir = Path("tests/results/test_reconstruction")
        results_dir.mkdir(parents=True, exist_ok=True)
        dist_name = dist_class.__name__.lower()
        spacing_suffix = "log-init" if log_spacing else "linear-init"
        class_suffix = "const" if distr_class is PiecewiseConstantBinnedCDF else "linear"
        plt.savefig(results_dir / f"{dist_name}_binned_{spacing_suffix}_{class_suffix}.png", bbox_inches="tight")

    # Run the Kolmogorov-Smirnov test which looks at the maximum difference between CDFs.
    ks_statistic, ks_p_value = stats.ks_2samp(original_samples.numpy(), reconstructed_samples.numpy())

    if plot:
        mean_error = abs(reconstructed_mean - target_mean) / target_mean
        std_error = abs(reconstructed_std - target_std) / target_std
        print(f"Original: mean={target_mean:.3f}, std={target_std:.3f}")
        print(f"Reconstructed: mean={reconstructed_mean:.3f}, std={reconstructed_std:.3f}")
        print(f"Mean error: {mean_error:.3f}, Std error: {std_error:.3f}")
        print(f"KS test: statistic={ks_statistic:.4f}, p-value={ks_p_value:.4f}")
