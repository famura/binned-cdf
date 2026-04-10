import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pytest
import seaborn as sns
import torch

from binned_cdf import BezierCDF, PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF
from tests.conftest import needs_cuda


@pytest.mark.parametrize("distr_class", [BezierCDF, PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF])
@pytest.mark.parametrize("logit_scale", [1e-3, 1, 1e3, 1e9])
@pytest.mark.parametrize("normalization_method", ["sigmoid", "softmax"], ids=["sigmoid", "softmax"])
@pytest.mark.parametrize("batch_size", [None, 1, 8])
@pytest.mark.parametrize(
    "use_cuda,plot",
    [
        pytest.param(False, True, marks=pytest.mark.plot, id="cpu_visual"),
        pytest.param(False, False, id="cpu_non-visual"),
        pytest.param(True, False, marks=needs_cuda, id="cuda_non-visual"),
    ],
)
def test_cdf_random_logits(
    distr_class: type[BezierCDF] | type[PiecewiseConstantBinnedCDF] | type[PiecewiseLinearBinnedCDF],
    logit_scale: float,
    batch_size: int | None,
    normalization_method: Literal["sigmoid", "softmax"],
    use_cuda: bool,
    plot: bool,
    bound_low: float = -10,
    bound_up: float = 10,
    num_bins: int = 100,
):
    """Test CDF evaluation with random logits at the bounds."""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = logit_scale * torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)

    distr = distr_class(logits, bound_low, bound_up, normalization_method=normalization_method)

    # Evaluate the CDF at the bounds.
    cdf_low = distr.cdf(torch.tensor(bound_low))
    cdf_up = distr.cdf(torch.tensor(bound_up))

    # Check the values at the bounds.
    assert torch.all(cdf_low <= 1.0 / num_bins), f"CDF at lower bound not <= 1/num_bins: {cdf_low}"
    assert torch.allclose(cdf_up, torch.ones_like(cdf_up)), f"CDF at upper bound not 1: {cdf_up}"

    if plot and batch_size is None:
        x = torch.linspace(bound_low, bound_up, 2000)
        cdf_vals = distr.cdf(x)
        plt.figure(figsize=(8, 5))
        plt.plot(x.numpy(), cdf_vals.numpy())
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title(f"CDF for random logits scaled by {logit_scale}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        results_dir = Path("tests/results/test_cdf_icdf")
        results_dir.mkdir(parents=True, exist_ok=True)
        class_suffix = "const" if distr_class is PiecewiseConstantBinnedCDF else "linear"
        plt.savefig(
            results_dir
            / f"cdf_random_logits_scale-{logit_scale}_normalization-{normalization_method}_{class_suffix}.png",
            bbox_inches="tight",
        )


@pytest.mark.parametrize("distr_class", [PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF])
@pytest.mark.parametrize("sample_batch_size", [None, 1, 8])
@pytest.mark.parametrize("distr_batch_size", [1, 3])
@pytest.mark.parametrize(
    "use_cuda,plot",
    [
        pytest.param(False, True, marks=pytest.mark.plot, id="cpu_visual"),
        pytest.param(False, False, id="cpu_non-visual"),
        pytest.param(True, False, marks=needs_cuda, id="cuda_non-visual"),
    ],
)
def test_sampling_and_cdf_consistency(
    distr_class: type[BezierCDF] | type[PiecewiseConstantBinnedCDF] | type[PiecewiseLinearBinnedCDF],
    sample_batch_size: int | None,  # number of samples to draw per distribution
    distr_batch_size: int,  # number of independent distributions to sample from
    use_cuda: bool,
    plot: bool,
    num_samples: int = 1000,
    bound_low: float = -3.0,
    bound_up: float = 3.0,
    num_bins: int = 20,
    abs_tol_per_bin: float = 0.08,
):
    """Test that samples follow the PiecewiseConstantBinnedCDF's CDF and have the correct shape."""
    torch.manual_seed(42)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create random distribution and sample from it.
    logits = torch.randn(distr_batch_size, num_bins, device=device)
    distr = distr_class(logits, bound_low, bound_up)
    sample_shape = (num_samples,) if sample_batch_size is None else (sample_batch_size, num_samples)
    samples = distr.sample(sample_shape)
    assert samples.device == device
    assert samples.shape == (*sample_shape, distr_batch_size)

    # Test that samples are within bounds.
    assert torch.all(samples >= bound_low)
    assert torch.all(samples <= bound_up)

    # For multiple distributions, we need to test each distribution separately.
    test_points = torch.linspace(bound_low, bound_up, num_bins)
    theoretical_cdf_for_plot = torch.empty(0)
    for batch_idx in range(distr_batch_size):
        # Extract samples for this specific distribution.
        batch_samples = samples.squeeze(-1) if distr_batch_size == 1 else samples[..., batch_idx]

        # Create a single-distribution version for easier CDF evaluation.
        single_logits = distr.logits[batch_idx : batch_idx + 1]  # Keep batch dimension of size 1
        single_distr = distr_class(single_logits, bound_low, bound_up)
        theoretical_cdf = single_distr.cdf(test_points).squeeze(0)  # remove batch dimension

        # Store for plotting (use the first distribution).
        if batch_idx == 0 and distr_batch_size == 1:
            theoretical_cdf_for_plot = theoretical_cdf

        for i, point in enumerate(test_points):
            empirical_cdf = (batch_samples <= point).float().mean()
            assert abs(theoretical_cdf[i] - empirical_cdf) < abs_tol_per_bin

    if plot and sample_batch_size is None and distr_batch_size == 1:
        plt.figure(figsize=(8, 5))
        plot_samples = samples.squeeze(-1)
        sns.histplot(
            plot_samples.numpy(),
            bins=30,
            stat="density",
            cumulative=True,
            label="Empirical CDF",
            color="blue",
            alpha=0.6,
        )
        plt.plot(
            test_points.numpy(), theoretical_cdf_for_plot.numpy(), label="Theoretical CDF", color="red", linewidth=2
        )
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title("Empirical vs Theoretical CDF")
        plt.legend()
        plt.grid(True, alpha=0.3)
        class_suffix = "const" if distr_class is PiecewiseConstantBinnedCDF else "linear"
        plt.savefig(f"tests/results/sampling_and_cdf_consistency_{class_suffix}.png", bbox_inches="tight")


@pytest.mark.parametrize("distr_class", [PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF])
@pytest.mark.parametrize("batch_size", [None, 1, 8, 16])
@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_icdf_random_quantiles(
    distr_class: type[BezierCDF] | type[PiecewiseConstantBinnedCDF] | type[PiecewiseLinearBinnedCDF],
    batch_size: int | None,
    use_cuda: bool,
    bound_low: float = -10,
    bound_up: float = 10,
    num_bins: int = 200,
    num_quantiles: int = 50,
):
    """Test ICDF evaluation with random quantiles - basic value and shape checking."""
    torch.manual_seed(42)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create random logits and quantiles.
    logits = torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)
    quantiles = torch.rand(num_quantiles, device=device)

    # Compute ICDF.
    distr = distr_class(logits, bound_low, bound_up)
    if batch_size is not None:
        # For batched distributions, expand quantiles to (num_quantiles, *batch_shape)
        quantiles = quantiles.unsqueeze(-1).expand(num_quantiles, batch_size)
    icdf_values = distr.icdf(quantiles)

    # Check shape and device. Output shape should match input shape: (*sample_shape, *batch_shape)
    expected_shape = (num_quantiles,) if batch_size is None else (num_quantiles, batch_size)
    assert icdf_values.shape == expected_shape, f"ICDF shape mismatch: {icdf_values.shape} vs {expected_shape}"
    assert icdf_values.device == device, f"ICDF device mismatch: {icdf_values.device} vs {device}"

    # Check values are within bounds.
    assert torch.all(icdf_values >= bound_low), f"ICDF values below lower bound: min={icdf_values.min()}"
    assert torch.all(icdf_values <= bound_up), f"ICDF values above upper bound: max={icdf_values.max()}"

    # Check boundary quantiles.
    icdf_at_0 = distr.icdf(torch.tensor(0.0, device=device))
    icdf_at_1 = distr.icdf(torch.tensor(1.0, device=device))
    assert torch.all(icdf_at_0 >= bound_low - 1e-5), f"ICDF(0) should be >= bound_low: {icdf_at_0}"
    assert torch.all(icdf_at_1 <= bound_up + 1e-5), f"ICDF(1) should be <= bound_up: {icdf_at_1}"


@pytest.mark.parametrize("distr_class", [PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF])
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
@pytest.mark.parametrize(
    "use_cuda,plot",
    [
        pytest.param(False, True, marks=pytest.mark.plot, id="cpu_visual"),
        pytest.param(False, False, id="cpu_non-visual"),
        pytest.param(True, False, marks=needs_cuda, id="cuda_non-visual"),
    ],
)
def test_icdf_fixed_quantiles(
    distr_class: type[BezierCDF] | type[PiecewiseConstantBinnedCDF] | type[PiecewiseLinearBinnedCDF],
    log_spacing: bool,
    use_cuda: bool,
    plot: bool,
    bound_low: float = -5.0,
    bound_up: float = 5.0,
    num_bins: int = 500,
):
    """Test inverse CDF at fixed quantiles and verify round-trip property: cdf(icdf(q)) ≈ q."""
    torch.manual_seed(42)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create a distribution with random logits.
    logits = torch.randn(num_bins, device=device)
    extra_init_kwargs: dict[str, Any] = {"log_spacing": log_spacing} if distr_class is PiecewiseLinearBinnedCDF else {}
    distr = distr_class(logits, bound_low, bound_up, **extra_init_kwargs)

    # Test fixed quantiles with both linear and log spacing for the quantiles themselves.
    quantiles = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=device)

    # Compute inverse CDF at these quantiles.
    icdf_values = distr.icdf(quantiles)

    # Verify that all icdf values are within bounds.
    assert torch.all(icdf_values >= bound_low), f"ICDF values below lower bound: min={icdf_values.min()}"
    assert torch.all(icdf_values <= bound_up), f"ICDF values above upper bound: max={icdf_values.max()}"

    # Test the round-trip property: cdf(icdf(q)) ≈ q.
    cdf_roundtrip = distr.cdf(icdf_values)
    if isinstance(distr, PiecewiseConstantBinnedCDF):
        atol = 2 / num_bins
    elif isinstance(distr, PiecewiseLinearBinnedCDF):
        atol = 1e-6
    else:
        raise TypeError("Unknown distribution class for setting atol")
    torch.testing.assert_close(
        cdf_roundtrip,
        quantiles,
        rtol=1e-3,
        atol=atol,
        msg="Round-trip cdf(icdf(q)) != q. This highly depends on the number of bins.",
    )

    # Test monotonicity: icdf should be non-decreasing.
    icdf_diffs = icdf_values[1:] - icdf_values[:-1]
    assert torch.all(icdf_diffs >= -1e-6), "ICDF is not monotonic"

    if plot:
        # Create visualization with multiple panels.
        _, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: ICDF function.
        ax1 = axes[0, 0]
        q_dense = torch.linspace(0, 1, 500)
        icdf_dense = distr.icdf(q_dense)
        ax1.plot(q_dense.numpy(), icdf_dense.numpy(), linewidth=2, label="ICDF")
        ax1.scatter(quantiles.numpy(), icdf_values.numpy(), color="red", s=30, alpha=0.6, label="Test quantiles")
        ax1.set_xlabel("Quantile (q)")
        ax1.set_ylabel("Value (icdf(q))")
        ax1.set_title("Inverse CDF (Quantile Function)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 2: CDF for reference.
        ax2 = axes[0, 1]
        x_dense = torch.linspace(bound_low, bound_up, 500)
        cdf_dense = distr.cdf(x_dense)
        ax2.plot(x_dense.numpy(), cdf_dense.numpy(), linewidth=2, label="CDF")
        ax2.scatter(icdf_values.numpy(), quantiles.numpy(), color="red", s=30, alpha=0.6, label="(icdf(q), q)")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("CDF")
        ax2.set_title("Cumulative Distribution Function")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Panel 3: Round-trip error.
        ax3 = axes[1, 0]
        roundtrip_error = (cdf_roundtrip - quantiles).numpy()
        ax3.scatter(quantiles.numpy(), roundtrip_error, s=30, alpha=0.7)
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Quantile (q)")
        ax3.set_ylabel("Error: cdf(icdf(q)) - q")
        ax3.set_title("Round-Trip Error")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Distribution properties.
        ax4 = axes[1, 1]
        ax4.axis("off")
        properties_text = (
            f"Distribution Properties:\n"
            f"{'=' * 40}\n"
            f"Num bins: {num_bins}\n"
            f"Log spacing: {log_spacing}\n"
            f"Bounds: [{bound_low}, {bound_up}]\n"
            f"Mean: {distr.mean.item():.3f}\n"
            f"Std: {math.sqrt(distr.variance.item()):.3f}\n"
            f"Entropy: {distr.entropy().item():.3f}\n\n"
            f"Test Results:\n"
            f"{'=' * 40}\n"
            f"Quantiles tested: {len(quantiles)}\n"
            f"Max round-trip error: {roundtrip_error.max():.6f}\n"
            f"Mean abs round-trip error: {abs(roundtrip_error).mean():.6f}\n"
            f"ICDF range: [{icdf_values.min():.3f}, {icdf_values.max():.3f}]"
        )
        ax4.text(0.1, 0.5, properties_text, fontsize=10, verticalalignment="center", family="monospace")

        class_suffix = "const" if distr_class is PiecewiseConstantBinnedCDF else "linear"
        spacing_suffix = "log-spacing" if log_spacing else "linear-spacing"
        plt.suptitle(f"ICDF Test with Fixed Quantiles ({spacing_suffix})", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"tests/results/icdf_fixed_quantiles_{class_suffix}_{spacing_suffix}.png", bbox_inches="tight", dpi=100
        )
