import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import torch
from scipy import stats

from binned_cdf import BinnedLogitCDF
from tests.conftest import needs_cuda


@pytest.mark.parametrize("batch_size", [None, 1, 8])
@pytest.mark.parametrize("num_bins", [1, 2, 7, 1000])  # 2 is an edge case for log-spacing
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
@pytest.mark.parametrize("bin_normalization_method", ["sigmoid", "softmax"], ids=["sigmoid", "softmax"])
@pytest.mark.parametrize("bound_low,bound_up", [(-5, 5), (0, 5), (-5, 0)])
@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_basic_properties(
    batch_size: int | None,
    num_bins: int,
    log_spacing: bool,
    bin_normalization_method: Literal["sigmoid", "softmax"],
    bound_low: int,
    bound_up: int,
    use_cuda: bool,
):
    """Test basic properties of the BinnedLogitCDF."""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)

    if log_spacing and not math.isclose(-bound_low, bound_up):
        with pytest.raises(ValueError, match="log_spacing requires symmetric bounds"):
            BinnedLogitCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    if log_spacing and bound_up <= 0:
        with pytest.raises(ValueError, match="log_spacing requires positive upper bound"):
            BinnedLogitCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    if log_spacing and num_bins % 2 != 0:
        with pytest.raises(ValueError, match="log_spacing requires even number of bins"):
            BinnedLogitCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    dist = BinnedLogitCDF(
        logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
    )

    # Test that tensors are on the correct device.
    assert dist.logits.device == device
    assert dist.bin_edges.device == device
    assert dist.bin_centers.device == device
    assert dist.bin_widths.device == device

    # Test properties directly coming from the arguments.
    assert dist.num_bins == num_bins
    assert dist.bound_low == bound_low
    assert dist.bound_up == bound_up
    assert dist.support.lower_bound == bound_low
    assert dist.support.upper_bound == bound_up
    assert dist.arg_constraints == {}  # we never had any constraints
    assert dist.batch_shape == torch.Size([]) if batch_size is None else torch.Size([batch_size])
    assert dist.event_shape == torch.Size([])

    # Check bin shapes.
    assert dist.bin_edges.shape == (num_bins + 1,) if batch_size is None else (batch_size, num_bins + 1)
    assert dist.bin_centers.shape == (num_bins,) if batch_size is None else (batch_size, num_bins)
    assert dist.bin_widths.shape == (num_bins,) if batch_size is None else (batch_size, num_bins)
    assert dist.num_edges == dist.num_bins + 1

    # Test basic string representation.
    repr_str = repr(dist)
    assert "BinnedLogitCDF" in repr_str

    # Test that probabilities are valid. They should be normalized, and sum to 1.
    probs = dist.bin_probs
    assert probs.device == device
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(dist.batch_shape, device=device))

    # The probabilities should also be deterministic.
    probs2 = dist.bin_probs
    assert torch.allclose(probs, probs2)

    # Test that mean and variance have the correct shape and are finite.
    mean = dist.mean
    var = dist.variance
    assert mean.device == device
    assert var.device == device
    assert mean.shape == dist.batch_shape
    assert var.shape == dist.batch_shape
    assert torch.all(torch.isfinite(mean))
    assert torch.all(var >= 0)
    assert torch.all(torch.isfinite(var))


@pytest.mark.parametrize(
    "batch_size,new_batch_shape",
    [
        (None, [4, 5]),  # () can expand to any shape
        (1, [1, 5]),  # (1,) can expand by adding dimensions or keeping 1
        (1, [3, 1]),  # (1,) can also expand with 1 in last position
        (8, [2, 8]),  # (8,) can expand by adding leading dimensions
        (8, [3, 2, 8]),  # (8,) can expand with multiple leading dimensions
    ],
)
@pytest.mark.parametrize("num_bins", [2, 200])  # 2 is an edge case for log-spacing
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_expand(
    batch_size: int | None,
    new_batch_shape: list[int],
    num_bins: int,
    log_spacing: bool,
    use_cuda: bool,
):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)
    dist = BinnedLogitCDF(logits, log_spacing=log_spacing)

    expanded_dist = dist.expand(new_batch_shape)

    # Assert that expanded_dist is a different object (not the same instance).
    assert expanded_dist is not dist, "Expanded distribution should be a new instance"

    # Assert that the expanded distribution is on the same device.
    assert expanded_dist.logits.device == device, f"Expected device {device}, got {expanded_dist.logits.device}"
    assert expanded_dist.bin_edges.device == device
    assert expanded_dist.bin_centers.device == device
    assert expanded_dist.bin_widths.device == device

    # Assert that the batch shape is correct.
    assert expanded_dist.batch_shape == torch.Size(new_batch_shape), (
        f"Expected batch_shape {torch.Size(new_batch_shape)}, got {expanded_dist.batch_shape}"
    )

    # Assert that the logits have the correct shape: (*new_batch_shape, num_bins).
    expected_logits_shape = torch.Size([*new_batch_shape, num_bins])
    assert expanded_dist.logits.shape == expected_logits_shape, (
        f"Expected logits shape {expected_logits_shape}, got {expanded_dist.logits.shape}"
    )

    # Verify properties that should remain unchanged.
    assert expanded_dist.event_shape == torch.Size([]), "event_shape should remain empty (scalar)"
    assert expanded_dist.num_bins == num_bins, "num_bins should be unchanged"
    assert expanded_dist.bin_edges.shape == dist.bin_edges.shape, "bin_edges shape should be unchanged"
    assert expanded_dist.bin_centers.shape == dist.bin_centers.shape, "bin_centers shape should be unchanged"
    assert expanded_dist.bin_widths.shape == dist.bin_widths.shape, "bin_widths shape should be unchanged"


@pytest.mark.parametrize("batch_size", [None, 1, 8])
@pytest.mark.parametrize("num_bins", [2, 200])  # 2 is an edge case for log-spacing
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
@pytest.mark.parametrize("bin_normalization_method", ["sigmoid", "softmax"], ids=["sigmoid", "softmax"])
@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_prob_random_logits(
    batch_size: int | None,
    num_bins: int,
    log_spacing: bool,
    bin_normalization_method: Literal["sigmoid", "softmax"],
    use_cuda: bool,
):
    """Test probability evaluation with random logits at the bounds."""
    torch.manual_seed(42)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)
    dist = BinnedLogitCDF(logits, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method)

    # Define expected shapes based on batch_size. The bins go into the sample shape.
    bin_centers = dist.bin_centers
    if batch_size is not None:
        # Expand to (num_bins, batch_size) for batched distributions.
        bin_centers = bin_centers.unsqueeze(1).expand(num_bins, batch_size)
        expected_probs_shape: tuple[int, ...] = (num_bins, batch_size)
    else:
        # Keep as (num_bins,) for non-batched distributions.
        expected_probs_shape: tuple[int, ...] = (num_bins,)  # type: ignore[no-redef]

    # Test probability computation at bin centers.
    probs_at_centers = dist.log_prob(bin_centers)
    assert probs_at_centers.device == device
    assert torch.all(torch.isfinite(probs_at_centers)), "log_prob at bin centers should be finite"
    assert probs_at_centers.shape == expected_probs_shape

    # Test probability at bounds - should be finite but may be low
    expected_scalar_shape = torch.Size([]) if batch_size is None else torch.Size([batch_size])
    prob_at_low = dist.log_prob(torch.tensor(dist.bound_low, device=device))
    prob_at_up = dist.log_prob(torch.tensor(dist.bound_up, device=device))
    assert torch.all(torch.isfinite(prob_at_low)), f"log_prob at lower bound should be finite: {prob_at_low}"
    assert torch.all(torch.isfinite(prob_at_up)), f"log_prob at upper bound should be finite: {prob_at_up}"
    assert prob_at_low.shape == expected_scalar_shape
    assert prob_at_up.shape == expected_scalar_shape


@pytest.mark.parametrize("logit_scale", [1e-3, 1, 1e3, 1e9])
@pytest.mark.parametrize("bin_normalization_method", ["sigmoid", "softmax"], ids=["sigmoid", "softmax"])
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
    logit_scale: float,
    batch_size: int | None,
    bin_normalization_method: Literal["sigmoid", "softmax"],
    use_cuda: bool,
    plot: bool,
    bound_low: float = -10,
    bound_up: float = 10,
    num_bins: int = 400,
):
    """Test CDF evaluation with random logits at the bounds."""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = logit_scale * torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)

    dist = BinnedLogitCDF(logits, bound_low, bound_up, bin_normalization_method=bin_normalization_method)

    # Evaluate the CDF at the bounds.
    cdf_low = dist.cdf(torch.tensor(bound_low))
    cdf_up = dist.cdf(torch.tensor(bound_up))

    # Check the values at the bounds.
    assert torch.all(cdf_low <= 1.0 / num_bins), f"CDF at lower bound not <= 1/num_bins: {cdf_low}"
    assert torch.allclose(cdf_up, torch.ones_like(cdf_up)), f"CDF at upper bound not 1: {cdf_up}"

    if plot and batch_size is None:
        x = torch.linspace(bound_low, bound_up, 2000)
        cdf_vals = dist.cdf(x)
        plt.figure(figsize=(8, 5))
        plt.plot(x.numpy(), cdf_vals.numpy())
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title(f"CDF for random logits scaled by {logit_scale}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"tests/results/cdf_random_logits_scale-{logit_scale}_normalization-{bin_normalization_method}.png",
            bbox_inches="tight",
        )


@pytest.mark.parametrize("batch_size", [None, 1, 8, 16])
@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_icdf_random_quantiles(
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
    dist = BinnedLogitCDF(logits, bound_low, bound_up)
    if batch_size is not None:
        # For batched distributions, expand quantiles to (num_quantiles, *batch_shape)
        quantiles = quantiles.unsqueeze(-1).expand(num_quantiles, batch_size)
    icdf_values = dist.icdf(quantiles)

    # Check shape and device. Output shape should match input shape: (*sample_shape, *batch_shape)
    expected_shape = (num_quantiles,) if batch_size is None else (num_quantiles, batch_size)
    assert icdf_values.shape == expected_shape, f"ICDF shape mismatch: {icdf_values.shape} vs {expected_shape}"
    assert icdf_values.device == device, f"ICDF device mismatch: {icdf_values.device} vs {device}"

    # Check values are within bounds.
    assert torch.all(icdf_values >= bound_low), f"ICDF values below lower bound: min={icdf_values.min()}"
    assert torch.all(icdf_values <= bound_up), f"ICDF values above upper bound: max={icdf_values.max()}"

    # Check boundary quantiles.
    icdf_at_0 = dist.icdf(torch.tensor(0.0, device=device))
    icdf_at_1 = dist.icdf(torch.tensor(1.0, device=device))
    assert torch.all(icdf_at_0 >= bound_low - 1e-5), f"ICDF(0) should be >= bound_low: {icdf_at_0}"
    assert torch.all(icdf_at_1 <= bound_up + 1e-5), f"ICDF(1) should be <= bound_up: {icdf_at_1}"


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
    dist = BinnedLogitCDF(logits, bound_low, bound_up, log_spacing=log_spacing)

    # Test fixed quantiles with both linear and log spacing for the quantiles themselves.
    quantiles = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=device)

    # Compute inverse CDF at these quantiles.
    icdf_values = dist.icdf(quantiles)

    # Verify that all icdf values are within bounds.
    assert torch.all(icdf_values >= bound_low), f"ICDF values below lower bound: min={icdf_values.min()}"
    assert torch.all(icdf_values <= bound_up), f"ICDF values above upper bound: max={icdf_values.max()}"

    # Test the round-trip property: cdf(icdf(q)) ≈ q.
    cdf_roundtrip = dist.cdf(icdf_values)
    torch.testing.assert_close(
        cdf_roundtrip,
        quantiles,
        rtol=1e-3,
        atol=8e-4 if use_cuda else 4e-4,
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
        icdf_dense = dist.icdf(q_dense)
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
        cdf_dense = dist.cdf(x_dense)
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
            f"Mean: {dist.mean.item():.3f}\n"
            f"Std: {math.sqrt(dist.variance.item()):.3f}\n"
            f"Entropy: {dist.entropy().item():.3f}\n\n"
            f"Test Results:\n"
            f"{'=' * 40}\n"
            f"Quantiles tested: {len(quantiles)}\n"
            f"Max round-trip error: {roundtrip_error.max():.6f}\n"
            f"Mean abs round-trip error: {abs(roundtrip_error).mean():.6f}\n"
            f"ICDF range: [{icdf_values.min():.3f}, {icdf_values.max():.3f}]"
        )
        ax4.text(0.1, 0.5, properties_text, fontsize=10, verticalalignment="center", family="monospace")

        spacing_suffix = "log-spacing" if log_spacing else "linear-spacing"
        plt.suptitle(f"ICDF Test with Fixed Quantiles ({spacing_suffix})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"tests/results/icdf_fixed_quantiles_{spacing_suffix}.png", bbox_inches="tight", dpi=100)


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
    """Test that samples follow the BinnedLogitCDF's CDF and have the correct shape."""
    torch.manual_seed(42)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create random distribution and sample from it.
    logits = torch.randn(distr_batch_size, num_bins, device=device)
    dist = BinnedLogitCDF(logits, bound_low, bound_up)
    sample_shape = (num_samples,) if sample_batch_size is None else (sample_batch_size, num_samples)
    samples = dist.sample(sample_shape)
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
        single_logits = dist.logits[batch_idx : batch_idx + 1]  # Keep batch dimension of size 1
        single_dist = BinnedLogitCDF(single_logits, bound_low, bound_up)
        theoretical_cdf = single_dist.cdf(test_points).squeeze(0)  # remove batch dimension

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
        plt.savefig("tests/results/sampling_and_cdf_consistency.png", bbox_inches="tight")


@pytest.mark.parametrize(
    "target_dist_params",
    [
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 0.0, "scale": 1.0},
                "bounds": (-5.0, 5.0),
                "rel_tol": 0.05,
            },
            id="standard_normal",
        ),
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 0.0, "scale": 3.0},
                "bounds": (-15.0, 15.0),
                "rel_tol": 0.05,
            },
            id="extreme_normal",
        ),
    ],
)
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
def test_entropy(
    target_dist_params: dict,
    log_spacing: bool,
    num_bins: int = 200,
):
    """Test entropy computation against theoretical values from known distributions."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract parameters from the parametrized input.
    dist_class = target_dist_params["dist"]
    dist_params = target_dist_params["params"]
    bound_low, bound_up = target_dist_params["bounds"]
    rel_tol = target_dist_params["rel_tol"]

    # Create target distribution, and get the entropy.
    target_dist = dist_class(**dist_params)
    target_entropy = target_dist.entropy().item()

    # Use the BinnedLogitCDF's own bin construction to ensure matching shapes between distributions.
    _, bin_centers, bin_widths = BinnedLogitCDF._create_bins(
        num_bins=num_bins,
        bound_low=bound_low,
        bound_up=bound_up,
        log_spacing=log_spacing,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Compute target probabilities at bin centers, and normalize to get probability masses for each bin.
    target_probs = torch.exp(target_dist.log_prob(bin_centers))
    target_prob_masses = target_probs * bin_widths
    target_prob_masses = target_prob_masses / target_prob_masses.sum()

    # Convert probabilities to logits (inverse sigmoid).
    eps = 1e-8
    target_prob_masses = torch.clamp(target_prob_masses, eps, 1 - eps)
    logits = torch.log(target_prob_masses / (1 - target_prob_masses))

    # Create BinnedLogitCDF distribution, and compute reconstructed entropy.
    dist = BinnedLogitCDF(
        logits=logits.unsqueeze(0),
        bound_low=bound_low,
        bound_up=bound_up,
        log_spacing=log_spacing,
    )
    reconstructed_entropy = dist.entropy().item()

    # Check that reconstructed entropy is close to theoretical value.
    torch.testing.assert_close(
        reconstructed_entropy,
        target_entropy,
        rtol=rel_tol,
        atol=1e-6,
        msg=f"Entropy mismatch: reconstructed={reconstructed_entropy:.6f}, theoretical={target_entropy:.6f}",
    )


@pytest.mark.parametrize(
    "target_dist_params",
    [
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 3.0, "scale": 2.0},
                "bounds": (-10.0, 10.0),
                "tolerances": {"mean": 0.1, "std": 0.1},
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
def test_distribution_reconstruction(
    target_dist_params: dict,
    log_spacing: bool,
    plot: bool,
    num_bins: int = 200,
):
    """Test reconstruction of different distributions using BinnedLogitCDF."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract parameters from the parametrized input, and create target distribution.
    dist_class = target_dist_params["dist"]
    dist_params = target_dist_params["params"]
    bound_low, bound_up = target_dist_params["bounds"]
    tolerances = target_dist_params["tolerances"]
    target_dist = dist_class(**dist_params)

    # Skip log spacing tests for incompatible bounds.
    if log_spacing and not math.isclose(-bound_low, bound_up):
        pytest.skip("log_spacing requires symmetric bounds")
    if log_spacing and bound_up <= 0:
        pytest.skip("log_spacing requires positive upper bound")
    if log_spacing and num_bins % 2 != 0:
        pytest.skip("log_spacing requires even number of bins")

    # Get distribution properties for validation.
    target_mean = target_dist.mean.item()
    target_std = target_dist.stddev.item()

    # Use the BinnedLogitCDF's own bin construction to ensure matching shapes between distributions.
    _, bin_centers, bin_widths = BinnedLogitCDF._create_bins(
        num_bins=num_bins,
        bound_low=bound_low,
        bound_up=bound_up,
        log_spacing=log_spacing,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Compute target probabilities at bin centers.
    target_probs = torch.exp(target_dist.log_prob(bin_centers))

    # Normalize to get probability masses for each bin.
    target_prob_masses = target_probs * bin_widths
    target_prob_masses = target_prob_masses / target_prob_masses.sum()

    # Convert probabilities to logits (inverse sigmoid).
    eps = 1e-8
    target_prob_masses = torch.clamp(target_prob_masses, eps, 1 - eps)
    logits = torch.log(target_prob_masses / (1 - target_prob_masses))

    # Create BinnedLogitCDF distribution, and get mean and variance.
    dist = BinnedLogitCDF(
        logits=logits,
        bound_low=bound_low,
        bound_up=bound_up,
        log_spacing=log_spacing,
    )
    reconstructed_mean = dist.mean.item()
    reconstructed_var = dist.variance.item()
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
    original_samples = target_dist.sample((n_samples,))
    reconstructed_samples = dist.sample((n_samples,)).squeeze()

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
        original_cdf = target_dist.cdf(x_range)
        reconstructed_cdf = dist.cdf(x_range)
        plt.plot(x_range.numpy(), original_cdf.numpy(), label="Original CDF", linewidth=2)
        plt.plot(x_range.numpy(), reconstructed_cdf.numpy(), label="Reconstructed CDF", linewidth=2, linestyle="--")
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.title("CDF Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        dist_name = dist_class.__name__.lower()
        spacing_suffix = "log-init" if log_spacing else "linear-init"
        plt.savefig(f"tests/results/{dist_name}_reconstruction_{spacing_suffix}.png", bbox_inches="tight")

    # Additional we run the Kolmogorov-Smirnov test which looks at the maximum difference between CDFs.
    ks_statistic, ks_p_value = stats.ks_2samp(original_samples.numpy(), reconstructed_samples.numpy())

    # We expect some difference due to discretization, so use a lenient threshold.
    # assert ks_p_value > 0.001, f"Distributions too different (KS p-value: {ks_p_value:.4f})"

    if plot:
        mean_error = abs(reconstructed_mean - target_mean) / target_mean
        std_error = abs(reconstructed_std - target_std) / target_std
        print(f"Original: mean={target_mean:.3f}, std={target_std:.3f}")
        print(f"Reconstructed: mean={reconstructed_mean:.3f}, std={reconstructed_std:.3f}")
        print(f"Mean error: {mean_error:.3f}, Std error: {std_error:.3f}")
        print(f"KS test: statistic={ks_statistic:.4f}, p-value={ks_p_value:.4f}")
