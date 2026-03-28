import math
from typing import Literal

import numpy as np
import pytest
import torch

from binned_cdf import PiecewiseConstantBinnedCDF
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
    """Test basic properties of the PiecewiseConstantBinnedCDF."""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logits = torch.randn((num_bins,)) if batch_size is None else torch.randn(batch_size, num_bins)
    logits = logits.to(device)

    if log_spacing and not math.isclose(-bound_low, bound_up):
        with pytest.raises(ValueError, match="log_spacing requires symmetric bounds"):
            PiecewiseConstantBinnedCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    if log_spacing and bound_up <= 0:
        with pytest.raises(ValueError, match="log_spacing requires positive upper bound"):
            PiecewiseConstantBinnedCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    if log_spacing and num_bins % 2 != 0:
        with pytest.raises(ValueError, match="log_spacing requires even number of bins"):
            PiecewiseConstantBinnedCDF(
                logits, bound_low, bound_up, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
            )
        return
    dist = PiecewiseConstantBinnedCDF(
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
    assert "PiecewiseConstantBinnedCDF" in repr_str

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
    dist = PiecewiseConstantBinnedCDF(logits, log_spacing=log_spacing)

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
    dist = PiecewiseConstantBinnedCDF(
        logits, log_spacing=log_spacing, bin_normalization_method=bin_normalization_method
    )

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


@pytest.mark.parametrize(
    "target_dist_params",
    [
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 0.0, "scale": 1.0},
                "bounds": (-5.0, 5.0),
            },
            id="standard_normal",
        ),
        pytest.param(
            {
                "dist": torch.distributions.Normal,
                "params": {"loc": 0.0, "scale": 3.0},
                "bounds": (-15.0, 15.0),
            },
            id="extreme_normal",
        ),
    ],
)
@pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
def test_shannon_entropy(
    target_dist_params: dict,
    log_spacing: bool,
    num_bins: int = 200,
):
    """Test Shannon entropy computation against theoretical values from known distributions."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract parameters from the parametrized input.
    dist_class = target_dist_params["dist"]
    dist_params = target_dist_params["params"]
    bound_low, bound_up = target_dist_params["bounds"]

    # Create target distribution.
    target_dist = dist_class(**dist_params)

    # Use the PiecewiseConstantBinnedCDF's own bin construction to ensure matching shapes between distributions.
    _, bin_centers, bin_widths = PiecewiseConstantBinnedCDF._create_bins(
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

    # Compute theoretical Shannon entropy from the target distribution's probability masses.
    target_entropy = -torch.sum(target_prob_masses * torch.log(target_prob_masses + 1e-8)).item()

    # Convert probabilities to logits (inverse sigmoid).
    logits = torch.log(target_prob_masses)

    # Create PiecewiseConstantBinnedCDF distribution, and compute reconstructed entropy.
    dist = PiecewiseConstantBinnedCDF(
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
        rtol=5e-3,
        atol=1e-6,
        msg=f"Shannon Entropy mismatch: reconstructed={reconstructed_entropy:.6f}, theoretical={target_entropy:.6f}",
    )


# @pytest.mark.parametrize(
#     "target_dist_params",
#     [
#         pytest.param(
#             {
#                 "dist": torch.distributions.Normal,
#                 "params": {"loc": 0.0, "scale": 1.0},
#                 "bounds": (-5.0, 5.0),
#                 "rel_tol": 0.05,
#             },
#             id="standard_normal",
#         ),
#         pytest.param(
#             {
#                 "dist": torch.distributions.Normal,
#                 "params": {"loc": 0.0, "scale": 3.0},
#                 "bounds": (-15.0, 15.0),
#                 "rel_tol": 0.05,
#             },
#             id="extreme_normal",
#         ),
#     ],
# )
# @pytest.mark.parametrize("log_spacing", [False, True], ids=["linear_spacing", "log_spacing"])
# def test_differential_entropy(
#     target_dist_params: dict,
#     log_spacing: bool,
#     num_bins: int = 200,
# ):
#     """Test differential entropy computation against theoretical values from known distributions."""
#     torch.manual_seed(42)
#     np.random.seed(42)

#     # Extract parameters from the parametrized input.
#     dist_class = target_dist_params["dist"]
#     dist_params = target_dist_params["params"]
#     bound_low, bound_up = target_dist_params["bounds"]
#     rel_tol = target_dist_params["rel_tol"]

#     # Create target distribution, and get the entropy.
#     target_dist = dist_class(**dist_params)
#     target_entropy = target_dist.entropy().item()

#     # Use the PiecewiseConstantBinnedCDF's own bin construction to ensure matching shapes between distributions.
#     _, bin_centers, bin_widths = PiecewiseConstantBinnedCDF._create_bins(
#         num_bins=num_bins,
#         bound_low=bound_low,
#         bound_up=bound_up,
#         log_spacing=log_spacing,
#         device=torch.device("cpu"),
#         dtype=torch.float32,
#     )

#     # Compute target probabilities at bin centers, and normalize to get probability masses for each bin.
#     target_probs = torch.exp(target_dist.log_prob(bin_centers))
#     target_prob_masses = target_probs * bin_widths
#     target_prob_masses = target_prob_masses / target_prob_masses.sum()

#     # Convert probabilities to logits (inverse sigmoid).
#     eps = 1e-8
#     target_prob_masses = torch.clamp(target_prob_masses, eps, 1 - eps)
#     logits = torch.log(target_prob_masses / (1 - target_prob_masses))

#     # Create PiecewiseConstantBinnedCDF distribution, and compute reconstructed entropy.
#     dist = PiecewiseConstantBinnedCDF(
#         logits=logits.unsqueeze(0),
#         bound_low=bound_low,
#         bound_up=bound_up,
#         log_spacing=log_spacing,
#     )
#     reconstructed_entropy = dist.entropy().item()

#     # Check that reconstructed entropy is close to theoretical value.
#     torch.testing.assert_close(
#         reconstructed_entropy,
#         target_entropy,
#         rtol=rel_tol,
#         atol=1e-6,
#         msg=f"Entropy mismatch: reconstructed={reconstructed_entropy:.6f}, theoretical={target_entropy:.6f}",
#     )
