import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints

_size = torch.Size()


class BinnedLogitCDF(Distribution):
    """A histogram-based probability distribution parameterized by a bins for the CDF.

    Each bin contributes a step function to the CDF when active.
    The activation of each bin is determined by applying a sigmoid to the corresponding logit.
    The distribution is defined over the interval [bound_low, bound_up] with either linear or logarithmic bin spacing.

    Note:
        This distribution is differentiable with respect to the logits, i.e., the arguments of `__init__`, but
        not through the inputs of the `prob` or `cfg` method.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        bound_low: float = -1e3,
        bound_up: float = 1e3,
        log_spacing: bool = False,
        bin_normalization_method: Literal["sigmoid", "softmax"] = "sigmoid",
        validate_args: bool | None = None,
    ) -> None:
        """Initializer.

        Args:
            logits: Raw logits for bin probabilities (before sigmoid), of shape (*batch_shape, num_bins)
            bound_low: Lower bound of the distribution support, needs to be finite.
            bound_up: Upper bound of the distribution support, needs to be finite.
            log_spacing: Whether logarithmic (base = 2) spacing for the bins or linear spacing should be used.
            bin_normalization_method: How to normalize bin probabilities. Either "sigmoid" or "softmax". With "sigmoid",
                each bin is independently activated, while with "softmax", the bins activations influence each other.
            validate_args: Whether to validate arguments. Carried over to keep the interface with the base class.
        """
        self.logits = logits
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.bin_normalization_method = bin_normalization_method
        self.log_spacing = log_spacing

        # Create bin structure (same for all batch dimensions).
        self.bin_edges, self.bin_centers, self.bin_widths = self._create_bins(
            num_bins=logits.shape[-1],
            bound_low=bound_low,
            bound_up=bound_up,
            log_spacing=log_spacing,
            device=logits.device,
            dtype=logits.dtype,
        )

        super().__init__(batch_shape=logits.shape[:-1], event_shape=torch.Size([]), validate_args=validate_args)

    @classmethod
    def _create_bins(
        cls,
        num_bins: int,
        bound_low: float,
        bound_up: float,
        log_spacing: bool,
        device: torch.device,
        dtype: torch.dtype,
        log_min_positive_edge: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create bin edges with symmetric log spacing around zero.

        Args:
            num_bins: Number of bins to create.
            bound_low: Lower bound of the distribution support.
            bound_up: Upper bound of the distribution support.
            log_spacing: Whether to use logarithmic spacing.
            device: Device for the tensors.
            dtype: Data type for the tensors.
            log_min_positive_edge: Minimum positive edge when using log spacing. The log2-value of this argument
                will be passed to torch.logspace. Too small values, approx below 1e-9, will result in poor bin spacing.

        Returns:
            Tuple of (bin_edges, bin_centers, bin_widths).

        Layout:
            - 1 edge at 0
            - num_bins//2 - 1 edges from 0 to bound_up (log spaced)
            - num_bins//2 - 1 edges from 0 to -bound_low (log spaced, mirrored)
            - 2 boundary edges at ±bounds

        Total: num_bins + 1 edges creating num_bins bins
        """
        if log_spacing:
            if not math.isclose(-bound_low, bound_up):
                raise ValueError("log_spacing requires symmetric bounds: -bound_low == bound_up")
            if bound_up <= 0:
                raise ValueError("log_spacing requires bound_up > 0")
            if num_bins % 2 != 0:
                raise ValueError("log_spacing requires even number of bins")

            half_bins = num_bins // 2

            # Create positive side: 0, internal edges, bound_up.
            if half_bins == 1:
                # Special case where we only use the boundary edges.
                positive_edges = torch.tensor([bound_up])
            else:
                # Create half_bins - 1 internal edges between 0 and bound_up.
                internal_positive = torch.logspace(
                    start=math.log2(log_min_positive_edge),
                    end=math.log2(bound_up),
                    steps=half_bins,
                    base=2,
                )
                positive_edges = torch.cat([internal_positive[:-1], torch.tensor([bound_up])])

            # Mirror for the negative side (excluding 0).
            negative_edges = -positive_edges.flip(0)

            # Combine to [negative_boundary, negative_internal, 0, positive_internal, positive_boundary].
            bin_edges = torch.cat([negative_edges, torch.tensor([0.0]), positive_edges])

        else:
            # Linear spacing.
            bin_edges = torch.linspace(start=bound_low, end=bound_up, steps=num_bins + 1)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Move to specified device and dtype.
        bin_edges = bin_edges.to(device=device, dtype=dtype)
        bin_centers = bin_centers.to(device=device, dtype=dtype)
        bin_widths = bin_widths.to(device=device, dtype=dtype)

        return bin_edges, bin_centers, bin_widths

    @property
    def num_bins(self) -> int:
        """Number of bins making up the BinnedLogitCDF."""
        return self.logits.shape[-1]

    @property
    def num_edges(self) -> int:
        """Number of bins edges of the BinnedLogitCDF."""
        return self.bin_edges.shape[0]

    @property
    def bin_probs(self) -> torch.Tensor:
        """Get normalized probabilities for each bin, of shape (*batch_shape, num_bins)."""
        if self.bin_normalization_method == "sigmoid":
            raw_probs = torch.sigmoid(self.logits)  # shape: (*batch_shape, num_bins)
            bin_probs = raw_probs / raw_probs.sum(dim=-1, keepdim=True)
        else:
            bin_probs = torch.softmax(self.logits, dim=-1)  # shape: (*batch_shape, num_bins)
        return bin_probs

    @property
    def mean(self) -> torch.Tensor:
        """Compute mean of the distribution, i.e., the weighted average of bin centers, of shape (*batch_shape,)."""
        weighted_centers = self.bin_probs * self.bin_centers  # shape: (*batch_shape, num_bins)
        return torch.sum(weighted_centers, dim=-1)

    @property
    def variance(self) -> torch.Tensor:
        """Compute variance of the distribution, of shape (*batch_shape,)."""
        # E[X^2] = weighted squared bin centers.
        weighted_centers_sq = self.bin_probs * (self.bin_centers**2)  # shape: (*batch_shape, num_bins)
        second_moment = torch.sum(weighted_centers_sq, dim=-1)  # shape: (*batch_shape,)

        # Var = E[X^2] - E[X]^2
        return second_moment - self.mean**2

    @property
    def support(self) -> constraints.Constraint:
        """Support of this distribution. Needs to be limitited to keep the number of bins manageable."""
        return constraints.interval(self.bound_low, self.bound_up)

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Constraints that should be satisfied by each argument of this distribution. None for this class."""
        return {}

    def expand(
        self, batch_shape: torch.Size | list[int] | tuple[int, ...], _instance: Distribution | None = None
    ) -> "BinnedLogitCDF":
        """Expand distribution to new batch shape. This creates a new instance."""
        expanded_logits = self.logits.expand((*torch.Size(batch_shape), self.num_bins))
        return BinnedLogitCDF(
            logits=expanded_logits,
            bound_low=self.bound_low,
            bound_up=self.bound_up,
            log_spacing=self.log_spacing,
            validate_args=self._validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability density at given values.

        Args:
            value: Values at which to compute the log PDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            Log PDF values corresponding to the input values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        return torch.log(self.prob(value) + 1e-8)  # small epsilon for stability

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute probability density at given values.

        Args:
            value: Values at which to compute the PDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            PDF values corresponding to the input values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        if self._validate_args:
            self._validate_sample(value)

        value = value.to(dtype=self.logits.dtype, device=self.logits.device)

        # Determine number of sample dimensions (dimensions before batch_shape).
        num_sample_dims = len(value.shape) - len(self.batch_shape)

        # Prepend singleton dimensions for sample_shape to bin_edges, bin_widths, and probs.
        # For all of them, the resulting shape will be: (*sample_shape, *batch_shape, num_bins)
        bin_edges_left = self.bin_edges[..., :-1]  # shape: (*batch_shape, num_bins)
        bin_edges_right = self.bin_edges[..., 1:]  # shape: (*batch_shape, num_bins)
        bin_edges_left = bin_edges_left.view((1,) * num_sample_dims + bin_edges_left.shape)
        bin_edges_right = bin_edges_right.view((1,) * num_sample_dims + bin_edges_right.shape)
        bin_widths = self.bin_widths.view((1,) * num_sample_dims + self.bin_widths.shape)
        probs = self.bin_probs.view((1,) * num_sample_dims + self.bin_probs.shape)

        # Add bin dimension to value for broadcasting.
        value_expanded = value.unsqueeze(-1)  # shape: (*sample_shape, *batch_shape, 1)

        # Check which bin each value falls into. Result shape: (*sample_shape, *batch_shape, num_bins).
        in_bin = ((value_expanded >= bin_edges_left) & (value_expanded < bin_edges_right)).to(self.logits.dtype)

        # Handle right edge case (include bound_up in last bin).
        at_right_edge = torch.isclose(
            value_expanded, torch.tensor(self.bound_up, dtype=self.logits.dtype, device=self.logits.device)
        )
        in_bin[..., -1] = torch.max(in_bin[..., -1], at_right_edge[..., -1])

        # PDF = (probability mass / bin width) for the containing bin.
        pdf_per_bin = probs / bin_widths  # shape: (*sample_shape, *batch_shape, num_bins)

        # Sum over bins is the same as selecting the bin, as there is only one bin active per value.
        return torch.sum(in_bin * pdf_per_bin, dim=-1)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute cumulative distribution function at given values.

        Args:
            value: Values at which to compute the CDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            CDF values in [0, 1] corresponding to the input values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        if self._validate_args:
            self._validate_sample(value)

        value = value.to(dtype=self.logits.dtype, device=self.logits.device)

        # Determine number of sample dimensions (dimensions before batch_shape).
        num_sample_dims = len(value.shape) - len(self.batch_shape)

        # Prepend singleton dimensions for sample_shape to bin_centers.
        # bin_centers: (*batch_shape, num_bins) -> (*sample_shape, *batch_shape, num_bins)
        bin_centers_expanded = self.bin_centers.view((1,) * num_sample_dims + self.bin_centers.shape)

        # Prepend singleton dimensions for sample_shape to probs.
        # probs: (*batch_shape, num_bins) -> (*sample_shape, *batch_shape, num_bins)
        probs_expanded = self.bin_probs.view((1,) * num_sample_dims + self.bin_probs.shape)

        # Add the bin dimension to the input which is used for comparing with the bin centers.
        value_expanded = value.unsqueeze(-1)  # shape: (*sample_shape, *batch_shape, 1)

        # Mask for bins with centers <= value.
        mask = bin_centers_expanded <= value_expanded  # shape: (*sample_shape, *batch_shape, num_bins)

        # Sum the bins for this value by their weighted "activation"=probability.
        return torch.sum(mask * probs_expanded, dim=-1)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the inverse CDF, i.e., the quantile function, at the given values.

        Args:
            value: Values in [0, 1] at which to compute the inverse CDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            Quantiles in [bound_low, bound_up] corresponding to the input CDF values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        if self._validate_args and not (value >= 0).all() and (value <= 1).all():
            raise ValueError("icdf input must be in [0, 1]")

        value = value.to(dtype=self.logits.dtype, device=self.logits.device)

        # Compute CDF at bin edges. prepend zeros to the cumsum of probabilities as this is always the first edge.
        cdf_edges = torch.cat(
            [
                torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device),
                torch.cumsum(self.bin_probs, dim=-1),  # shape: (*batch_shape, num_bins)
            ],
            dim=-1,
        )  # shape: (*batch_shape, num_bins + 1)

        # Determine number of sample dimensions (dimensions before batch_shape).
        num_sample_dims = len(value.shape) - len(self.batch_shape)

        # Prepend singleton dimensions for sample_shape to cdf_edges.
        # cdf_edges: (*batch_shape, num_bins + 1) -> (*sample_shape, *batch_shape, num_bins + 1)
        cdf_edges = cdf_edges.view((1,) * num_sample_dims + cdf_edges.shape)

        # Prepend singleton dimensions for  both sample_shape and batch_shape.
        # bin_edges: (num_bins + 1,) -> (*sample_shape, *batch_shape, num_bins + 1)
        bin_edges_expanded = self.bin_edges.view(
            (1,) * (num_sample_dims + len(self.batch_shape)) + self.bin_edges.shape
        )

        # Add bin dimension to value for comparison.
        value_expanded = value.unsqueeze(-1)

        # Find bins containing the value: left_cdf <= value < right_cdf.
        bin_mask = (cdf_edges[..., :-1] <= value_expanded) & (value_expanded < cdf_edges[..., 1:])
        bin_mask = bin_mask.to(self.logits.dtype)

        # Handle edge case where value ≈ 1.0 (use isclose with dtype-appropriate defaults).
        value_is_one = torch.isclose(value_expanded, torch.ones_like(value_expanded))
        bin_mask[..., -1] = torch.max(bin_mask[..., -1], value_is_one[..., 0])  # last bin could be selected already

        # Selected the correct bin edges using the mask. Summing is essentially selecting here.
        # Summing fast and differentiable.
        cfd_value_bin_starts = torch.sum(bin_mask * cdf_edges[..., :-1], dim=-1)
        cdf_value_bin_ends = torch.sum(bin_mask * cdf_edges[..., 1:], dim=-1)
        bin_left_edges = torch.sum(bin_mask * bin_edges_expanded[..., :-1], dim=-1)
        bin_right_edges = torch.sum(bin_mask * bin_edges_expanded[..., 1:], dim=-1)

        # Avoid division by zero.
        bin_width = cdf_value_bin_ends - cfd_value_bin_starts
        safe_bin_width = torch.where(bin_width > 1e-8, bin_width, torch.ones_like(bin_width))

        # Linear interpolation within the bin.
        alpha = (value - cfd_value_bin_starts) / safe_bin_width
        quantiles = bin_left_edges + alpha * (bin_right_edges - bin_left_edges)

        return quantiles

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size | list[int] | tuple[int, ...] = _size) -> torch.Tensor:
        """Sample from the distribution by passing uniformly random draws from [0, 1] thought the inverse CDF.

        Args:
            sample_shape: Shape of the samples to draw.

        Returns:
            Samples of shape (sample_shape + batch_shape), where batch_shape is the batch shape of the distribution.
        """
        shape = torch.Size(sample_shape) + self.batch_shape
        uniform_samples = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        return self.icdf(uniform_samples)

    def entropy(self) -> torch.Tensor:
        r"""Compute differential entropy of the distribution.

        Entropy H(X) = -\sum_{x \in \mathcal{X}} p(x) \log( p(x) )

        Note:
            Here, we are doing an approximation by treating each bin as a uniform distribution over its width.
        """
        bin_probs = self.bin_probs

        # Get the PDF values at bin centers.
        pdf_values = bin_probs / self.bin_widths  # shape: (*batch_shape, num_bins)

        # Entropy ≈ -∑ p_i * log(pdf_i) * bin_width_i.
        log_pdf = torch.log(pdf_values + 1e-8)  # small epsilon for stability
        entropy_per_bin = -bin_probs * log_pdf

        # Sum over bins to get total entropy.
        return torch.sum(entropy_per_bin, dim=-1)

    def __repr__(self) -> str:
        """String representation of the distribution."""
        return (
            f"{self.__class__.__name__}(logits_shape: {self.logits.shape}, bound_low: {self.bound_low}, "
            f"bound_up: {self.bound_up}, log_spacing: {self.log_spacing})"
        )
