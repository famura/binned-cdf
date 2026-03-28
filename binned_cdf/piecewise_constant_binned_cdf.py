import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints
from torch.nn.functional import log_softmax, logsigmoid

_size = torch.Size()


class PiecewiseConstantBinnedCDF(Distribution):
    """A discrete probability distribution parameterized by binned logits for the CDF.

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
        """Number of bins making up the PiecewiseConstantBinnedCDF."""
        return self.logits.shape[-1]

    @property
    def num_edges(self) -> int:
        """Number of bins edges of the PiecewiseConstantBinnedCDF."""
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
    ) -> "PiecewiseConstantBinnedCDF":
        """Expand distribution to new batch shape. This creates a new instance."""
        expanded_logits = self.logits.expand((*torch.Size(batch_shape), self.num_bins))
        return self.__class__(
            logits=expanded_logits,
            bound_low=self.bound_low,
            bound_up=self.bound_up,
            log_spacing=self.log_spacing,
            validate_args=self._validate_args,
        )

    def _prepare_input(self, value: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Prepare the input tensor for `log_prob`, `prob`, `cdf` and `icdf` computations.

        This method handles device/dtype transfer, batch dimension alignment, and broadcasting.

        Args:
            value: Input tensor to prepare. Expected shape: `(*sample_shape, *batch_shape)` or broadcastable to it.
                For example, if `batch_shape` is `(B1, B2)` and `value` is `(S1, S2)`, it will be broadcast to
                `(S1, S2, B1, B2)`. If `value` is `(B1, B2)` (no sample dims), it remains `(B1, B2)`.

        Returns:
            A tuple containing:
            - Prepared `value` tensor, of shape: `(*sample_shape, *batch_shape)`.
            - `num_sample_dims`: The number of sample dimensions in the prepared `value` tensor.
        """
        value = value.to(dtype=self.logits.dtype, device=self.logits.device)

        # This ensures the batch dimension is the last dimension.
        if len(self.batch_shape) > 0:  # noqa: SIM102
            # Check if the rightmost dimensions of value match batch_shape.
            # If they don't, we assume value is missing the batch dimensions.
            if value.shape[-len(self.batch_shape) :] != self.batch_shape:
                value = value.unsqueeze(-1)

        num_sample_dims = max(0, value.ndim - len(self.batch_shape))
        target_shape = torch.Size(value.shape[:num_sample_dims]) + self.batch_shape
        value = value.expand(target_shape)
        value = value.contiguous()  # for searchsorted later

        return value, num_sample_dims

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability density at given values.

        Args:
            value: Values at which to compute the log PDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            Log PDF values corresponding to the input values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        if self._validate_args:
            self._validate_sample(value)

        value, num_sample_dims = self._prepare_input(value)

        # Use binary search to find which bin each value belongs to. The torch.searchsorted function returns the
        # index where value would be inserted to maintain sorted order.
        # Since bins are defined as [edge[i], edge[i+1]), we subtract 1 to get the bin index.
        bin_indices = torch.searchsorted(self.bin_edges, value, right=True) - 1  # shape: (*sample_shape, *batch_shape)

        # Clamp to valid range [0, num_bins - 1] to handle edge cases:
        # - values below bound_low would give bin_idx = -1
        # - values at bound_up would give bin_idx = num_bins
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Calculate log_probs directly for stability
        if self.bin_normalization_method == "sigmoid":
            # Normalized logsigmoid: log(sigmoid(x) / sum(sigmoid(x)))
            log_raw = logsigmoid(self.logits)
            log_normalization = torch.logsumexp(log_raw, dim=-1, keepdim=True)
            log_bin_probs = log_raw - log_normalization
        else:
            log_bin_probs = log_softmax(self.logits, dim=-1)

        log_bin_probs_expanded = log_bin_probs.view((1,) * num_sample_dims + log_bin_probs.shape)
        log_bin_probs_expanded = log_bin_probs_expanded.expand(*value.shape, -1)
        # Use gather with automatic broadcasting. unsqueeze(-1) provides the index dimension,
        # and squeeze(-1) removes it from the result.
        log_prob = torch.gather(log_bin_probs_expanded, dim=-1, index=bin_indices.unsqueeze(-1)).squeeze(-1)

        return log_prob

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

        value, num_sample_dims = self._prepare_input(value)

        # Use binary search to find which bin each value belongs to. The torch.searchsorted function returns the
        # index where value would be inserted to maintain sorted order.
        # Since bins are defined as [edge[i], edge[i+1]), we subtract 1 to get the bin index.
        bin_indices = torch.searchsorted(self.bin_edges, value, right=True) - 1  # shape: (*sample_shape, *batch_shape)

        # Clamp to valid range [0, num_bins - 1] to handle edge cases:
        # - values below bound_low would give bin_idx = -1
        # - values at bound_up would give bin_idx = num_bins
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        bin_probs_expanded = self.bin_probs.view((1,) * num_sample_dims + self.bin_probs.shape)
        bin_probs_expanded = bin_probs_expanded.expand(*value.shape, -1)
        # Use gather with automatic broadcasting. unsqueeze(-1) provides the index dimension,
        # and squeeze(-1) removes it from the result.
        bin_probs_selected = torch.gather(bin_probs_expanded, dim=-1, index=bin_indices.unsqueeze(-1)).squeeze(-1)

        return bin_probs_selected

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

        value, num_sample_dims = self._prepare_input(value)

        # Use binary search to find how many bin centers are <= value.
        # torch.searchsorted with right=True gives us the number of elements <= value.
        # If value < first center, returns 0 -> gets 0.0 from cumsum_probs
        # If value >= last center, returns num_bins -> gets 1.0 from cumsum_probs
        bin_indices = torch.searchsorted(self.bin_centers, value, right=True)

        # Clamp to valid range [0, num_bins].
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins)  # shape: (*sample_shape, *batch_shape)

        # Compute the cumulative sum of bin probabilities.
        # Prepend 0 for the case where no bins are active.
        cumsum_probs = torch.cumsum(self.bin_probs, dim=-1)  # shape: (*batch_shape, num_bins)
        zero_prefix = torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device)
        cumsum_probs = torch.cat([zero_prefix, cumsum_probs], dim=-1)  # shape: (*batch_shape, num_bins + 1)

        cumsum_probs_expanded = cumsum_probs.view((1,) * num_sample_dims + cumsum_probs.shape)
        cumsum_probs_expanded = cumsum_probs_expanded.expand(*value.shape, -1)
        # Use gather with automatic broadcasting. unsqueeze(-1) provides the index dimension,
        # and squeeze(-1) removes it from the result.
        cdf_values = torch.gather(cumsum_probs_expanded, dim=-1, index=bin_indices.unsqueeze(-1)).squeeze(-1)

        return cdf_values

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

        value, num_sample_dims = self._prepare_input(value)

        # Compute CDF at bin edges. Prepend zeros to the cumsum of probabilities as this is always the first edge.
        cdf_edges = torch.cat(
            [
                torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device),
                torch.cumsum(self.bin_probs, dim=-1),  # shape: (*batch_shape, num_bins)
            ],
            dim=-1,
        )  # [0, p1, p1+p2, ..., 1.0], shape: (*batch_shape, num_bins + 1)

        # Prepend singleton dimensions for sample_shape to cdf_edges and expand to match value.
        cdf_edges_expanded = cdf_edges.view((1,) * num_sample_dims + cdf_edges.shape)
        cdf_edges_expanded = cdf_edges_expanded.expand(*value.shape, -1)
        cdf_edges_expanded = cdf_edges_expanded.contiguous()

        # Find bins containing the value using binary search instead of a mask.
        # right=True ensures that value=1.0 is handled correctly by selecting the last bin.
        bin_indices = torch.searchsorted(cdf_edges_expanded, value.unsqueeze(-1), right=True) - 1

        # Clamp to valid range [0, num_bins - 1] to handle edge cases:
        # - values below bound_low would give bin_idx = -1
        # - values at bound_up would give bin_idx = num_bins
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Gather efficiently. We only add singleton dimensions for the sample dimensions, as self.bin_centers
        # already contains the batch dimensions..
        bin_centers_expanded = self.bin_centers.view((1,) * num_sample_dims + self.bin_centers.shape)
        bin_centers_expanded = bin_centers_expanded.expand(*value.shape, -1)
        # Use gather with automatic broadcasting.
        quantiles = torch.gather(bin_centers_expanded, dim=-1, index=bin_indices).squeeze(-1)

        # Return the discrete quantiles.
        return quantiles  # shape: (*sample_shape, *batch_shape)

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
        r"""Compute Shannon entropy of the discrete distribution.

        $$H(X) = -\sum_{i=1}^{n} p_i \log p_i$$
        where $p_i$ is the probability mass of bin $i$.
        """
        bin_probs = self.bin_probs

        # Compute entropy per bin and sum over bins. Add small epsilon for numerical stability in log.
        entropy_per_bin = bin_probs * torch.log(bin_probs + 1e-8)  # shape: (*batch_shape, num_bins)

        # Sum over bins to get total entropy.
        return -torch.sum(entropy_per_bin, dim=-1)

    def __repr__(self) -> str:
        """String representation of the distribution."""
        return (
            f"{self.__class__.__name__}(logits_shape: {self.logits.shape}, bound_low: {self.bound_low}, "
            f"bound_up: {self.bound_up}, log_spacing: {self.log_spacing})"
        )
