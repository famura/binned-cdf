import torch

from .piecewise_constant_binned_cdf import PiecewiseConstantBinnedCDF


class PiecewiseLinearBinnedCDF(PiecewiseConstantBinnedCDF):
    """A continuous probability distribution parameterized by binned logits for the CDF.

    Unlike [PiecewiseConstantBinnedCDF][binned_cdf.piecewise_constant_cdf.PiecewiseConstantBinnedCDF], which evaluates the CDF as a
    step function over bin centers, this class implements a true piecewise-linear CDF, i.e., histogram PDF,
    interpolating smoothly between bin edges.
    """

    @property
    def variance(self) -> torch.Tensor:
        """Compute variance of the distribution, of shape (*batch_shape,).

        Note:
            Since the distribution is piecewise linear, the variance includes both the discrete variance from the
            bin probabilities and the intra-bin variance due to linear interpolation called Sheppard's correction,
            which assumes that probabilities are uniformly distributed within each bin.
        """
        discrete_var = super().variance
        intra_bin_var = torch.sum(self.bin_probs * (self.bin_widths**2) / 12.0, dim=-1)  # Sheppard's correction
        return discrete_var + intra_bin_var

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

        # Explicitly broadcast value to batch_shape if needed (e.g., scalar inputs with batched distributions).
        if len(self.batch_shape) > 0 and value.ndim < len(self.batch_shape):
            value = value.expand(self.batch_shape)

        # Use binary search to find how many bin centers are <= value.
        value = value.contiguous()

        # Find which bin the value falls into
        bin_indices_active = torch.searchsorted(self.bin_edges, value) - 1

        # Clamp to valid range [0, num_bins - 1].
        bin_indices_active = torch.clamp(bin_indices_active, 0, self.num_bins - 1)

        # Gather left edges, bin widths, and bin probabilities
        left_edges = self.bin_edges[bin_indices_active]
        bin_widths_selected = self.bin_widths[bin_indices_active]

        num_sample_dims = len(bin_indices_active.shape) - len(self.batch_shape)

        # Compute base CDF values at left edges (y0)
        cumsum_probs = torch.cumsum(self.bin_probs, dim=-1)  # shape: (*batch_shape, num_bins)
        cumsum_probs = torch.cat(
            [torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device), cumsum_probs],
            dim=-1,
        )  # shape: (*batch_shape, num_bins + 1)

        # Expand cumsum_probs to match sample dimensions and gather.
        cumsum_probs_for_gather = cumsum_probs.view((1,) * num_sample_dims + cumsum_probs.shape)
        cumsum_probs_for_gather = cumsum_probs_for_gather.expand(*bin_indices_active.shape, -1)
        base_cdf = torch.gather(cumsum_probs_for_gather, dim=-1, index=bin_indices_active.unsqueeze(-1)).squeeze(-1)

        # Gather probability masses for the active bins
        bin_probs_for_gather = self.bin_probs.view((1,) * num_sample_dims + self.bin_probs.shape)
        bin_probs_for_gather = bin_probs_for_gather.expand(*bin_indices_active.shape, -1)
        bin_probs_selected = torch.gather(bin_probs_for_gather, dim=-1, index=bin_indices_active.unsqueeze(-1)).squeeze(
            -1
        )

        # Linear interpolation fraction
        alpha = (value - left_edges) / bin_widths_selected
        alpha = torch.clamp(alpha, 0.0, 1.0)
        cdf_values = base_cdf + alpha * bin_probs_selected

        # Enforce hard boundaries just in case
        cdf_values = torch.where(value < self.bound_low, torch.zeros_like(cdf_values), cdf_values)
        cdf_values = torch.where(value >= self.bound_up, torch.ones_like(cdf_values), cdf_values)

        return cdf_values
