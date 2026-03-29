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

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability density at given values.

        Args:
            value: Values at which to compute the log PDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            Log PDF values corresponding to the input values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        # Compute the log of the probability mass for the bin the value falls into.
        log_mass = super().log_prob(value)

        # We need to gather the width of the bin the value falls into
        value_prep, num_sample_dims = self._prepare_input(value)
        bin_indices = self._get_bin_indices(value_prep, bin_edges=self.bin_edges)
        widths = self._gather_from_bins(self.bin_widths, bin_indices, num_sample_dims, value_prep.shape)

        # Log density = log(mass / width) = log_mass - log_width.
        log_prob = log_mass - torch.log(widths + 2 * widths.dtype.eps)

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

        value_prep, num_sample_dims = self._prepare_input(value)

        bin_indices = self._get_bin_indices(value_prep, bin_edges=self.bin_edges)

        # Gather normalized mass and bin width.
        masses = self._gather_from_bins(self.bin_probs, bin_indices, num_sample_dims, value_prep.shape)
        widths = self._gather_from_bins(self.bin_widths, bin_indices, num_sample_dims, value_prep.shape)

        # Density = p(bin_i) / width_i.
        prob = masses / (widths + 2 * widths.dtype.eps)

        return prob

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

        value_prep, num_sample_dims = self._prepare_input(value)

        # Find the bin in probability space.
        bin_indices = self._get_bin_indices(value_prep, bin_edges=self.bin_edges)

        # Gather the interpolation parameters.
        left_edges = self._gather_from_bins(self.bin_edges[:-1], bin_indices, num_sample_dims, value_prep.shape)
        widths = self._gather_from_bins(self.bin_widths, bin_indices, num_sample_dims, value_prep.shape)
        masses = self._gather_from_bins(self.bin_probs, bin_indices, num_sample_dims, value_prep.shape)

        # Get base CDF at the left edge of the bin.
        cumsum_probs = torch.cumsum(self.bin_probs, dim=-1)
        zero_prefix = torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device)
        base_cdf_table = torch.cat([zero_prefix, cumsum_probs], dim=-1)
        base_cdf = self._gather_from_bins(base_cdf_table, bin_indices, num_sample_dims, value_prep.shape)

        # Interpolate: cdf = base_cdf + (value - left_edge) * (mass / width)
        alpha = (value_prep - left_edges) / (widths + 2 * widths.dtype.eps)
        alpha = torch.clamp(alpha, 0.0, 1.0)  # prevent extrapolation
        cdf_vals = base_cdf + alpha * masses
        return cdf_vals

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the inverse CDF, i.e., the quantile function, at the given values.

        Args:
            value: Values in [0, 1] at which to compute the inverse CDF.
                Expected shape: (*sample_shape, *batch_shape) or broadcastable to it.

        Returns:
            Quantiles in [bound_low, bound_up] corresponding to the input CDF values.
            Output shape: same as `value` shape after broadcasting, i.e., (*sample_shape, *batch_shape).
        """
        if self._validate_args:
            raise ValueError("icdf input must be in [0, 1]")

        value_prep, num_sample_dims = self._prepare_input(value)

        # Get the CDF edges (y-coordinates of the piecewise linear segments).
        zero_prefix = torch.zeros(*self.batch_shape, 1, dtype=self.logits.dtype, device=self.logits.device)
        cdf_edges = torch.cat([zero_prefix, torch.cumsum(self.bin_probs, dim=-1)], dim=-1)

        # Find the bin in probability space.
        cdf_edges_aligned = (
            cdf_edges.view((1,) * num_sample_dims + cdf_edges.shape).expand(*value_prep.shape, -1).contiguous()
        )
        bin_indices = self._get_bin_indices(value_prep.unsqueeze(-1), bin_edges=cdf_edges_aligned)

        # Gather the probability base.
        base_cdf = self._gather_from_bins(cdf_edges, bin_indices, num_sample_dims, value_prep.shape)

        # Gather the interpolation parameters.
        left_edges = self._gather_from_bins(self.bin_edges[:-1], bin_indices, num_sample_dims, value_prep.shape)
        widths = self._gather_from_bins(self.bin_widths, bin_indices, num_sample_dims, value_prep.shape)
        masses = self._gather_from_bins(self.bin_probs, bin_indices, num_sample_dims, value_prep.shape)

        # Interpolate: x = x0 + (target_y - y0) * (width / mass)
        # Handle division by zero for bins with no mass
        slope = widths / (masses + 2 * masses.dtype.eps)
        interp_value = left_edges + (value_prep - base_cdf) * slope

        quantiles = torch.clamp(interp_value, self.bound_low, self.bound_up)

        return quantiles

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
