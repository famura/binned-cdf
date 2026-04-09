import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints

_size = torch.Size()


class BernsteinDistribution(Distribution):
    """A continuous probability distribution parameterized by Bernstein polynomials.

    The idea is that the CDF is represented as a Bezier curve, which is a weighted sum of Bernstein basis polynomials,
    defined by control points (betas) that are derived from the input logits.
    This allows for a smooth, flexible CDF that can capture complex shapes while still being differentiable.
    In fact, this formulation is mathematically equivalent to a mixture of Beta distributions, where the mixture
    weights are given by the deltas (softmax of the logits) and the Beta components are defined by the control points.

    The spacing of the control points along the domain-axis ("x-axis") is strictly uniform and determined by the
    degree of the Bernstein polynomial, hence, number of input logits.
    """

    has_rsample = True

    def __init__(
        self,
        logits: torch.Tensor,
        bound_low: float = -1e3,
        bound_up: float = 1e3,
        normalization_method: Literal["sigmoid", "softmax"] = "sigmoid",
        validate_args: bool | None = None,
    ) -> None:
        """Initializer.

        Args:
            logits: Raw logits for the probabilities (before sigmoid), of shape (*batch_shape, dim_logits).
                The logits also determine the degree of the Bernstein polynomial, which is dim_logits + 2.
            bound_low: Lower bound of the distribution support, needs to be finite.
            bound_up: Upper bound of the distribution support, needs to be finite.
            normalization_method: How to normalize the probabilities. Either "sigmoid" or "softmax". With "sigmoid",
                each control point is independently activated, while with "softmax", the control point activations
                influence each other.
            validate_args: Whether to validate arguments. Carried over to keep the interface with the base class.
        """
        self.logits = logits
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.normalization_method = normalization_method

        # Precompute binomial coefficients, and store them on the same device as logits.
        # comb(n, k) = n! / (k! * (n-k)!) is the binomial coefficient, which counts the number of ways to choose k
        # elements from a set of n elements.
        self._cdf_combs = torch.tensor(
            [math.comb(self.degree, i) for i in range(self.degree + 1)], device=logits.device, dtype=logits.dtype
        )
        self._pdf_combs = torch.tensor(
            [math.comb(self.degree - 1, i) for i in range(self.degree)],
            device=logits.device,
            dtype=logits.dtype,
        )

        # Calculate parameters (deltas and betas).
        self.deltas, self.betas = self._compute_coefficients()

        # Determine batch shape based on the logits. The event shape is scalar since this is a univariate distribution.
        super().__init__(batch_shape=logits.shape[:-1], event_shape=torch.Size([]), validate_args=validate_args)

    @property
    def degree(self) -> int:
        """Get the degree of the Bernstein polynomial based on the number of logits.

        For a Bernstein polynomial of degree n, there are n + 1 control points (betas) and n deltas (weights).
        """
        return self.logits.shape[-1] + 2

    def _compute_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the deltas (Beta mixture component weights) and betas (control points) for the Bezier curve based
        on the given logits.

        The deltas are the forward differences of the betas
        $$ \Delta_i = \beta_{i + 1} - \beta_i $$

        Returns:
            deltas: Weights of the Beta components in the mixture, of shape (*batch_shape, degree - 1)
            betas: Control points of the Bezier curve, of shape (*batch_shape, degree + 1)
        """
        if self.normalization_method == "softmax":
            steps = torch.softmax(self.logits, dim=-1)  # shape: (*batch_shape, dim_logits)

        elif self.normalization_method == "sigmoid":
            raw_steps = torch.sigmoid(self.logits)
            sum_steps = raw_steps.sum(dim=-1, keepdim=True)

            # Prevent division by zero in the rare case where all logits are massively negative.
            eps = torch.finfo(raw_steps.dtype).eps
            sum_steps = sum_steps.clamp_min(eps)

            steps = raw_steps / sum_steps  # shape: (*batch_shape, dim_logits)

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        # Pad the deltas with 0 for the flat start and flat end constraints.
        zeros = torch.zeros(*steps.shape[:-1], 1, device=steps.device, dtype=steps.dtype)  # shape: (*batch_shape, 1)
        deltas = torch.cat([zeros, steps, zeros], dim=-1)  # shape: (*batch_shape, dim_logits + 2)

        # Pad the betas with 0 and 1 for the flat start and flat end constraints.
        inner_betas = torch.cumsum(steps, dim=-1)[..., :-1]
        ones = torch.ones(*steps.shape[:-1], 2, device=steps.device, dtype=steps.dtype)
        betas = torch.cat([zeros, zeros, inner_betas, ones], dim=-1)

        return deltas, betas

    @property
    def mean(self) -> torch.Tensor:
        r"""Compute mean of the distribution, i.e., the weighted average of the control points.

        We transform the random variable $X$ to $T$ in [0, 1] by scaling and shifting according to the bounds.
        Then, the mean of $T$ can be computed as

        $$ E[T] = \sum_{i=0}^{n-1} \Delta_i \frac{i+1}{n+1} $$

        where $\Delta_i$ is the weight of the $i$-th control point, and $n$ is the degree of the Bernstein polynomial.
        We can then get the mean by rescaling $E[T]$ back to the original support:

        $$ E[X] = (U - L) E[T] + L $$

        where $L$ and $U$ are the lower and upper bounds of the distribution support, respectively.

        Note:
            This method uses the exact Beta mixture formula.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        i_vals = torch.arange(self.degree, device=self.deltas.device, dtype=self.deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self.deltas * (i_vals + 1) / (self.degree + 1), dim=-1)

        return (self.bound_up - self.bound_low) * e_t + self.bound_low

    @property
    def variance(self) -> torch.Tensor:
        r"""Compute variance of the distribution.

        We transform the random variable $X$ to $T$ in [0, 1] by scaling and shifting according to the bounds.
        Then, the variance of $T$ can be computed as

        $$ Var[T] = E[T^2] - (E[T])^2 $$

        with

        $$ E[T^2] = \sum_{i=0}^{n-1} \Delta_i \frac{(i+1)(i+2)}{(n+1)(n+2)} $$

        where $\Delta_i$ is the weight of the $i$-th control point, and $n$ is the degree of the Bernstein polynomial.
        We can then get the variance by rescaling $Var[T]$ back to the original support:

        $$ Var[X] = (U - L)^2 Var[T] $$

        Note:
            This method uses the exact Beta mixture formula.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        i_vals = torch.arange(self.degree, device=self.deltas.device, dtype=self.deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self.deltas * (i_vals + 1) / (self.degree + 1), dim=-1)
        e_t2 = torch.sum(self.deltas * ((i_vals + 1) * (i_vals + 2)) / ((self.degree + 1) * (self.degree + 2)), dim=-1)
        var_t = e_t2 - e_t**2

        return ((self.bound_up - self.bound_low) ** 2) * var_t

    @property
    def support(self) -> constraints.Constraint:
        """Support of this distribution."""
        return constraints.interval(self.bound_low, self.bound_up)

    @property
    def support_range(self) -> float:
        """Range of the support, i.e., upper bound - lower bound."""
        return self.bound_up - self.bound_low

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Constraints that should be satisfied by each argument of this distribution. None for this class."""
        return {"logits": constraints.real, "bound_low": constraints.real, "bound_up": constraints.real}

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute cumulative distribution function at given values.

        Args:
            value: Values at which to compute the CDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            CDF values in [0, 1] corresponding to the input values. Output shape: same as `value` argument.
        """
        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = torch.clamp((value - self.bound_low) / self.support_range, 0.0, 1.0)

        # Evaluate Bezier curve defined in the T space.
        val = torch.zeros_like(t)
        for i in range(self.degree + 1):
            basis = self._cdf_combs[i] * (t**i) * ((1 - t) ** (self.degree - i))
            val += self.betas[..., i] * basis

        return val

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability density at given values.

        Args:
            value: Values at which to compute the log-PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            Log-PDF values corresponding to the input values. Output shape: same as `value` argument.
        """
        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = torch.clamp((value - self.bound_low) / self.support_range, 0.0, 1.0)

        val = torch.zeros_like(t)
        for i in range(self.degree):
            basis = self._pdf_combs[i] * (t**i) * ((1 - t) ** (self.degree - 1 - i))
            val += self.deltas[..., i] * basis

        # Apply the chain rule: dt/dx = 1 / (U - L).
        pdf_val = val * self.degree / self.support_range

        # Mask out values outside [bound_low, bound_up] to prevent log(0) issues.
        mask = (value >= self.bound_low) & (value <= self.bound_up)
        pdf_val = torch.where(mask, pdf_val, torch.zeros_like(pdf_val))

        # Add a tiny epsilon to prevent log(0) exactly at the boundaries.
        eps = torch.finfo(pdf_val.dtype).eps
        return torch.log(pdf_val + 2 * eps)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute probability density at given values.

        Args:
            value: Values at which to compute the PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            PDF values corresponding to the input values. Output shape: same as `value` argument.
        """
        return torch.exp(self.log_prob(value))

    def icdf(self, value: torch.Tensor, num_bisect_iter: int = 20) -> torch.Tensor:
        """Compute the inverse CDF, i.e., the quantile function, at the given values.

        Note:
            We are using a batched bisection search to invert the CDF.

        Args:
            value: Values in [0, 1] at which to compute the inverse CDF. Expected shape: (*sample_shape, *batch_shape).
            num_bisect_iter: Number of bisection iterations to perform. 20 iterations yield ~20 bits of precision
                (more than enough for floats).

        Returns:
            Quantiles in [bound_low, bound_up] corresponding to the input CDF values.
            Output shape: same as `value` argument.
        """
        # Ensure target probability value is strictly in [0, 1].
        value = torch.clamp(value, 0.0, 1.0)

        # Create search interval tensors.
        low = torch.full_like(value, self.bound_low)
        high = torch.full_like(value, self.bound_up)

        # Run the batched bisection search.
        for _ in range(num_bisect_iter):
            # Compute current midpoint and its CDF value.
            mid = (low + high) / 2
            cdf_mid = self.cdf(mid)
            # If current CDF is too low, the root is to the right. Thus we set the new low to the current mid.
            low = torch.where(cdf_mid < value, input=mid, other=low)
            # If current CDF is too high, the root is to the left. Thus we set the new high to the current mid.
            high = torch.where(cdf_mid >= value, input=mid, other=high)

        return (low + high) / 2

    def rsample(self, sample_shape: torch.Size | list[int] | tuple[int, ...] = _size) -> torch.Tensor:
        """Draws reparameterized samples from the distribution, and allows gradients to flow backawards.

        Args:
            sample_shape: Desired shape of the samples to be drawn. Default is empty, which means one sample per batch element.

        Returns:
            Samples drawn from the distribution, with shape (*sample_shape, *batch_shape).
        """
        # Determine the final shape of the output tensor.
        shape = self._extended_shape(sample_shape)

        # 1. Sample uniform noise, u ~ U(0, 1)
        u = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)

        # 2. Find the root (the sample x) without tracking gradients for the loop
        with torch.no_grad():
            x_root = self.icdf(u)

        # 3. Apply the implicit differentiation trick
        # Evaluate CDF to connect the parameters (theta) to the computational graph
        cdf_val = self.cdf(x_root)

        # Evaluate PDF and detach it to act as the constant denominator.
        pdf_val = self.prob(x_root).detach()

        # Clamp PDF to avoid division by zero near the boundaries where slope is 0. This limits the gradients.
        eps = torch.finfo(pdf_val.dtype).eps
        pdf_val = pdf_val.clamp_min(2 * eps)

        # Attach the exact reparameterized gradient.
        x = x_root + (u - cdf_val) / pdf_val

        return x
