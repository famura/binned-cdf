import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints

_size = torch.Size()


class BezierCDF(Distribution):
    """A continuous probability distribution parameterized by Bernstein polynomials with custom constraints.

    The idea is that the CDF is represented as a Bezier curve, which is a weighted sum of Bernstein basis polynomials,
    defined by control points (betas) that are derived from the input logits.
    This allows for a smooth, flexible CDF that can capture complex shapes while still being differentiable.
    In fact, this formulation is mathematically equivalent to a mixture of Beta distributions, where the mixture
    weights are given by the deltas (softmax of the logits) and the Beta components are defined by the control points.

    Since we know that any CDF must start at 0 and end at 1, we can enforce these constraints by fixing the first
    control point to 0 and the last control point to 1. Moreover, we know that the 1st derivative of the CDF at the
    boundaries must be 0, which means the 2nd must also be 0 and the 2nd-to-last control point must also be 1.

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
        self._binom_coeffs_cdf, self._binom_coeffs_pdf = self._compute_binomial_coefficients()

        # Calculate parameters (deltas and betas).
        self._deltas, self._betas = self._compute_deltas_and_betas()

        # Determine batch shape based on the logits. The event shape is scalar since this is a univariate distribution.
        super().__init__(batch_shape=logits.shape[:-1], event_shape=torch.Size([]), validate_args=validate_args)

    def __repr__(self) -> str:
        """String representation of the distribution."""
        return (
            f"{self.__class__.__name__}(logits_shape: {self.logits.shape}, bound_low: {self.bound_low}, "
            f"bound_up: {self.bound_up}, normalization_method: {self.normalization_method})"
        )

    def _compute_binomial_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the binomial coefficients for the CDF and PDF based on the degree of the Bernstein polynomial.

        comb(n, k) = n! / (k! * (n-k)!) is the binomial coefficient, which counts the number of ways to choose k
        elements from a set of n elements.

        Returns:
            coeffs_cdf: Binomial coefficients for the CDF, of shape (degree + 1,)
            coeffs_pdf: Binomial coefficients for the PDF, of shape (degree,)
        """
        coeffs_cdf = torch.tensor(
            [math.comb(self.degree, i) for i in range(self.degree + 1)],
            device=self.logits.device,
            dtype=self.logits.dtype,
        )

        coeffs_pdf = torch.tensor(
            [math.comb(self.degree - 1, i) for i in range(self.degree)],
            device=self.logits.device,
            dtype=self.logits.dtype,
        )

        # Check if any of the binomial coefficients became infinite.
        if torch.isinf(coeffs_cdf).any() or torch.isinf(coeffs_pdf).any():
            raise ValueError(
                f"Binomial coefficients became infinite for degree {self.degree}. "
                "Consider reducing the (last) dimension of the logits, leading to lower degree polynomial."
            )

        return coeffs_cdf, coeffs_pdf

    def _compute_deltas_and_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the deltas (Beta mixture component weights) and betas (control points) for the Bezier curve based
        on the given logits.

        The deltas are the forward differences of the betas, i.e., $ \Delta_i = \beta_{i + 1} - \beta_i $.

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
        # deltas = [0, delta_0, ..., delta_{n-1}, 0]
        zeros = torch.zeros(*steps.shape[:-1], 1, device=steps.device, dtype=steps.dtype)  # shape: (*batch_shape, 1)
        deltas = torch.cat([zeros, steps, zeros], dim=-1)  # shape: (*batch_shape, dim_logits + 2)

        # Pad the betas with 0 and 1 for the flat start and flat end constraints.
        # betas = [0, 0, beta_0, ..., beta_{n-1}, 1, 1]
        inner_betas = torch.cumsum(steps, dim=-1)[..., :-1]
        ones = torch.ones(*steps.shape[:-1], 2, device=steps.device, dtype=steps.dtype)
        betas = torch.cat([zeros, zeros, inner_betas, ones], dim=-1)

        return deltas, betas

    def _map_to_t_space(self, value: torch.Tensor) -> torch.Tensor:
        r"""Map values from the original $X$ space to the $T$ space $[0, 1]$ using the bounds."""
        return torch.clamp((value - self.bound_low) / self.support_range, 0, 1)

    def _map_to_x_space(self, t: torch.Tensor) -> torch.Tensor:
        r"""Map values from the $T$ space $[0, 1]$ back to the original $X$ space using the bounds."""
        return t * self.support_range + self.bound_low

    @property
    def degree(self) -> int:
        """Get the degree of the Bernstein polynomial based on the number of logits.

        For a Bernstein polynomial of degree n, there are n + 1 control points (betas) and n deltas (weights).
        """
        return self.logits.shape[-1] + 2

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
        i_vals = torch.arange(self.degree, device=self._deltas.device, dtype=self._deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self._deltas * (i_vals + 1) / (self.degree + 1), dim=-1)

        return self._map_to_x_space(e_t)

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
        i_vals = torch.arange(self.degree, device=self._deltas.device, dtype=self._deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self._deltas * (i_vals + 1) / (self.degree + 1), dim=-1)
        e_t2 = torch.sum(self._deltas * ((i_vals + 1) * (i_vals + 2)) / ((self.degree + 1) * (self.degree + 2)), dim=-1)
        var_t = e_t2 - e_t**2

        return self.support_range**2 * var_t

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
        return {"logits": constraints.real}

    def _eval_bezier_curve(
        self,
        t: torch.Tensor,
        weights: torch.Tensor,
        binom_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Evaluates the Bezier curve, i.e., a Bernstein polynomial, in the $T \in [0, 1]$ space.

        This method computes the weighted sum of Bernstein basis polynomials, where each basis polynomial is defined as

        $$ B_{i, n-1}(t) = \binom{n-1}{i} t^i (1-t)^{n-1-i} $$

        where $n$ is the degree of the polynomial. The polynom's value $p(t)$ is computed as

        $$ p(t) = \sum_{i=0}^{n-1} w_i B_{i, n-1}(t) $$

        where $w_i$ are the weights (either betas for CDF or deltas for PDF).

        Args:
            t: Normalized input values in [0, 1].
                Expected shape: (*sample_shape, *batch_shape).
            weights: The coefficients for the basis polynomials (betas for CDF, deltas for PDF).
                Expected shape: (*batch_shape, degree + 1).
            binom_coeffs: Precomputed binomial coefficients corresponding to the polynom's degree.
                Expected shape: (degree + 1,).

        Returns:
            The evaluated polynomial values.
            Output shape: (*sample_shape, *batch_shape)
        """
        # Create a tensor of indices: [0, 1, ..., degree], of shape (degree + 1,).
        i = torch.arange(self.degree + 1, device=t.device, dtype=t.dtype)

        # Add an empty dimension to t for broadcasting, resulting in shape: (*sample_shape, *batch_shape, 1).
        t_exp = t.unsqueeze(-1)

        # Compute the entire basis in one shot.
        # PyTorch broadcasts the shapes to shape (*sample_shape, *batch_shape, degree + 1)
        basis = binom_coeffs * (t_exp**i) * ((1 - t_exp) ** (self.degree - i))

        # Multiply by weights and sum across the final dimension, resulting in shape (*sample_shape, *batch_shape).
        val = torch.sum(weights * basis, dim=-1)

        return val

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute cumulative distribution function at given values.

        Args:
            value: Values at which to compute the CDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            CDF values in [0, 1] corresponding to the input values. Output shape: same as `value` argument.
        """
        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = self._map_to_t_space(value)

        # Construct and evaluate the Bezier curve in T space.
        val = self._eval_bezier_curve(t, weights=self._betas, binom_coeffs=self._binom_coeffs_cdf)

        return val

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability density at given values.

        Args:
            value: Values at which to compute the log-PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            Log-PDF values corresponding to the input values. Output shape: same as `value` argument.
        """
        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = self._map_to_t_space(value)

        # Construct and evaluate the Bezier curve in T space.
        val = self._eval_bezier_curve(t, weights=self._deltas, binom_coeffs=self._binom_coeffs_pdf)

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
