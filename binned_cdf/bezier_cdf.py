import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints

_size = torch.Size()


class BezierCDF(Distribution):
    r"""A continuous probability distribution parameterized by Bernstein polynomials with custom constraints.

    The idea is that the CDF is represented as a Bezier curve, which is a weighted sum of Bernstein basis polynomials,
    defined by control points (betas) that are derived from the input logits.
    This allows for a smooth, flexible CDF that can capture complex shapes while still being differentiable.
    In fact, this formulation is mathematically equivalent to a mixture of Beta distributions, where the mixture
    weights are given by the deltas (softmax of the logits) and the Beta components are defined by the control points.

    Since we know that any CDF must start at 0 and end at 1, we can enforce these constraints by fixing the first
    control point to 0 and the last control point to 1.

    The spacing of the control points along the domain-axis ("x-axis") is strictly uniform and determined by the
    degree of the Bernstein polynomial, hence, number of input logits.

    Note:
        Bernstein polynomials converge slowly: the worst-case pointwise approximation error is $O(1/n)$ where $n$ is
        the polynomial degree, leading to a standard deviation error of $O(1/\sqrt{n})$. However, for smooth CDFs the
        effective rate is better, and Bernstein density estimators achieve the optimal minimax rate (Babu et al., 2002;
        Petrone, 1999). This slower convergence is an inherent trade-off for the structural guarantees they provide:
        monotonicity, values in $[0, 1]$, non-negative PDF, and an unconstrained parameterization (any real-valued
        logits yield a valid distribution). No other polynomial basis offers all of these simultaneously. In practice,
        the bias matters less when logits are learned end-to-end via gradient descent, as the optimizer can compensate.
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
            logits: Raw logits for the probabilities before normalization, of shape (*batch_shape, degree).
                The logits also determine the degree of the Bernstein polynomial $n$.
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
            deltas: Weights of the Beta components in the mixture, of shape (*batch_shape, degree)
            betas: Control points of the Bezier curve, of shape (*batch_shape, degree + 1)
        """
        # The deltas are the steps themselves (forward differences of betas).
        if self.normalization_method == "softmax":
            deltas = torch.softmax(self.logits, dim=-1)  # shape: (*batch_shape, degree)

        elif self.normalization_method == "sigmoid":
            raw_deltas = torch.sigmoid(self.logits)
            sum_deltas = raw_deltas.sum(dim=-1, keepdim=True)

            # Prevent division by zero in the rare case where all logits are massively negative.
            eps = torch.finfo(raw_deltas.dtype).eps
            sum_deltas = sum_deltas.clamp_min(eps)

            deltas = raw_deltas / sum_deltas  # shape: (*batch_shape, degree)

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        # Pad with zeros and ones to enforce the CDF boundary conditions:
        # betas = [0, beta_0, ..., beta_{n-2}, 1]
        zeros = torch.zeros(*deltas.shape[:-1], 1, device=deltas.device, dtype=deltas.dtype)  # shape: (*batch_shape, 1)
        inner_betas = torch.cumsum(deltas, dim=-1)[..., :-1]
        ones = torch.ones(*deltas.shape[:-1], 1, device=deltas.device, dtype=deltas.dtype)
        betas = torch.cat([zeros, inner_betas, ones], dim=-1)

        return deltas, betas

    def _map_to_t_space(self, value: torch.Tensor) -> torch.Tensor:
        r"""Map values from the original $X$ space to the $T$ space $[0, 1]$ using the bounds."""
        return torch.clamp((value - self.bound_low) / self.support_range, 0, 1)

    def _map_to_x_space(self, t: torch.Tensor) -> torch.Tensor:
        r"""Map values from the $T$ space $[0, 1]$ back to the original $X$ space using the bounds."""
        return t * self.support_range + self.bound_low

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

    @property
    def degree(self) -> int:
        r"""Get the degree $n$ of the Bernstein polynomial based on the number of logits.

        For a Bernstein polynomial of degree $n$, there are $n + 1$ control points (betas) and $n$ weights (deltas).
        """
        return self.logits.shape[-1]

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
        i = torch.arange(self.degree, device=self._deltas.device, dtype=self._deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self._deltas * (i + 1) / (self.degree + 1), dim=-1)

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
        i = torch.arange(self.degree, device=self._deltas.device, dtype=self._deltas.dtype)  # shape: (degree,)
        e_t = torch.sum(self._deltas * (i + 1) / (self.degree + 1), dim=-1)
        e_t2 = torch.sum(self._deltas * ((i + 1) * (i + 2)) / ((self.degree + 1) * (self.degree + 2)), dim=-1)
        var_t = e_t2 - e_t**2

        return self.support_range**2 * var_t

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
                Expected shape: (*batch_shape, n + 1).
            binom_coeffs: Precomputed binomial coefficients corresponding to the polynom's degree.
                Expected shape: (n,).

        Returns:
            The evaluated polynomial values.
            Output shape: (*sample_shape, *batch_shape)
        """
        # Get n which can be != self.degree as we use this method for both CDF and PDF which have different degrees.
        n = binom_coeffs.shape[0]

        # Create a tensor of indices matching the number of basis polynomials.
        i = torch.arange(n, device=t.device, dtype=t.dtype)

        # Add an empty dimension to t for broadcasting, resulting in shape: (*sample_shape, *batch_shape, 1).
        t_expanded = t.unsqueeze(-1)

        # Compute the entire basis in one shot.
        # PyTorch broadcasts the shapes to shape (*sample_shape, *batch_shape, degree).
        basis = binom_coeffs * (t_expanded**i) * ((1 - t_expanded) ** (n - 1 - i))

        # Multiply by weights and sum across the final dimension, resulting in shape (*sample_shape, *batch_shape).
        return torch.sum(weights * basis, dim=-1)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute cumulative distribution function at given values.

        Args:
            value: Values at which to compute the CDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            CDF values in [0, 1] corresponding to the input values. Output shape: same as `value` argument.
        """
        x = value.to(device=self.logits.device, dtype=self.logits.dtype)

        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = self._map_to_t_space(x)

        # Construct and evaluate the Bezier curve in T space.
        return self._eval_bezier_curve(t, weights=self._betas, binom_coeffs=self._binom_coeffs_cdf)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute probability density at given values.

        Args:
            value: Values at which to compute the PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            PDF values corresponding to the input values. Output shape: same as `value` argument.
        """
        x = value.to(device=self.logits.device, dtype=self.logits.dtype)

        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = self._map_to_t_space(x)

        # Construct and evaluate the Bezier curve in T space.
        val = self._eval_bezier_curve(t, weights=self._deltas, binom_coeffs=self._binom_coeffs_pdf)

        # Apply the chain rule: dt/dx = 1 / (U - L).
        pdf_val = val * self.degree / self.support_range

        # Mask out values outside [bound_low, bound_up].
        mask = (value >= self.bound_low) & (value <= self.bound_up)
        return torch.where(mask, pdf_val, torch.zeros_like(pdf_val))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability density at given values.

        Args:
            value: Values at which to compute the log-PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            Log-PDF values corresponding to the input values. Output shape: same as `value` argument.
        """
        pdf_val = self.prob(value)

        # Add an epsilon to prevent log(0) exactly at the boundaries.
        eps = torch.finfo(pdf_val.dtype).eps
        return torch.log(pdf_val + 2 * eps)

    def entropy(self, num_quadrature_points: int = 251) -> torch.Tensor:
        r"""Compute differential entropy of the distribution via numerical quadrature.

        $$ H(X) = -\int_{L}^{U} p(x) \log p(x) \, dx $$

        where $L$ and $U$ are the lower and upper bounds of the distribution support, respectively.

        Args:
            num_quadrature_points: Number of points for the trapezoidal rule approximation.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        # Create quadrature points over the support.
        x = torch.linspace(
            self.bound_low, self.bound_up, num_quadrature_points, device=self.logits.device, dtype=self.logits.dtype
        )

        # For batched distributions, expand quadrature points to shape (num_quadrature_points, *batch_shape)
        # so prob/log_prob receive values with explicit batch dimensions.
        x_eval = x.reshape(num_quadrature_points, *([1] * len(self.batch_shape)))
        x_eval = x_eval.expand(num_quadrature_points, *self.batch_shape)

        # Evaluate PDF at quadrature points.
        pdf_val = self.prob(x_eval)  # shape: (num_quadrature_points, *batch_shape)

        # Compute the integrand: -p(x) * log(p(x)), with epsilon for stability.
        eps = torch.finfo(pdf_val.dtype).eps
        log_pdf = torch.log(pdf_val + 2 * eps)
        integrand = -pdf_val * log_pdf  # shape: (num_quadrature_points, *batch_shape)

        # Integrate using the trapezoidal rule.
        return torch.trapezoid(integrand, x, dim=0)

    def icdf(
        self,
        value: torch.Tensor,
        num_iter: int = 8,
        use_newton: bool = True,
        convergence_eps_factor: float = 20.0,
    ) -> torch.Tensor:
        r"""Compute the inverse CDF, i.e., the quantile function, at the given values.

        Two solvers are available for inverting $ F(x) - q = 0 $:

        **Newton's method** uses the PDF as the exact derivative of the CDF and iterates

        $$ x_{k+1} = x_k - \frac{F(x_k) - q}{f(x_k)} $$

        where $F(x)$ is the CDF, $f(x)$ is the PDF, and $q$ is the target quantile in [0, 1].
        This achieves quadratic convergence (number of correct digits roughly doubles each iteration).
        Each step is projected back onto the support $[L, U]$ to prevent divergence
        when the PDF is near zero. The loop exits early once all elements satisfy $|F(x) - q| < \epsilon$.

        **Bisection** halves the search interval each iteration, gaining ~1 bit of precision per step.

        Args:
            value: Values in [0, 1] at which to compute the inverse CDF. Expected shape: (*sample_shape, *batch_shape).
            num_iter: Maximum number of solver iterations. Newton typically converges in ~5-6 iterations;
                bisection needs ~15-20 for full float32 precision.
            use_newton: If True, use Newton's method. If False, use pure bisection.
            convergence_eps_factor: The factor multiplied by machine epsilon to determine the convergence criterion.

        Returns:
            Quantiles in [bound_low, bound_up] corresponding to the input CDF values.
            Output shape: same as `value` argument.
        """

        def _has_converged(cdf_mid: torch.Tensor, q: torch.Tensor, eps: float, convergence_eps_factor: float) -> bool:
            """Check if all elements have converged based on the CDF values at the current midpoint.

            We use the somewhat arbitrary criterion that the maximum absolute deviation across all elements is less than
            `convergence_eps_factor` times machine epsilon.
            """
            abs_deviation = (cdf_mid - q).abs().max()
            return bool(abs_deviation < convergence_eps_factor * eps)

        q = value.to(device=self.logits.device, dtype=self.logits.dtype)
        eps = torch.finfo(q.dtype).eps

        # Ensure target probability value is strictly in [0, 1].
        q = torch.clamp(q, 0.0, 1.0)

        # Start from the midpoint of the support.
        mid = torch.full_like(q, (self.bound_low + self.bound_up) / 2)

        if use_newton:
            # Root finding via Newton's method.
            for _ in range(num_iter):
                cdf_mid = self.cdf(mid)

                # Early stop when all elements have converged.
                if _has_converged(cdf_mid, q, eps, convergence_eps_factor):
                    break

                # Newton step: x_{k+1} = x_k - (F(x_k) - q) / f(x_k).
                pdf_mid = self.prob(mid)
                mid = mid - (cdf_mid - q) / pdf_mid.clamp_min(2 * eps)

                # Project back onto the support.
                mid = mid.clamp(min=self.bound_low, max=self.bound_up)

        else:
            # Root finding via bisection method.
            low = torch.full_like(q, self.bound_low)
            high = torch.full_like(q, self.bound_up)

            for _ in range(num_iter):
                cdf_mid = self.cdf(mid)

                # Early stop when all elements have converged.
                if _has_converged(cdf_mid, q, eps, convergence_eps_factor):
                    break

                # Bisection step: update low or high based on whether cdf(mid) is less than or greater than q.
                low = torch.where(cdf_mid < q, input=mid, other=low)
                high = torch.where(cdf_mid >= q, input=mid, other=high)
                mid = (low + high) / 2

        return mid

    def rsample(self, sample_shape: torch.Size | list[int] | tuple[int, ...] = _size) -> torch.Tensor:
        """Draws reparameterized samples from the distribution, and allows gradients to flow backawards.

        Args:
            sample_shape: Desired shape of the samples to be drawn. Default is empty, which means one sample per batch element.

        Returns:
            Samples drawn from the distribution, with shape (*sample_shape, *batch_shape).
        """
        # Determine the final shape of the output tensor.
        shape = self._extended_shape(sample_shape)

        # Sample uniform noise, u ~ U(0, 1).
        u = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)

        # Find the root (the sample x) without tracking gradients for the loop.
        with torch.no_grad():
            x_root = self.icdf(u)

        # Apply the implicit differentiation trick, i.e., evaluate CDF to connect the parameters to the
        # computational graph.
        cdf_val = self.cdf(x_root)

        # Evaluate PDF and detach it to act as the constant denominator.
        pdf_val = self.prob(x_root).detach()

        # Clamp PDF to avoid division by zero near the boundaries where slope is 0. This limits the gradients.
        eps = torch.finfo(pdf_val.dtype).eps
        pdf_val = pdf_val.clamp_min(2 * eps)

        # Attach the exact reparameterized gradient.
        x = x_root + (u - cdf_val) / pdf_val

        # Clamp to the support to prevent the implicit-differentiation correction from pushing samples
        # slightly past the domain boundaries when the CDF is very flat near the bounds.
        return x.clamp(min=self.bound_low, max=self.bound_up)
