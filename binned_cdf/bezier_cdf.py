import math
from typing import Literal

import torch
from torch.distributions import Distribution, constraints

_size = torch.Size()


class BezierCDF(Distribution):
    r"""A continuous probability distribution represented by a global parametric Bezier curve.

    The idea is that the CDF is represented as a Bezier curve, which is a weighted sum of Bernstein basis
    polynomials, defined by control points (betas) that are derived from the input logits.
    This allows for a smooth, flexible CDF that can capture complex shapes while still being differentiable.
    With two learned axes, this is a global parametric curve. Its vertical derivative defines a mixture of Beta
    densities in parameter space, with mixture weights given by the normalized vertical increments (deltas).

    Normalized, positive increments define monotone control points on each axis, with endpoints fixed at 0 and 1.
    For $t \in [0, 1]$,

    $$ x(t) = L + (U-L) B_x(t), \qquad F(x(t)) = B_y(t) $$

    so

    $$ p(x(t)) = \frac{B_y'(t)}{(U-L) B_x'(t)}. $$

    The two polynomials may have different degrees. Both logit tensors are required.
    A one-element ``logits_x`` fixes the horizontal curve to

    $$ B_x(t) = t. $$

    Learning multiple horizontal increments allows density to concentrate without increasing the vertical
    polynomial degree, but CDF and density evaluation then require numerical inversion. Quantiles only invert
    the vertical curve.

    Inversion uses bracketed Newton steps and implicit first-order gradients, without retaining the solver graph.
    No application-level minimum increment is imposed; extremely concentrated curves remain limited by the input
    dtype's precision. Higher-order derivatives through the implicit inverse are not supported.

    Note:
        Monotone control points provide the structural guarantees of a CDF: monotonicity, values in $[0, 1]$,
        and a non-negative PDF. In exact arithmetic, normalizing positive increments gives these guarantees
        for any finite logits, without imposing constraints on the parameters during optimization.

        With fixed horizontal spacing, the density is a Beta mixture on the normalized support. Classical
        Bernstein approximation has uniform error $O(1/n)$ for functions with a bounded second derivative;
        this is not a general convergence rate for an end-to-end learned two-axis model. A degree-$n$ mixture
        component near the center of a fixed support of width $R$ has approximate standard deviation

        $$ std \approx \frac{R}{2\sqrt{n}}. $$

        This explains the smoothing of narrow interior peaks with fixed spacing. It is not a universal minimum
        width: boundary components can be narrower, and learned horizontal spacing can further compress peaks.
    """

    has_rsample = True

    def __init__(
        self,
        logits_y: torch.Tensor,
        logits_x: torch.Tensor,
        bound_low: float = -1e3,
        bound_up: float = 1e3,
        normalization_method: Literal["sigmoid", "softmax"] = "softmax",
        validate_args: bool | None = None,
    ) -> None:
        """Initializer.

        Args:
            logits_y: Raw vertical-increment logits, of shape (*batch_shape, degree_y).
            logits_x: Raw horizontal-increment logits, of shape (*batch_shape, degree_x). Batch shapes must
                broadcast with logits_y, and both tensors must have the same floating dtype and device.
            bound_low: Lower bound of the distribution support, needs to be finite.
            bound_up: Upper bound of the distribution support, needs to be finite.
            normalization_method: How to normalize the probabilities. Either "sigmoid" or "softmax". With "sigmoid",
                each control point is independently activated, while with "softmax", the control point activations
                influence each other.
            validate_args: Whether to validate arguments. Carried over to keep the interface with the base class.
        """
        if not math.isfinite(bound_low) or not math.isfinite(bound_up) or bound_low >= bound_up:
            raise ValueError("Bounds must be finite with bound_low < bound_up.")
        if logits_y.ndim < 1 or logits_y.shape[-1] < 1 or not logits_y.is_floating_point():
            raise ValueError("logits_y must be a floating tensor with at least one increment.")
        if logits_x.ndim < 1 or logits_x.shape[-1] < 1 or not logits_x.is_floating_point():
            raise ValueError("logits_x must be a floating tensor with at least one increment.")
        if logits_x.dtype != logits_y.dtype or logits_x.device != logits_y.device:
            raise ValueError("logits_x and logits_y must have the same floating dtype and device.")

        batch_shape = torch.broadcast_shapes(logits_y.shape[:-1], logits_x.shape[:-1])
        self.logits_y = logits_y.expand(*batch_shape, logits_y.shape[-1])
        self.logits_x = logits_x.expand(*batch_shape, logits_x.shape[-1])
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.normalization_method = normalization_method

        # Precompute binomial coefficients, and store them on the same device as logits.
        self._binom_coeffs_cdf, self._binom_coeffs_pdf = self._compute_binomial_coefficients(self.degree_y)

        # Calculate parameters (deltas and betas).
        self._deltas, self._betas, self._log_deltas = self._compute_deltas_and_betas(self.logits_y)
        self._x_binom_coeffs_cdf, self._x_binom_coeffs_pdf = self._compute_binomial_coefficients(self.degree_x)
        self._x_deltas, self._x_betas, self._x_log_deltas = self._compute_deltas_and_betas(self.logits_x)

        # Determine batch shape based on the logits. The event shape is scalar since this is a univariate distribution.
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([]), validate_args=validate_args)

    def __repr__(self) -> str:
        """String representation of the distribution."""
        return (
            f"{self.__class__.__name__}(degree_y: {self.degree_y}, degree_x: {self.degree_x}, "
            f"batch_shape: {self.batch_shape}, bound_low: {self.bound_low}, "
            f"bound_up: {self.bound_up}, normalization_method: {self.normalization_method})"
        )

    def _compute_binomial_coefficients(self, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the binomial coefficients for the CDF and PDF based on the degree of the Bernstein polynomial.

        comb(n, k) = n! / (k! * (n-k)!) is the binomial coefficient, which counts the number of ways to choose k
        elements from a set of n elements.

        Returns:
            coeffs_cdf: Binomial coefficients for the CDF, of shape (degree + 1,).
            coeffs_pdf: Binomial coefficients for the PDF, of shape (degree,).
        """
        coeffs_cdf = torch.tensor(
            [math.comb(degree, i) for i in range(degree + 1)], device=self.logits_y.device, dtype=self.logits_y.dtype
        )

        coeffs_pdf = torch.tensor(
            [math.comb(degree - 1, i) for i in range(degree)], device=self.logits_y.device, dtype=self.logits_y.dtype
        )

        # Check if any of the binomial coefficients became infinite.
        if torch.isinf(coeffs_cdf).any() or torch.isinf(coeffs_pdf).any():
            raise ValueError(
                f"Binomial coefficients became infinite for degree {degree}. "
                "Consider reducing the (last) dimension of the logits, leading to lower degree polynomial."
            )

        return coeffs_cdf, coeffs_pdf

    def _compute_deltas_and_betas(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute the deltas (Beta mixture component weights) and betas (control points) for the Bezier curve based
        on the given logits.

        The deltas are the forward differences of the betas, i.e., $ \Delta_i = \beta_{i + 1} - \beta_i $.

        Returns:
            deltas: Weights of the Beta components in the mixture, of shape (*batch_shape, degree).
            betas: Control points of the Bezier curve, of shape (*batch_shape, degree + 1).
            log_deltas: Log of the deltas, computed in a numerically stable way, of shape (*batch_shape, degree)
        """
        # The deltas are the steps themselves (forward differences of betas).
        if self.normalization_method == "softmax":
            deltas = torch.softmax(logits, dim=-1)  # shape: (*batch_shape, degree)
            log_deltas = torch.log_softmax(logits, dim=-1)

        elif self.normalization_method == "sigmoid":
            log_deltas = torch.log_softmax(torch.nn.functional.logsigmoid(logits), dim=-1)
            deltas = log_deltas.exp()

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        # Pad with zeros and ones to enforce the CDF boundary conditions:
        # betas = [0, beta_1, ..., beta_{n-1}, beta_n = 1]
        zeros = torch.zeros(*deltas.shape[:-1], 1, device=deltas.device, dtype=deltas.dtype)  # shape: (*batch_shape, 1)
        inner_betas = torch.cumsum(deltas, dim=-1)[..., :-1]
        ones = torch.ones(*deltas.shape[:-1], 1, device=deltas.device, dtype=deltas.dtype)
        betas = torch.cat([zeros, inner_betas, ones], dim=-1)

        return deltas, betas, log_deltas

    def _map_to_t_space(self, value: torch.Tensor) -> torch.Tensor:
        r"""Map physical values to curve parameters by normalizing to [0, 1], clamping, and inverting $B_x(t)$."""
        value = torch.clamp((value - self.bound_low) / self.support_range, 0, 1)
        return self._invert_bezier(
            value, self._x_betas, self._x_deltas, self._x_binom_coeffs_cdf, self._x_binom_coeffs_pdf
        )

    def _map_to_x_space(self, t: torch.Tensor) -> torch.Tensor:
        r"""Map values from the $T$ space $[0, 1]$ back to the original $X$ space using the bounds."""
        t = self._eval_bezier_curve(t, self._x_betas, self._x_binom_coeffs_cdf)
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
        """Both logit tensors must contain real values."""
        return {"logits_y": constraints.real, "logits_x": constraints.real}

    @property
    def degree_y(self) -> int:
        r"""Get the degree $n$ of the Bernstein polynomial based on the number of logits.

        For a Bernstein polynomial of degree $n$, there are $n + 1$ control points (betas) and $n$ weights (deltas).
        """
        return self.logits_y.shape[-1]

    @property
    def degree_x(self) -> int:
        """Degree of the horizontal polynomial; one for the fixed affine map."""
        return self.logits_x.shape[-1]

    @property
    def mean(self) -> torch.Tensor:
        r"""Compute the mean by integrating the parametric curve.

        $$ E[X] = L + (U-L) \int_0^1 B_x(t) B_y'(t) \, dt $$

        Note:
            Gaussian quadrature is polynomial-exact here, up to floating-point error.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        e_t, _ = self._moments()
        return self.bound_low + self.support_range * e_t

    @property
    def variance(self) -> torch.Tensor:
        r"""Compute variance from the normalized horizontal coordinate's first two raw moments.

        For $Z=(X-L)/(U-L)$, $E[Z^k]=\int_0^1 B_x(t)^k B_y'(t)\,dt$, and

        $$ Var[X] = (U-L)^2 (E[Z^2] - E[Z]^2) $$

        Note:
            Gaussian quadrature is polynomial-exact here, up to floating-point error.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        e_t, e_t2 = self._moments()
        return self.support_range**2 * (e_t2 - e_t.square()).clamp_min(0)

    def _quadrature(self, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Generate Gauss-Legendre nodes and weights on $[0, 1]$ using the symmetric Jacobi matrix.

        For $n=\text{num_points}$, the Golub-Welsch construction forms a symmetric tridiagonal matrix
        $J \in \mathbb{R}^{n \times n}$ from the recurrence coefficients of the orthonormal Legendre polynomials:

        $$ J_{k,k}=0, \qquad J_{k,k+1}=J_{k+1,k}=\frac{k}{\sqrt{4k^2-1}},
        \qquad k=1,\ldots,n-1, $$

        with all other entries zero. Its eigenvalues $\lambda_i$ are the roots of the degree-$n$ Legendre
        polynomial, hence the quadrature nodes on $[-1, 1]$. For orthonormal eigenvectors $v_i$ satisfying
        $Jv_i=\lambda_i v_i$, the corresponding weights on that interval are $2(v_i)_1^2$, where $(v_i)_1$
        is the first component of $v_i$. Shifting the nodes and halving the weights gives the rule on $[0, 1]$:

        $$ t_i=\frac{\lambda_i+1}{2}, \qquad w_i=(v_i)_1^2, \qquad
        \int_0^1 f(t)\,dt \approx \sum_{i=1}^{n} w_i f(t_i). $$

        The rule is exact for polynomials of degree at most $2n-1$, up to floating-point error.
        The weights integrate with respect to $dt$; callers supply any additional density factor.

        Note:
            Both tensors use the logits' dtype and device, which must be supported by torch.linalg.eigh.

        Args:
            num_points: Number $n$ of quadrature nodes; must be at least one. Increasing this count raises
                the maximum polynomial degree integrated exactly and refines the approximation for smooth,
                non-polynomial integrands.

        Returns:
            nodes: Quadrature locations in $[0, 1]$, ordered increasingly, with shape
                (num_points,) + (1,) * len(self.batch_shape). The trailing singleton dimensions allow
                the shared quadrature nodes to broadcast across the distribution's batch dimensions.
            weights: Positive integration weights corresponding to nodes, with the same shape. They sum
                to one along the quadrature dimension, up to floating-point error.
        """
        indices = torch.arange(1, num_points, dtype=self.logits_y.dtype, device=self.logits_y.device)
        off_diagonal = indices / (4 * indices.square() - 1).sqrt()

        nodes, vectors = torch.linalg.eigh(torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1))

        shape = (num_points,) + (1,) * len(self.batch_shape)
        nodes = ((nodes + 1) / 2).reshape(shape)
        first_comp_of_vectors = vectors[0]  # 1st component of each eigenvector which represents constants in the L-poly
        weights = first_comp_of_vectors.square().reshape(shape)

        return nodes, weights

    def _moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the first two raw moments of the normalized horizontal coordinate by Gaussian quadrature.

        For support bounds $L$ and $U$, let $Z = (X - L) / (U - L) = B_x(T)$. The curve parameter $T$
        has CDF $B_y(t)$ and density $B_y'(t)$ on $[0, 1]$. Changing variables from $X$ to $T$ cancels
        the horizontal Jacobian:

        $$ p_X(x(t)) x'(t) = B_y'(t), \qquad
        E[Z^k] = \int_0^1 B_x(t)^k B_y'(t) \, dt, \qquad k \in \{1, 2\}. $$

        The vertical derivative is evaluated from its degree-$(n_y - 1)$ Bernstein basis:

        $$ B_y'(t) = n_y \sum_{j = 0}^{n_y - 1} \Delta_j B_{j, n_y - 1}(t), $$

        where $n_y = \text{degree_y}$ and $\Delta_j$ are the normalized vertical increments. For Gauss-Legendre
        nodes $t_i$ and weights $w_i$, the code forms horizontal values $h_i$ and integration masses $a_i$:

        $$ h_i = B_x(t_i), \qquad a_i = w_i B_y'(t_i), \qquad
        e_t = \sum_{i = 1}^{m} a_i h_i, \qquad e_{t2} = \sum_{i = 1}^{m} a_i h_i^2. $$

        The masses sum to one up to floating-point error. Horizontal values and masses have shape
        (m, *batch_shape); summing along dimension zero removes only the quadrature dimension.

        With $n_x = \text{degree_x}$, the integrand for moment $k$ has degree at most $k n_x + n_y - 1$.
        An $m$-point Gauss-Legendre rule integrates polynomials through degree $2 m - 1$ exactly, so both
        moments are covered by the node count used here:

        $$ m = \left\lceil \frac{2 n_x + n_y}{2} \right\rceil
        = \left\lfloor \frac{2 n_x + n_y + 1}{2} \right\rfloor. $$

        Note:
            Both moments are polynomial-exact up to floating-point error and retain gradients through both axes.
            Tensors use the logits' dtype and device, subject to the requirements of _quadrature.
            No numerical inversion or support rescaling is performed here; mean and variance apply the bounds.

        Returns:
            e_t: First raw moment $E[Z]$, with shape (*batch_shape,).
            e_t2: Second raw moment $E[Z^2]$, with shape (*batch_shape,); this is not the variance.
                Despite their names, these are moments of $Z$, not of $T$, unless $B_x(t) = t$.
        """
        nodes, weights = self._quadrature(num_points=(2 * self.degree_x + self.degree_y + 1) // 2)

        horizontal = self._eval_bezier_curve(nodes, self._x_betas, self._x_binom_coeffs_cdf)

        masses = weights * self.degree_y * self._eval_bezier_curve(nodes, self._deltas, self._binom_coeffs_pdf)

        return (masses * horizontal).sum(dim=0), (masses * horizontal.square()).sum(dim=0)

    def _eval_bezier_curve(
        self,
        t: torch.Tensor,
        weights: torch.Tensor,
        binom_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Evaluates a Bezier curve (a Bernstein polynomial) in the $T \in [0, 1]$ space.

        This method computes the weighted sum of Bernstein basis polynomials. Let $d$ be the degree of the polynomial
        being evaluated ($n$ for an axis curve or $n-1$ for its derivative basis). Each basis polynomial is defined as:

        $$ B_{i, d}(t) = \binom{d}{i} t^i (1-t)^{d-i} $$

        The polynomial's value $p(t)$ is computed as:

        $$ p(t) = \sum_{i=0}^{d} w_i B_{i, d}(t) $$

        where $w_i$ are the weights (betas for an axis curve or deltas for its derivative basis).
        When using deltas, callers multiply the result by the axis degree to obtain the derivative.

        Args:
            t: Normalized input values in [0, 1].
                Expected shape: (*sample_shape, *batch_shape).
            weights: The coefficients for the basis polynomials.
                Expected shape: (*batch_shape, d + 1).
            binom_coeffs: Precomputed binomial coefficients corresponding to the polynomial's degree.
                Expected shape: (d + 1,).

        Returns:
            The evaluated polynomial values.
            Output shape: (*sample_shape, *batch_shape)
        """
        # Get n which can be != self.degree as we use this method for both CDF and PDF which have different degrees.
        nun_coeffs = binom_coeffs.shape[0]

        # Create a tensor of indices matching the number of basis polynomials.
        i = torch.arange(nun_coeffs, device=t.device, dtype=t.dtype)

        # Add an empty dimension to t for broadcasting, resulting in shape: (*sample_shape, *batch_shape, 1).
        t_expanded = t.unsqueeze(-1)

        # Compute the entire basis in one shot.
        # PyTorch broadcasts the shapes to shape (*sample_shape, *batch_shape, degree).
        basis = binom_coeffs * (t_expanded**i) * ((1 - t_expanded) ** (nun_coeffs - 1 - i))

        # Multiply by weights and sum across the final dimension, resulting in shape (*sample_shape, *batch_shape).
        return torch.sum(weights * basis, dim=-1)

    def _eval_log_derivative(
        self, t: torch.Tensor, log_deltas: torch.Tensor, binom_coeffs: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluate the log derivative of either normalized axis entirely in log-space.

        For an axis of degree $n$, its derivative uses the degree-$(n-1)$ Bernstein basis:

        $$ \log B'(t) = \log n + \operatorname{logsumexp}_i\!\left(
        \log \Delta_i + \log \binom{n-1}{i} + i\log t + (n-1-i)\log(1-t)\right). $$

        This avoids the numerically problematic log(polynomial + eps) path. Exact endpoints are handled
        separately, where only the first or last Bernstein basis term survives.

        Args:
            t: Curve parameters in [0, 1], of shape (*sample_shape, *batch_shape).
            log_deltas: Normalized log increments for the chosen axis, of shape (*batch_shape, n).
            binom_coeffs: Degree-(n-1) Bernstein coefficients, of shape (n,).

        Returns:
            Log derivatives with the broadcast sample/batch shape, before support-range scaling.
        """
        degree = log_deltas.shape[-1]
        # Indices for the Bernstein basis: i = 0, ..., n-1. Shape: (n,).
        indices = torch.arange(degree, device=t.device, dtype=t.dtype)
        # Clamp away from exact 0/1 to avoid log(0), then append a basis dimension: (*t.shape, 1).
        t_safe = t.clamp(torch.finfo(t.dtype).tiny, 1 - torch.finfo(t.dtype).eps).unsqueeze(-1)
        # Log of each Bernstein basis term: log(binom) + i*log(t) + (n-1-i)*log(1-t).
        # log_basis shape: (*t.shape, n); singleton batch dimensions expand when adding log_deltas.
        log_basis = binom_coeffs.log() + indices * t_safe.log() + (degree - 1 - indices) * torch.log1p(-t_safe)
        # Broadcast log_deltas (*batch_shape, n) across samples, then sum weighted terms via logsumexp.
        # The degree factor accounts for differentiation; result shape: (*sample_shape, *batch_shape).
        result = math.log(degree) + torch.logsumexp(log_deltas + log_basis, dim=-1)
        # Restore analytic endpoint derivatives instead of returning the clamped interior approximations.
        result = torch.where(t == 0, math.log(degree) + log_deltas[..., 0], result)
        return torch.where(t == 1, math.log(degree) + log_deltas[..., -1], result)

    def _invert_bezier(
        self,
        value: torch.Tensor,
        betas: torch.Tensor,
        deltas: torch.Tensor,
        coeffs_cdf: torch.Tensor,
        coeffs_pdf: torch.Tensor,
        num_iter: int = 64,
        use_newton: bool = True,
        newton_damping: float = 1.0,
        convergence_eps_factor: float = 2.0,
    ) -> torch.Tensor:
        r"""Invert a normalized monotone Bezier axis using a bracketed solver and implicit gradients.

        For the chosen horizontal or vertical axis of degree $n$, solve for the curve parameter $t$:

        $$ B(t) = \sum_{i=0}^{n} \beta_i B_{i,n}(t) = q, \qquad
        q = \operatorname{clamp}(\text{value}, 0, 1). $$

        Here $B_{i,n}$ is the Bernstein basis, and the supplied control points must increase from 0 to 1.
        Their positive increments determine the derivative:

        $$ B'(t) = n \sum_{i=0}^{n-1} \Delta_i B_{i,n-1}(t), \qquad
        \Delta_i = \beta_{i+1} - \beta_i. $$

        Each sample/batch element starts at $t=q$ with its own bracket $[0, 1]$. The current curve value
        tightens one bracket endpoint. A damped Newton proposal is accepted only strictly inside the updated
        bracket; otherwise its midpoint is used. Setting use_newton=False always selects the midpoint.

        $$ c_k = \frac{B(t_k)-q}{\max(B'(t_k), \text{tiny})}, \qquad
        t_{\mathrm{Newton}} = t_k - \alpha c_k. $$

        An element stops updating when either the undamped correction $|c_k|$ or its bracket width is at
        most convergence_eps_factor times machine epsilon. Both criteria measure error in parameter space,
        rather than relying on a small curve-value residual in a flat region. The loop ends when all elements
        converge or num_iter is reached; exhausting the iteration budget returns the current estimate.

        Iterations run without gradient tracking. At the resulting estimate, a forward-zero residual
        correction attaches the implicit first-order derivatives for interior targets:

        $$ \frac{\partial t}{\partial q} = \frac{1}{B'(t)}, \qquad
        \frac{\partial t}{\partial\theta} = -\frac{\partial_\theta B(t;\theta)}{B'(t)}. $$

        Here $\theta$ denotes the chosen axis's parameters. Detaching the denominator avoids differentiating
        this correction as another solver iteration. Gradient accuracy depends on the numerical root;
        higher-order inverse derivatives are not supported. Derivative denominators are clamped to the dtype's
        smallest positive normal value (tiny) to avoid division by zero, modifying the formulas below that floor.

        Args:
            value: Target values on the normalized axis, clamped to [0, 1]. Shape:
                (*sample_shape, *value_batch_shape), with trailing batch dimensions broadcastable to batch_shape.
            betas: Increasing control points with first/last values 0/1. Shape: (*batch_shape, n + 1).
            deltas: Positive forward differences of betas, summing to one. Shape: (*batch_shape, n).
            coeffs_cdf: Binomial coefficients for the degree-n curve. Shape: (n + 1,).
            coeffs_pdf: Binomial coefficients for the degree-(n-1) derivative basis. Shape: (n,).
                Despite their names, both coefficient tensors apply to either axis.
            num_iter: Maximum number of solver iterations; must be at least one.
            use_newton: Whether to try damped Newton steps with bisection fallback. False uses pure bisection,
                but still evaluates the derivative for the convergence check.
            newton_damping: Newton step multiplier $\alpha$ in (0, 1]; ignored when use_newton=False.
            convergence_eps_factor: Positive multiplier of the input dtype's machine epsilon for the tolerance.

        Returns:
            Estimated curve parameters in [0, 1], with shape (*sample_shape, *broadcast_batch_shape).
            The exact shape is the broadcast of value.shape and betas.shape[:-1]. Clamped target values 0 and 1
            return exact endpoints; a degree-one normalized axis returns the clamped targets directly.

        Note:
            Callers supply consistent axis tensors with a shared floating dtype/device and valid solver settings.
            This helper does not rescale physical coordinates or validate these preconditions.
        """
        # Remove the control-point dimension before broadcasting targets across the axis batch dimensions.
        # q and all per-element solver tensors have shape (*sample_shape, *broadcast_batch_shape).
        q, _ = torch.broadcast_tensors(value.clamp(0, 1), betas[..., 0])

        # With control points [0, 1], the degree-one axis and its inverse are both the identity.
        degree = deltas.shape[-1]
        if degree == 1:
            return q

        tiny = torch.finfo(q.dtype).tiny
        tolerance = convergence_eps_factor * torch.finfo(q.dtype).eps

        # Find the numerical root without retaining an autograd graph for the iterations.
        with torch.no_grad():
            mid = q.clone()
            low, high = torch.zeros_like(mid), torch.ones_like(mid)

            for _ in range(num_iter):
                current = self._eval_bezier_curve(mid, betas, coeffs_cdf)
                derivative = degree * self._eval_bezier_curve(mid, deltas, coeffs_pdf)
                correction = (current - q) / derivative.clamp_min(tiny)

                # The Boolean mask has the same sample/batch shape; only the all() check reduces it to a scalar.
                converged = (correction.abs() <= tolerance) | (high - low <= tolerance)
                if converged.all():
                    break

                # Monotonicity selects which bracket endpoint to move; bisection supplies the fallback step.
                low = torch.where(current < q, mid, low)
                high = torch.where(current >= q, mid, high)
                candidate = (low + high) / 2

                if use_newton:
                    # Reject overshooting Newton steps instead of letting them leave the root's bracket.
                    newton = mid - newton_damping * correction
                    candidate = torch.where((newton > low) & (newton < high), newton, candidate)

                # Keep converged estimates fixed while other sample/batch elements continue iterating.
                mid = torch.where(converged, mid, candidate)

        if torch.is_grad_enabled():
            # residual - residual.detach() is forward-zero but preserves gradients to q and the axis parameters.
            # Dividing by the detached slope attaches the implicit inverse derivative at the fixed estimate.
            residual = q - self._eval_bezier_curve(mid, betas, coeffs_cdf)
            derivative = degree * self._eval_bezier_curve(mid, deltas, coeffs_pdf)
            mid = mid + (residual - residual.detach()) / derivative.detach().clamp_min(tiny)

        # Return exact endpoints rather than tolerance-dependent interior approximations.
        mid = torch.where(q == 0, torch.zeros_like(mid), mid)
        return torch.where(q == 1, torch.ones_like(mid), mid)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute cumulative distribution function at given values.

        Args:
            value: Values at which to compute the CDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            CDF values in [0, 1] corresponding to the input values.
            Output shape: same as `value` argument.
        """
        x = value.to(device=self.logits_y.device, dtype=self.logits_y.dtype)

        # Map X in [bound_low, bound_up] to T in [0, 1].
        t = self._map_to_t_space(x)

        # Construct and evaluate the Bezier curve in T space.
        return self._eval_bezier_curve(t, weights=self._betas, binom_coeffs=self._binom_coeffs_cdf)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute probability density at given values.

        Args:
            value: Values at which to compute the PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            PDF values corresponding to the input values.
            Output shape: same as `value` argument.
        """
        return self.log_prob(value).exp()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""Compute the log-probability density at given values, entirely in log-space for numerical stability.

        Uses $\log p(x(t)) = \log B_y'(t) - \log B_x'(t) - \log(U-L)$.
        Both derivatives are evaluated with logsumexp. With fixed horizontal spacing, $B_x'(t)=1$.

        Args:
            value: Values at which to compute the log-PDF. Expected shape: (*sample_shape, *batch_shape).

        Returns:
            Log-PDF values corresponding to the input values.
            Output shape: same as `value` argument.
        """
        x = value.to(device=self.logits_y.device, dtype=self.logits_y.dtype)
        t = self._map_to_t_space(x)
        log_pdf = self._eval_log_derivative(t, self._log_deltas, self._binom_coeffs_pdf)
        log_pdf = log_pdf - self._eval_log_derivative(t, self._x_log_deltas, self._x_binom_coeffs_pdf)
        log_pdf = log_pdf - math.log(self.support_range)

        # Mask values outside the support.
        mask = (value >= self.bound_low) & (value <= self.bound_up)
        return torch.where(mask, log_pdf, torch.full_like(log_pdf, -math.inf))

    def entropy(self, num_quadrature_points: int = 251) -> torch.Tensor:
        r"""Compute differential entropy via Gaussian quadrature in parameter space, without inversion.

        $$ H(X) = -\int_0^1 B_y'(t) \log p(x(t)) \, dt $$

        where $L$ and $U$ are the lower and upper bounds of the distribution support, respectively.

        Args:
            num_quadrature_points: Number of Gauss-Legendre nodes for the approximation.

        Returns:
            Tensor of shape (*batch_shape,).
        """
        if num_quadrature_points < 1:
            raise ValueError("num_quadrature_points must be positive.")
        nodes, weights = self._quadrature(num_quadrature_points)
        log_derivative_y = self._eval_log_derivative(nodes, self._log_deltas, self._binom_coeffs_pdf)
        log_pdf = log_derivative_y - math.log(self.support_range)
        log_pdf = log_pdf - self._eval_log_derivative(nodes, self._x_log_deltas, self._x_binom_coeffs_pdf)
        return -(weights * log_derivative_y.exp() * log_pdf).sum(dim=0)

    def icdf(
        self,
        value: torch.Tensor,
        num_iter: int = 64,
        use_newton: bool = True,
        newton_damping: float = 1.0,
        convergence_eps_factor: float = 2.0,
    ) -> torch.Tensor:
        r"""Compute the inverse CDF, i.e., the quantile function, at the given values.

        Both solvers invert $B_y(t)-q=0$ and then evaluate $x(t)$, without a nested horizontal solve:

        **Newton's method** uses the vertical derivative and iterates

        $$ t_{k+1} = t_k - \alpha \frac{B_y(t_k) - q}{B_y'(t_k)} $$

        where $q$ is the target quantile in [0, 1],
        and $\alpha \in (0, 1]$ is a damping factor that shrinks each Newton step to improve robustness.
        A bracket within [0, 1] is maintained alongside: whenever $B_y(t_k) < q$ the lower bound tightens,
        otherwise the upper bound tightens. If the Newton step would leave the bracket, a bisection
        step is used instead, guaranteeing monotonic bracket contraction and preventing oscillation.
        The loop exits when the parameter-space Newton correction or bracket width is below tolerance.

        **Bisection** halves the search interval each iteration, gaining ~1 bit of precision per step.

        Args:
            value: Values in [0, 1] at which to compute the inverse CDF. Expected shape: (*sample_shape, *batch_shape).
            num_iter: Maximum number of solver iterations. The loop exits early upon convergence.
            use_newton: If True, use Newton's method. If False, use pure bisection.
            newton_damping: Damping factor in (0, 1] applied to the Newton step. A value of 1.0 gives the
                full Newton step (quadratic convergence), while smaller values improve robustness
                at the cost of slower convergence.
            convergence_eps_factor: The factor multiplied by machine epsilon to determine the convergence criterion.

        Returns:
            Quantiles in [bound_low, bound_up] corresponding to the input CDF values.
            Output shape: same as `value` argument.
        """
        if num_iter < 1 or not 0 < newton_damping <= 1 or convergence_eps_factor <= 0:
            raise ValueError("Require num_iter >= 1, 0 < newton_damping <= 1, and convergence_eps_factor > 0.")
        q = value.to(device=self.logits_y.device, dtype=self.logits_y.dtype)
        t = self._invert_bezier(
            q,
            self._betas,
            self._deltas,
            self._binom_coeffs_cdf,
            self._binom_coeffs_pdf,
            num_iter,
            use_newton,
            newton_damping,
            convergence_eps_factor,
        )
        return self._map_to_x_space(t)

    def rsample(self, sample_shape: torch.Size | list[int] | tuple[int, ...] = _size) -> torch.Tensor:
        """Draw reparameterized samples with first-order gradients through both axes.

        Args:
            sample_shape: Desired shape of the samples to be drawn. Default is empty, which means one sample per batch element.

        Returns:
            Samples drawn from the distribution, with shape (*sample_shape, *batch_shape).
        """
        # Determine the final shape of the output tensor.
        shape = self._extended_shape(sample_shape)

        # Sample uniform noise, u ~ U(0, 1).
        u = torch.rand(shape, dtype=self.logits_y.dtype, device=self.logits_y.device)
        return self.icdf(u)
