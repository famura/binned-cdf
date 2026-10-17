import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal
from tqdm import trange

from binned_cdf import BezierCDF


def fit_distribution(
    degree_y: int,
    degree_x: int,
    values: torch.Tensor,
    target_masses: torch.Tensor,
    num_iter: int,
    learning_rate: float,
    label: str,
) -> BezierCDF:
    """Fit a Bezier distribution to quadrature-weighted target probabilities using expected NLL.

    Args:
        degree_y: Number of vertical logits.
        degree_x: Number of horizontal logits; one gives the fixed-axis baseline.
        values: Increasing quadrature nodes, including both support endpoints, of shape (num_points,).
        target_masses: Target PDF multiplied by integration weights at those nodes, of shape (num_points,).
        num_iter: Number of Adam updates.
        learning_rate: Adam learning rate, shared by all comparisons.
        label: Model name displayed in the training progress bar.

    Returns:
        Fitted unbatched distribution.
    """
    # A single horizontal increment defines the fixed affine axis and is excluded from the learned budget.
    logits_x = torch.zeros(degree_x, device=values.device, dtype=values.dtype, requires_grad=degree_x > 1)
    logits_y = torch.zeros(degree_y, device=values.device, dtype=values.dtype, requires_grad=True)

    parameters = [logits_y, logits_x] if logits_x.requires_grad else [logits_y]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    bound_low, bound_up = values[0].item(), values[-1].item()

    progress = trange(num_iter, desc=f"Fitting {label}")
    for epoch in progress:
        optimizer.zero_grad()
        distr = BezierCDF(logits_y, logits_x, bound_low, bound_up)
        # Approximate E_target[-log p(X)] by integration, rather than drawing a noisy training sample.
        loss = -(target_masses * distr.log_prob(values)).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite NLL while fitting the reconstruction target.")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == num_iter:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    distr = BezierCDF(logits_y.detach(), logits_x.detach(), bound_low, bound_up)
    return distr


if __name__ == "__main__":
    """Compare fixed and learned horizontal spacing, then plot PDFs and generated samples.

    The target is a truncated Gaussian mixture with a narrow interior peak and a broad component.
    All models start uniform and receive the same optimizer settings and number of updates.
    """
    parser = argparse.ArgumentParser(
        description="Fit a 1D density using BezierCDF and compare with the target distribution."
    )
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--grid-points", type=int, default=2001)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("examples/results/density_reconstruction.png"))
    parser.add_argument("--show", action="store_true", default=True)
    args = parser.parse_args()
    if args.num_iter < 1 or args.degree < 2 or args.degree % 2 or args.grid_points < 3 or args.learning_rate <= 0:
        parser.error("Require positive iterations/rate, an even degree >= 2, and at least three grid points.")

    torch.set_num_threads(1)
    torch.manual_seed(42)
    bound_low, bound_up = -2.0, 2.0
    target = MixtureSameFamily(
        Categorical(torch.tensor([0.4, 0.6])),
        Normal(torch.tensor([-0.6, 0.6]), torch.tensor([0.035, 0.45])),
    )

    # Condition the Gaussian mixture on the bounded support used by every fitted distribution.
    normalizer = target.cdf(torch.tensor(bound_up)) - target.cdf(torch.tensor(bound_low))
    values = torch.linspace(bound_low, bound_up, args.grid_points)

    # Trapezoidal masses have shape (grid_points,); the two endpoint integration weights are halved.
    target_masses = target.log_prob(values).exp() / normalizer * (values[1] - values[0])
    target_masses[[0, -1]] *= 0.5

    # Evaluate on a finer grid that is not just the training nodes, to reveal unresolved narrow peaks.
    evaluation_values = torch.linspace(bound_low, bound_up, 4 * args.grid_points - 2)
    target_pdf = target.log_prob(evaluation_values).exp() / normalizer
    target_nll = -torch.trapezoid(
        target_pdf * (target.log_prob(evaluation_values) - normalizer.log()), evaluation_values
    )

    # Compare equal per-axis degree and equal total learned-logit count against the fixed-spacing baseline.
    models = {
        "fixed x": (args.degree, 1),
        "learned x, same degree": (args.degree, args.degree),
        "learned x, same budget": (args.degree // 2, args.degree // 2),
    }
    best_label, best_nll = "", math.inf
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), layout="constrained")
    axes[0].plot(evaluation_values.numpy(), target_pdf.numpy(), color="black", linestyle="--", label="Target", zorder=5)

    print(f"CPU, {values.dtype}, {args.num_iter} Adam iterations, {args.grid_points} integration nodes", flush=True)
    print(f"Target NLL: {target_nll.item():.6f}", flush=True)

    fitted = {}
    for (label, (degree_y, degree_x)), color in zip(models.items(), colors, strict=True):
        trainable_logits = degree_y + (degree_x if degree_x > 1 else 0)
        print(
            f"Fitting {label}: {degree_y} vertical, {degree_x} horizontal logits ({trainable_logits} trainable)",
            flush=True,
        )
        distr = fit_distribution(degree_y, degree_x, values, target_masses, args.num_iter, args.learning_rate, label)
        with torch.no_grad():
            log_pdf = distr.log_prob(evaluation_values)
            nll = -torch.trapezoid(target_pdf * log_pdf, evaluation_values).item()
            integral = torch.trapezoid(log_pdf.exp(), evaluation_values).item()
        fitted[label] = distr
        if degree_x > 1 and nll < best_nll:
            best_label, best_nll = label, nll
        print(f"  NLL={nll:.6f}, KL={nll - target_nll.item():.6f}, integral={integral:.6f}", flush=True)
        axes[0].plot(evaluation_values.numpy(), log_pdf.exp().numpy(), color=color, label=label)

    # Plot samples from the best two-axis fit.
    samples = fitted[best_label].sample((20_000,)).numpy()
    sample_color = colors[list(models).index(best_label)]
    axes[1].hist(
        samples, bins=140, range=(bound_low, bound_up), density=True, alpha=0.45, color=sample_color, label=best_label
    )
    axes[1].plot(evaluation_values.numpy(), target_pdf.numpy(), "k--", label="target")

    # Set axis labels, titles, legends, and grids.
    for axis, title in zip(axes, ["Expected-NLL reconstruction", "Samples from the best two-axis fit"], strict=True):
        axis.set(xlabel="value", ylabel="density", title=title, xlim=(bound_low, bound_up))
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)
