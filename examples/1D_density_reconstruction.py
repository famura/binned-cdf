import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from binned_cdf import BezierCDF


def fit_distribution(
    degree_y: int,
    degree_x: int,
    values: torch.Tensor,
    target_masses: torch.Tensor,
    num_steps: int,
    learning_rate: float,
) -> tuple[BezierCDF, list[float], dict[str, float]]:
    """Fit a Bezier distribution to quadrature-weighted target probabilities using expected NLL.

    Args:
        degree_y: Number of vertical logits.
        degree_x: Number of horizontal logits; one gives the fixed-axis baseline.
        values: Increasing quadrature nodes, including both support endpoints, of shape (num_points,).
        target_masses: Target PDF multiplied by integration weights at those nodes, of shape (num_points,).
        num_steps: Number of Adam updates.
        learning_rate: Adam learning rate, shared by all comparisons.

    Returns:
        Fitted unbatched distribution, NLL history, and timings: total_seconds, forward_ms, and backward_ms.
    """
    logits_y = torch.zeros(degree_y, device=values.device, dtype=values.dtype, requires_grad=True)
    # A single horizontal increment defines the fixed affine axis and is excluded from the learned budget.
    logits_x = torch.zeros(degree_x, device=values.device, dtype=values.dtype, requires_grad=degree_x > 1)
    parameters = [logits_y, logits_x] if logits_x.requires_grad else [logits_y]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    bound_low, bound_up = values[0].item(), values[-1].item()
    history = []
    forward_seconds = backward_seconds = 0.0
    started = perf_counter()
    for _ in range(num_steps):
        optimizer.zero_grad()
        tick = perf_counter()
        distr = BezierCDF(logits_y, logits_x, bound_low, bound_up)
        # Approximate E_target[-log p(X)] by integration, rather than drawing a noisy training sample.
        loss = -(target_masses * distr.log_prob(values)).sum()
        forward_seconds += perf_counter() - tick
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite NLL while fitting the reconstruction target.")
        tick = perf_counter()
        loss.backward()
        backward_seconds += perf_counter() - tick
        optimizer.step()
        history.append(loss.item())
    timings = {
        "total_seconds": perf_counter() - started,
        "forward_ms": 1000 * forward_seconds / num_steps,
        "backward_ms": 1000 * backward_seconds / num_steps,
    }
    distr = BezierCDF(logits_y.detach(), logits_x.detach(), bound_low, bound_up)
    return distr, history, timings


def main() -> None:
    """Compare fixed and learned horizontal spacing, then plot PDFs and generated samples.

    The target is a truncated Gaussian mixture with a narrow interior peak and a broad component.
    All models start uniform and receive the same optimizer settings and number of updates.
    The JSON sidecar records the configuration, fitted logits, evaluation scores, and CPU timings.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--grid-points", type=int, default=2049)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("examples/results/density_reconstruction.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.degree < 2 or args.degree % 2 or args.grid_points < 3 or args.learning_rate <= 0:
        parser.error("Require positive steps/rate, an even degree >= 2, and at least three grid points.")

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
        "Fixed x": (args.degree, 1),
        "Learned x, same degree": (args.degree, args.degree),
        "Learned x, same budget": (args.degree // 2, args.degree // 2),
    }
    fitted = {}
    results = {}
    best_label, best_nll = "", math.inf
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    axes[0].plot(evaluation_values.numpy(), target_pdf.numpy(), color="black", linestyle="--", label="Target", zorder=5)
    print(f"CPU, {values.dtype}, {args.steps} Adam steps, {args.grid_points} integration nodes", flush=True)
    print(f"Target NLL: {target_nll.item():.6f}", flush=True)
    for (label, (degree_y, degree_x)), color in zip(models.items(), colors, strict=True):
        trainable_logits = degree_y + (degree_x if degree_x > 1 else 0)
        print(
            f"Fitting {label}: {degree_y} vertical, {degree_x} horizontal logits ({trainable_logits} trainable)",
            flush=True,
        )
        distr, history, timings = fit_distribution(
            degree_y, degree_x, values, target_masses, args.steps, args.learning_rate
        )
        with torch.no_grad():
            log_pdf = distr.log_prob(evaluation_values)
            nll = -torch.trapezoid(target_pdf * log_pdf, evaluation_values).item()
            integral = torch.trapezoid(log_pdf.exp(), evaluation_values).item()
        fitted[label] = distr
        if degree_x > 1 and nll < best_nll:
            best_label, best_nll = label, nll
        results[label] = {
            "degree_y": degree_y,
            "degree_x": degree_x,
            "trainable_logits": trainable_logits,
            "nll": nll,
            "kl": nll - target_nll.item(),
            "pdf_integral": integral,
            **timings,
            "logits_y": distr.logits_y.tolist(),
            "logits_x": distr.logits_x.tolist(),
            "initial_nll": history[0],
            "final_training_nll": history[-1],
        }
        print(
            f"  NLL={nll:.6f}, KL={nll - target_nll.item():.6f}, integral={integral:.6f}, "
            f"forward={timings['forward_ms']:.2f} ms, backward={timings['backward_ms']:.2f} ms, "
            f"total={timings['total_seconds']:.1f} s",
            flush=True,
        )
        axes[0].plot(evaluation_values.numpy(), log_pdf.exp().numpy(), color=color, label=label)

    samples = fitted[best_label].sample((20_000,)).numpy()
    sample_color = colors[list(models).index(best_label)]
    axes[1].hist(
        samples, bins=140, range=(bound_low, bound_up), density=True, alpha=0.45, color=sample_color, label=best_label
    )
    axes[1].plot(evaluation_values.numpy(), target_pdf.numpy(), "k--", label="Target")
    for axis, title in zip(axes, ["Expected-NLL reconstruction", "Samples from the best two-axis fit"], strict=True):
        axis.set(xlabel="Value", ylabel="Density", title=title, xlim=(bound_low, bound_up))
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    report = {
        "steps": args.steps,
        "grid_points": args.grid_points,
        "learning_rate": args.learning_rate,
        "target_nll": target_nll.item(),
        "target": {"weights": [0.4, 0.6], "means": [-0.6, 0.6], "stds": [0.035, 0.45], "bounds": [bound_low, bound_up]},
        "results": results,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}", flush=True)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
