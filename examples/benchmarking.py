import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.benchmark as benchmark

from binned_cdf import BezierCDF, PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF

ResultsMap = dict[str, tuple[float, int]]
BenchmarkInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def make_benchmark_inputs(shape: tuple[int, ...], seed: int) -> BenchmarkInputs:
    """Create deterministic benchmark inputs for a given shape."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    logits = torch.randn(*shape, generator=gen)
    y = torch.randn(shape[:-1], generator=gen)
    quantiles = torch.rand(shape[:-1], generator=gen)
    return logits, y, quantiles


def _short_name(cls_or_name: type | str) -> str:
    name = cls_or_name if isinstance(cls_or_name, str) else cls_or_name.__name__
    return name.replace("Piecewise", "PW").replace("BinnedCDF", "")


def measure_performance(
    func: Callable[..., Any],
    *args: Any,
    num_iter_measure: int,
    num_iter_warmup: int = 5,
    **kwargs: Any,
) -> tuple[float, int]:
    """Measures median time and memory consumption for a given function.

    Notes:
        - Use `torch.profiler.profile` as `tracemalloc` only tracks Python objects, missing C++ tensor allocations.
        - `torch.utils.benchmark.Timer` is more accurate than `time.perf_counter` for PyTorch operations.

    Args:
        func: The function to benchmark.
        *args: Positional arguments to pass to the function.
        num_iter_measure: Number of iterations to run for measuring time.
        num_iter_warmup: Number of iterations to run for warming up.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        - median time in seconds
        - total memory in bytes
    """
    # Warmup.
    for _ in range(num_iter_warmup):
        _ = func(*args, **kwargs)

    # Measure memory.
    # We suppress the UserWarning about acc_events because we are doing one-off measurements.
    # The W301 CPUAllocator warning is a known artifact of profiling pre-allocated tensors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Profiler clears events.*")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            _ = func(*args, **kwargs)

    # Sum of positive self-allocations as a proxy for memory footprint.
    total_mem = sum(e.self_cpu_memory_usage for e in prof.key_averages() if e.self_cpu_memory_usage > 0)

    # Measure time.
    t = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"func": func, "args": args, "kwargs": kwargs},
    )
    median_time = t.timeit(num_iter_measure).median

    return median_time, total_mem


def benchmark_class(
    distr_class: type,
    logits: torch.Tensor,
    y: torch.Tensor,
    quantiles: torch.Tensor,
    num_iter: int,
) -> ResultsMap | None:
    """Benchmarks a single distribution class. Returns None if the class fails (e.g. degree overflow)."""
    try:
        distr = distr_class(logits)
        return {
            "__init__": measure_performance(distr_class, logits, num_iter_measure=num_iter),
            "prob": measure_performance(distr.prob, y, num_iter_measure=num_iter),
            "log_prob": measure_performance(distr.log_prob, y, num_iter_measure=num_iter),
            "cdf": measure_performance(distr.cdf, y, num_iter_measure=num_iter),
            "icdf": measure_performance(distr.icdf, quantiles, num_iter_measure=num_iter),
            "sample": measure_performance(distr.sample, num_iter_measure=num_iter),
        }
    except Exception as e:
        print(f"  Skipping {distr_class.__name__}: {e}")
        return None


def report_results(all_results: dict[str, ResultsMap | None]) -> None:
    """Print a comparison table with one column pair (time, memory) per class."""
    class_names = list(all_results.keys())
    operations = list(next(r for r in all_results.values() if r is not None).keys())

    col_op, col_val = 10, 13

    header = f"{'operation':<{col_op}}"
    for name in class_names:
        short = _short_name(name)
        header += f" | {short + ' ms':<{col_val}} | {short + ' MB':<{col_val}}"
    print(header)
    print("-" * len(header))

    for op in operations:
        row = f"{op:<{col_op}}"
        for name in class_names:
            result = all_results[name]
            if result is None:
                row += f" | {'N/A':<{col_val}} | {'N/A':<{col_val}}"
            else:
                time_ms, total_mem = result[op]
                row += f" | {time_ms * 1000:<{col_val}.4f} | {total_mem / (1024**2):<{col_val}.4f}"
        print(row)


def benchmark_shape(
    distr_classes: list[type],
    shape: tuple[int, ...],
    inputs: BenchmarkInputs,
    num_iter: int = 100,
) -> None:
    """Benchmarks all given distribution classes for a specific logit shape and prints a comparison table.

    Args:
        distr_classes: Distribution classes to benchmark.
        shape: Shape of the input logits tensor (*batch_shape, num_bins).
        inputs: Precomputed (logits, y, quantiles) tensors for this shape.
        num_iter: Number of iterations for timing measurements.
    """
    print(f"\n===== Benchmarking Shape {shape} =====")
    logits, y, quantiles = inputs

    all_results = {cls.__name__: benchmark_class(cls, logits, y, quantiles, num_iter) for cls in distr_classes}
    report_results(all_results)


if __name__ == "__main__":
    """Execute the benchmark."""
    DISTR_CLASSES: list[type] = [PiecewiseConstantBinnedCDF, PiecewiseLinearBinnedCDF, BezierCDF]

    # Force CPU execution.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    print("Running benchmarks on CPU with float32.")
    print("Using deterministic benchmark inputs.")

    # Define use cases (batch size, sequence length, etc.)
    test_shapes = [
        (1, 128),  # single vector
        (10, 128),  # batch vector
        (1, 4, 4),  # single mini image-like
        (10, 4, 4),  # batch mini image-like
        (1, 512, 512),  # single image-like
        (10, 512, 512),  # batch image-like
    ]

    base_seed = 20260410
    shape_inputs = {shape: make_benchmark_inputs(shape, seed=base_seed + idx) for idx, shape in enumerate(test_shapes)}

    for shape in test_shapes:
        benchmark_shape(DISTR_CLASSES, shape, inputs=shape_inputs[shape])
