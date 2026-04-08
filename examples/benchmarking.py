import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.benchmark as benchmark
from binned_cdf.binned_logit_cdf import PiecewiseLinearBinnedCDF


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


def report_results(results: dict[str, tuple[float, int]]) -> None:
    """Print the benchmarking results."""
    print(f"{'operation':<16} | {'median time (ms)':<16} | {'total memory (MB)':<16}")
    print("-" * 55)
    for op, (median_time, total_mem) in results.items():
        time_ms = median_time * 1000
        mem_mb = total_mem / (1024**2)
        print(f"{op:<16} | {time_ms:<16.4f} | {mem_mb:<16.4f}")


def benchmark_shape(distr_class: type, shape: tuple[int, ...], num_bins: int = 32, num_iter: int = 100) -> None:
    """Benchmarks several function of the PiecewiseConstantBinnedCDF (sub)class for a specific logit shape.

    This function measures the average time and memory usage.

    Args:
        distr_class: The distribution class to benchmark: PiecewiseConstantBinnedCDF or PiecewiseLinearBinnedCDF.
        shape: A tuple of integers defining the shape of the input logits tensor.
        num_bins: The number of bins to use for the CDF.
        num_iter: The number of iterations to run for estimating the timing results.
    """
    print(f"\n===== Benchmarking Shape {shape} =====")

    # Generate prerequisites.
    logits = torch.randn(*shape)
    y = torch.randn(shape[:-1])
    quantiles = torch.rand_like(y)
    distr = PiecewiseLinearBinnedCDF(logits)  # instantiate once for forward testing

    # Run benchmarks.
    results = {}
    results["__init__"] = measure_performance(PiecewiseLinearBinnedCDF, logits, num_iter_measure=num_iter)
    results["prob"] = measure_performance(distr.prob, y, num_iter_measure=num_iter)
    results["cdf"] = measure_performance(distr.cdf, y, num_iter_measure=num_iter)
    results["icdf"] = measure_performance(distr.icdf, quantiles, num_iter_measure=num_iter)
    results["sample"] = measure_performance(distr.sample, num_iter_measure=num_iter)

    # Report results.
    report_results(results)


if __name__ == "__main__":
    """Execute the benchmark."""
    # Force CPU execution
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    print("Running benchmarks on CPU with float32.")

    # Define use cases (batch Size, sequence length, etc.)
    test_shapes = [
        (1, 128),  # single vector
        (10, 128),  # batch vector
        (1, 512, 512),  # single image-like
        (10, 512, 512),  # batch image-like
    ]
    distr_class = PiecewiseLinearBinnedCDF

    for shape in test_shapes:
        try:
            benchmark_shape(distr_class, shape)
        except Exception as e:
            print(f"Failed to benchmark shape {shape}: {e}\n")
