"""
Benchmark and visualization tools for Fibonacci algorithms.

This module compares two main implementations from FastFib:
- fib(n):  self-developed method using a global cache (memoized fast-doubling style).
- fast_fibonacci(n) / fibonacci_fast_doubling(n): standard fast-doubling O(log n).

It also includes iterative and naive versions for reference and produces
well-organized matplotlib figures to illustrate performance and growth.
"""

import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from FastFib import fast_fibonacci, fib, cache


def naive_fibonacci(n: int) -> int:
    """
    Naive recursive Fibonacci implementation for comparison.
    O(2^n) time complexity - very inefficient for large n.
    """
    if n <= 1:
        return n
    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2)


def iterative_fibonacci(n: int) -> int:
    """
    Iterative Fibonacci implementation for comparison.
    O(n) time complexity.
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def measure_execution_time(func, n: int, num_runs: int = 3) -> float:
    """
    Measure the average execution time of a function.
    
    Args:
        func: The function to measure
        n: Input value
        num_runs: Number of runs to average
    
    Returns:
        Average execution time in seconds
    """
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(n)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / len(times)


def benchmark_fast_fibonacci(max_n: int = 1000, step: int = 50) -> Tuple[List[int], List[float]]:
    """
    Benchmark the fast Fibonacci algorithm across different input sizes.
    
    Args:
        max_n: Maximum Fibonacci number index to test
        step: Step size between test points
    
    Returns:
        Tuple of (n_values, execution_times)
    """
    n_values = []
    execution_times = []
    
    print("Benchmarking Fast Fibonacci Algorithm...")
    for n in range(0, max_n + 1, step):
        try:
            elapsed = measure_execution_time(fast_fibonacci, n)
            n_values.append(n)
            execution_times.append(elapsed)
            print(f"F({n}): {elapsed*1000:.4f} ms")
        except Exception as e:
            print(f"Error at n={n}: {e}")
            break
    
    return n_values, execution_times


def benchmark_two_methods(max_n: int = 1000, step: int = 50) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark both fast_fibonacci and the cached fib across different input sizes.

    Args:
        max_n: Maximum Fibonacci number index to test
        step: Step size between test points

    Returns:
        Tuple of (n_values, fast_times, cached_times)
    """
    n_values: List[int] = []
    fast_times: List[float] = []
    cached_times: List[float] = []

    print("Benchmarking Fast vs Cached Fibonacci...")
    for n in range(0, max_n + 1, step):
        try:
            fast_time = measure_execution_time(fast_fibonacci, n)
            cached_time = measure_execution_time(fib, n)
            n_values.append(n)
            fast_times.append(fast_time)
            cached_times.append(cached_time)
            print(f"F({n}): fast={fast_time*1000:.4f} ms, cached={cached_time*1000:.4f} ms")
        except Exception as e:
            print(f"Error at n={n}: {e}")
            break

    return n_values, fast_times, cached_times


def compare_algorithms(max_n: int = 35) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Compare fast Fibonacci with iterative and naive implementations.
    
    Args:
        max_n: Maximum n to test (limited for naive method)
    
    Returns:
        Tuple of (n_values, fast_times, iterative_times, naive_times)
    """
    n_values: List[int] = []
    fast_times: List[float] = []
    cached_times: List[float] = []
    iterative_times: List[float] = []
    naive_times: List[float] = []
    
    print("\nComparing Algorithms...")
    for n in range(0, max_n + 1, 5):
        n_values.append(n)
        
        # Fast O(log n) method
        fast_time = measure_execution_time(fast_fibonacci, n)
        fast_times.append(fast_time)

        # Your original cached fib
        cached_time = measure_execution_time(fib, n)
        cached_times.append(cached_time)
        
        # Iterative method
        iter_time = measure_execution_time(iterative_fibonacci, n)
        iterative_times.append(iter_time)
        
        # Naive method (only for small n)
        if n <= 35:
            naive_time = measure_execution_time(naive_fibonacci, n, num_runs=1)
            naive_times.append(naive_time)
        else:
            naive_times.append(float('inf'))
        
        print(
            f"n={n}: "
            f"Fast={fast_time*1000:.4f}ms, "
            f"Cached={cached_time*1000:.4f}ms, "
            f"Iterative={iter_time*1000:.4f}ms, "
            f"Naive={'N/A' if n > 35 else f'{naive_time*1000:.4f}ms'}"
        )

    return n_values, fast_times, cached_times, iterative_times, naive_times


def analyze_large_numbers() -> Tuple[List[int], List[float], List[float], List[int]]:
    """
    Analyze performance on very large Fibonacci numbers.
    
    Returns:
        Tuple of (n_values, fast_times, self_times, cache_sizes)
    """
    n_values: List[int] = []
    fast_times: List[float] = []
    self_times: List[float] = []
    cache_sizes: List[int] = []
    
    # Test points: powers of 2 and some large values
    test_points = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    print("\nAnalyzing Large Fibonacci Numbers...")
    # Start with a clean cache so we can see how it grows as n increases
    cache.clear()
    for n in test_points:
        try:
            # Fast-doubling timing
            start_fast = time.perf_counter()
            _ = fast_fibonacci(n)
            elapsed_fast = time.perf_counter() - start_fast

            # Self-developed cached timing (reuses and grows global cache)
            start_self = time.perf_counter()
            _ = fib(n)
            elapsed_self = time.perf_counter() - start_self
            
            n_values.append(n)
            fast_times.append(elapsed_fast)
            self_times.append(elapsed_self)
            cache_sizes.append(len(cache))
            
            print(
                f"F({n}): fast={elapsed_fast*1000:.4f} ms, "
                f"self-developed={elapsed_self*1000:.4f} ms, "
                f"cache size={len(cache)}"
            )
        except Exception as e:
            print(f"Error at n={n}: {e}")
            break
    
    return n_values, fast_times, self_times, cache_sizes


def create_visualizations():
    """Create organized visualizations of Fibonacci algorithm performance."""

    # Use a pleasant default style
    plt.style.use("seaborn-v0_8-darkgrid")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Fibonacci Performance Comparison – Self-developed vs Fast-doubling",
        fontsize=16,
        fontweight="bold",
    )

    # Color mapping consistent across subplots
    color_fast = "#56CCF2"   # cyan/blue
    color_self = "#FF6B81"   # coral
    color_iter = "#6FCF97"   # green
    color_naive = "#F2C94C"  # yellow

    # 1. Execution Time vs Input Size (Linear) – two main methods
    ax1 = plt.subplot(2, 3, 1)
    n_vals, fast_times, cached_times = benchmark_two_methods(max_n=2000, step=100)
    ax1.plot(
        n_vals,
        [t * 1000 for t in fast_times],
        "-o",
        color=color_fast,
        markersize=4,
        linewidth=2,
        label="Fast‑doubling (O(log n))",
    )
    ax1.plot(
        n_vals,
        [t * 1000 for t in cached_times],
        "-s",
        color=color_self,
        markersize=4,
        linewidth=2,
        label="Self‑developed (cached)",
    )
    ax1.set_xlabel("Fibonacci index n", fontsize=10)
    ax1.set_ylabel("Execution time (ms)", fontsize=10)
    ax1.set_title("Linear scale – main methods", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # 2. Execution Time vs Input Size (Log Y) – two main methods
    ax2 = plt.subplot(2, 3, 2)
    n_vals_log, fast_times_log, cached_times_log = benchmark_two_methods(max_n=5000, step=200)
    ax2.semilogy(
        n_vals_log,
        [t * 1000 for t in fast_times_log],
        "-o",
        color=color_fast,
        markersize=4,
        linewidth=2,
        label="Fast‑doubling (O(log n))",
    )
    ax2.semilogy(
        n_vals_log,
        [t * 1000 for t in cached_times_log],
        "-s",
        color=color_self,
        markersize=4,
        linewidth=2,
        label="Self‑developed (cached)",
    )
    ax2.set_xlabel("Fibonacci index n", fontsize=10)
    ax2.set_ylabel("Execution time (ms, log scale)", fontsize=10)
    ax2.set_title("Log scale – main methods", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    # 3. Algorithm comparison – four methods on a small n range
    ax3 = plt.subplot(2, 3, 3)
    n_comp, fast_t, cached_t, iter_t, naive_t = compare_algorithms(max_n=35)
    ax3.plot(
        n_comp,
        [t * 1000 for t in fast_t],
        "-o",
        color=color_fast,
        linewidth=2,
        markersize=5,
        label="Fast‑doubling (O(log n))",
    )
    ax3.plot(
        n_comp,
        [t * 1000 for t in cached_t],
        "-D",
        color=color_self,
        linewidth=2,
        markersize=5,
        label="Self‑developed (cached)",
    )
    ax3.plot(
        n_comp,
        [t * 1000 for t in iter_t],
        "-s",
        color=color_iter,
        linewidth=2,
        markersize=5,
        label="Iterative (O(n))",
    )
    ax3.plot(
        n_comp,
        [t * 1000 if t != float("inf") else 0 for t in naive_t],
        "-^",
        color=color_naive,
        linewidth=2,
        markersize=5,
        label="Naive recursive (O(2^n))",
    )
    ax3.set_xlabel("Fibonacci index n", fontsize=10)
    ax3.set_ylabel("Execution time (ms)", fontsize=10)
    ax3.set_title("Algorithm comparison (small n)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Performance on larger n for both main methods
    ax4 = plt.subplot(2, 3, 4)
    n_large, times_fast_large, times_self_large, cache_sizes = analyze_large_numbers()
    ax4.plot(
        n_large,
        [t * 1000 for t in times_fast_large],
        "-o",
        color=color_fast,
        linewidth=2,
        markersize=6,
    )
    ax4.plot(
        n_large,
        [t * 1000 for t in times_self_large],
        "-s",
        color=color_self,
        linewidth=2,
        markersize=6,
    )
    ax4.set_xlabel("Fibonacci index n", fontsize=10)
    ax4.set_ylabel("Execution time (ms)", fontsize=10)
    ax4.set_title("Large n – self-developed vs fast‑doubling", fontsize=12, fontweight="bold")

    # 5. Cache size growth (number of cached Fibonacci values)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(
        n_large,
        cache_sizes,
        "-s",
        color=color_self,
        linewidth=2,
        markersize=6,
    )
    ax5.set_xlabel("Fibonacci index n", fontsize=10)
    ax5.set_ylabel("Total cached Fibonacci values", fontsize=10)
    ax5.set_title("Growth of cache size (self-developed)", fontsize=12, fontweight="bold")

    # 6. Time complexity (log-log) – both main methods vs O(log n) reference
    ax6 = plt.subplot(2, 3, 6)
    n_log, times_fast_log, times_self_log = benchmark_two_methods(max_n=10000, step=500)
    n_filtered = [n for n, t in zip(n_log, times_fast_log) if t > 0]
    fast_filtered = [t for t in times_fast_log if t > 0]
    self_filtered = [t for t in times_self_log if t > 0]

    ax6.loglog(
        n_filtered,
        [t * 1000 for t in fast_filtered],
        "-o",
        color=color_fast,
        linewidth=2,
        markersize=4,
        label="Fast‑doubling measured",
    )

    if self_filtered:
        ax6.loglog(
            n_filtered[: len(self_filtered)],
            [t * 1000 for t in self_filtered],
            "-s",
            color=color_self,
            linewidth=2,
            markersize=4,
            label="Self‑developed measured",
    )

    if len(n_filtered) > 1:
        ref_n = np.array(n_filtered)
        ref_time = fast_filtered[0] * 1000 * (np.log2(ref_n) / np.log2(ref_n[0]))
        ax6.loglog(
            ref_n,
            ref_time,
            "--",
            color="white",
            alpha=0.7,
            linewidth=1.5,
            label="O(log n) reference",
        )

    ax6.set_xlabel("Fibonacci index n (log scale)", fontsize=10)
    ax6.set_ylabel("Execution time (ms, log scale)", fontsize=10)
    ax6.set_title("Time complexity (log-log) – main methods", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fastfibonacci_performance_analysis.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualizations saved to 'fastfibonacci_performance_analysis.png'")
    plt.show()


def print_performance_summary():
    """Print a summary of performance metrics."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Test various sizes
    test_sizes = [100, 500, 1000, 5000, 10000]
    
    print("\nExecution Times:")
    print(f"{'n':<10} {'Time (ms)':<15} {'Result Digits':<15}")
    print("-" * 40)
    
    for n in test_sizes:
        try:
            start = time.perf_counter()
            result = fast_fibonacci(n)
            elapsed = (time.perf_counter() - start) * 1000
            digits = len(str(result))
            print(f"{n:<10} {elapsed:<15.4f} {digits:<15}")
        except Exception as e:
            print(f"{n:<10} Error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("="*60)
    print("Fast Fibonacci Algorithm Benchmark & Visualization")
    print("="*60)
    
    # Print performance summary
    print_performance_summary()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations()
    
    print("\n✓ Benchmark complete!")

