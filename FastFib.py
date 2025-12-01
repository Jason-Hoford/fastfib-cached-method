"""
Fast Fibonacci Number Calculator

This module contains two different Fibonacci implementations:

1. fib(n)  - your original implementation using a global cache / memoization
             (referred to as the *self‑developed method* in the visualizations).
2. fibonacci_fast_doubling(n) - the standard fast‑doubling O(log n) method.
"""

import sys

# Allow conversion of very large integers to strings
sys.set_int_max_str_digits(0)

# ---------------------------------------------------------------------------
# Your original cached Fibonacci implementation (self‑developed method)
# ---------------------------------------------------------------------------

cache = {}


def fib(n: int) -> int:
    """Self‑developed Fibonacci using a global cache (memoization).

    This is your original implementation, kept exactly in spirit but with
    type hints and a clearer docstring.
    """
    if n in cache:
        return cache[n]
    if n < 3:
        return 1

    k = n >> 1  # Faster than n // 2

    if n & 1:  # n is odd - faster than n % 2
        fib_k = fib(k)
        fib_k1 = fib(k + 1)
        result = fib_k * fib_k + fib_k1 * fib_k1
        cache[n] = result
        return result
    else:  # n is even
        fib_k1 = fib(k + 1)
        fib_k_1 = fib(k - 1)
        res = fib_k1 * fib_k1 - fib_k_1 * fib_k_1
        cache[n] = res
        return res


# ---------------------------------------------------------------------------
# Fast‑doubling O(log n) Fibonacci using pair-based recursion
# ---------------------------------------------------------------------------


def fibonacci_fast_doubling(n: int) -> int:
    """Calculate the nth Fibonacci number using the *fast‑doubling* algorithm.

    This is the standard fast‑doubling O(log n) method, implemented via a
    recursive pair function that returns (F(k), F(k+1)).

    Args:
        n: The index of the Fibonacci number to calculate (non-negative integer).

    Returns:
        The nth Fibonacci number.

    Examples:
        >>> fibonacci_fast_doubling(0)
        0
        >>> fibonacci_fast_doubling(1)
        1
        >>> fibonacci_fast_doubling(10)
        55
    """

    def _fibonacci_pair(k: int) -> tuple[int, int]:
        """Return the pair (F(k), F(k+1)) using fast‑doubling identities."""
        if k <= 1:
            return (k, 1)

        # Recursively compute pair for k//2
        fib_half, fib_half_plus_one = _fibonacci_pair(k >> 1)

        # Fast‑doubling identities:
        # F(2k)   = F(k) * (2*F(k+1) - F(k))
        # F(2k+1) = F(k+1)^2 + F(k)^2
        fib_k = fib_half * ((fib_half_plus_one << 1) - fib_half)
        fib_k_plus_one = fib_half_plus_one ** 2 + fib_half ** 2

        # If k is odd, adjust to get F(k) and F(k+1)
        if k & 1:
            fib_k, fib_k_plus_one = fib_k_plus_one, fib_k + fib_k_plus_one

        return (fib_k, fib_k_plus_one)

    return _fibonacci_pair(n)[0]


# Backwards‑compatible alias (old name used elsewhere)
fast_fibonacci = fibonacci_fast_doubling


# ---------------------------------------------------------------------------
# Example usage (for reference / manual testing)
# ---------------------------------------------------------------------------
#
# The code below is commented out so that importing this module never performs
# any work by default. You can uncomment and run this file directly to see
# example outputs from both implementations.
#
# if __name__ == "__main__":
#     # Small demonstration values
#     test_indices = [0, 1, 2, 5, 10, 20, 50]
# 
#     print("Self‑developed method (fib with global cache)")
#     for n in test_indices:
#         # Clear the cache between separate experiments if desired:
#         # cache.clear()
#         print(f"F({n}) = {fib(n)}")
# 
#     print("\nFast‑doubling method (fibonacci_fast_doubling)")
#     for n in test_indices:
#         print(f"F({n}) = {fibonacci_fast_doubling(n)}")
# 
#     # Example: very large n using fast‑doubling
#     big_n = 100_000
#     print(f"\nComputing F({big_n}) using fast‑doubling...")
#     result = fibonacci_fast_doubling(big_n)
#     print(f"F({big_n}) has {len(str(result))} digits")