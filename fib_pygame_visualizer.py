"""
Animated Pygame visualizer for Fibonacci performance.

This script compares:
- Self‑developed method:     FastFib.fib (global cache implementation)
- Fast‑doubling method:      FastFib.fibonacci_fast_doubling

It animates the growth of execution time as the Fibonacci index increases,
showing both a linear scale and a log scale in real time, up to n = 5_000_000.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import pygame

from FastFib import fib, fibonacci_fast_doubling


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 720
FPS = 30

BACKGROUND_COLOR = (10, 12, 24)
PLOT_BACKGROUND = (18, 22, 40)
GRID_COLOR = (60, 70, 95)

FAST_COLOR = (86, 204, 242)      # Fast‑doubling method – cyan/blue
SELF_COLOR = (255, 107, 129)     # Self‑developed cached method – coral

TEXT_COLOR = (235, 240, 250)

MAX_N = 6_000_000


def generate_n_values() -> List[int]:
    """Generate increasing Fibonacci indices up to MAX_N with larger intervals."""
    values: List[int] = []
    n = 1_000
    while n <= MAX_N:
        values.append(n)
        # Increase interval as n grows
        if n < 50_000:
            n += 5_000
        elif n < 500_000:
            n += 50_000
        elif n < 2_000_000:
            n += 200_000
        else:
            n += 500_000
    return values


@dataclass
class SamplePoint:
    n: int
    t_fast_ms: float
    t_self_ms: float


class FibPerformanceAnimator:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Fibonacci Performance – Self‑developed vs Fast‑doubling")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("consolas", 16)
        self.font_medium = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 26, bold=True)

        self.n_values = generate_n_values()
        self.samples: List[SamplePoint] = []
        self.current_index = 0

        self.running = True
        # mode: "linear" or "log"
        self.mode = "linear"
        # whether to show an info overlay with controls/mode
        self.show_info = False

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def _time_call(self, func, n: int) -> float:
        """Measure execution time of func(n) in milliseconds."""
        start = time.perf_counter()
        func(n)
        end = time.perf_counter()
        return (end - start) * 1000.0

    def compute_next_sample(self) -> None:
        """Compute timings for the next n value."""
        if self.current_index >= len(self.n_values):
            return

        n = self.n_values[self.current_index]
        try:
            t_fast = self._time_call(fibonacci_fast_doubling, n)
            t_self = self._time_call(fib, n)
        except Exception:
            # If anything goes wrong at very large n, stop further sampling.
            self.current_index = len(self.n_values)
            return

        self.samples.append(SamplePoint(n=n, t_fast_ms=t_fast, t_self_ms=t_self))
        self.current_index += 1

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_text(self, text: str, pos: Tuple[int, int], font, color=TEXT_COLOR) -> None:
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)

    def draw_axes_and_grid(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        margin_left = 90
        margin_right = 60
        margin_top = 90
        margin_bottom = WINDOW_HEIGHT - 80

        # Single main plot area
        pygame.draw.rect(
            self.screen,
            PLOT_BACKGROUND,
            (margin_left, margin_top, WINDOW_WIDTH - margin_left - margin_right, margin_bottom - margin_top),
            border_radius=14,
        )

        # Title
        self.draw_text(
            "Self‑developed vs Fast‑doubling Fibonacci (up to n = 5,000,000)",
            (margin_left, 24),
            self.font_large,
        )
        # Optional info overlay (toggled with 'm')
        if self.show_info:
            mode_label = "Linear scale (ms)" if self.mode == "linear" else "Log scale (log10 ms)"
            info_lines = [
                f"Current view: {mode_label}",
                "SPACE – toggle between linear / log view",
                "M – hide/show this help",
                "ESC – quit",
            ]
            panel_w = 420
            panel_h = 90
            panel_x = WINDOW_WIDTH - panel_w - 30
            panel_y = 20
            pygame.draw.rect(
                self.screen,
                (25, 30, 55),
                (panel_x, panel_y, panel_w, panel_h),
                border_radius=10,
            )
            pygame.draw.rect(
                self.screen,
                (80, 90, 140),
                (panel_x, panel_y, panel_w, panel_h),
                width=1,
                border_radius=10,
            )
            for i, line in enumerate(info_lines):
                self.draw_text(line, (panel_x + 12, panel_y + 8 + i * 18), self.font_small)

        # Legend
        legend_y = 60
        pygame.draw.rect(self.screen, SELF_COLOR, (margin_left + 520, legend_y, 24, 4))
        self.draw_text("Self‑developed method (cached)", (margin_left + 550, legend_y - 5), self.font_small)
        pygame.draw.rect(self.screen, FAST_COLOR, (margin_left + 840, legend_y, 24, 4))
        self.draw_text("Fast‑doubling method (O(log n))", (margin_left + 872, legend_y - 5), self.font_small)

        # Store layout for plotting
        self.layout = {
            "margin_left": margin_left,
            "margin_right": margin_right,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
        }

    def plot_samples(self) -> None:
        if not self.samples:
            return

        m = self.layout
        ml = m["margin_left"]
        mr = m["margin_right"]
        mt = m["margin_top"]
        mb = m["margin_bottom"]

        plot_width = WINDOW_WIDTH - ml - mr
        plot_height = mb - mt

        ns = [s.n for s in self.samples]
        t_fast = [s.t_fast_ms for s in self.samples]
        t_self = [s.t_self_ms for s in self.samples]

        # Dynamic x-axis: grow as max n grows, with a little padding
        max_n_current = max(ns) if ns else 1
        x_max = max_n_current * 1.05

        max_time_linear = max(t_fast + t_self) if (t_fast or t_self) else 1.0

        # For log scale, avoid zero
        min_time_nonzero = min([t for t in (t_fast + t_self) if t > 0.0], default=1e-6)

        # Draw grid lines (vertical based on n)
        num_x_ticks = 6
        for i in range(num_x_ticks):
            frac = i / (num_x_ticks - 1) if num_x_ticks > 1 else 0
            x = ml + int(frac * plot_width)
            pygame.draw.line(self.screen, GRID_COLOR, (x, mt + 8), (x, mb - 8), 1)
            n_label = int(x_max * frac)
            self.draw_text(f"{n_label:,}", (x - 30, mb + 8), self.font_small)

        # Horizontal grid with tick labels on Y-axis
        num_y_ticks = 5
        for i in range(num_y_ticks):
            frac = i / (num_y_ticks - 1) if num_y_ticks > 1 else 0
            y = mt + int(frac * plot_height)
            # Note: end_pos must be a (x, y) tuple
            pygame.draw.line(self.screen, GRID_COLOR, (ml + 8, y), (ml + plot_width - 8, y), 1)
            # Y tick label
            if self.mode == "linear":
                value = max_time_linear * (1.0 - frac)
                label = f"{value:.3f}"
            else:
                # Map frac back to log-space
                log_min = math.log10(min_time_nonzero)
                log_max = math.log10(max_time_linear if max_time_linear > 0 else 1.0)
                if log_max == log_min:
                    val_log = log_min
                else:
                    val_log = log_max - frac * (log_max - log_min)
                value = 10 ** val_log
                label = f"{value:.3e}"
            self.draw_text(label, (ml - 80, y - 8), self.font_small)

        # Axis labels (x: text, y: rotated text)
        self.draw_text("Fibonacci index (n)", (ml + plot_width // 2 - 80, mb + 32), self.font_small)

        # Rotated Y label
        y_label = "Execution time (ms)" if self.mode == "linear" else "Execution time (log10 ms)"
        y_surf = self.font_small.render(y_label, True, TEXT_COLOR)
        y_surf_rot = pygame.transform.rotate(y_surf, 90)
        self.screen.blit(y_surf_rot, (ml - 60, mt + plot_height // 2 - y_surf_rot.get_height() // 2))

        # Helper to convert data to screen coordinates
        def to_linear_point(n_val: int, t_ms: float) -> Tuple[int, int]:
            x = ml + int((n_val / x_max) * plot_width)
            y = mt + plot_height - int((t_ms / max_time_linear) * (plot_height - 30))
            return x, y

        def to_log_point(n_val: int, t_ms: float) -> Tuple[int, int]:
            x = ml + int((n_val / x_max) * plot_width)
            t = max(t_ms, min_time_nonzero)
            log_t = math.log10(t)
            # Normalize log_t into [0, 1] based on current range
            log_min = math.log10(min_time_nonzero)
            log_max = math.log10(max_time_linear if max_time_linear > 0 else 1.0)
            if log_max == log_min:
                frac = 0.0
            else:
                frac = (log_t - log_min) / (log_max - log_min)
            y = mt + plot_height - int(frac * (plot_height - 30))
            return x, y

        # Draw polylines for both methods (only one scale at a time)
        if len(self.samples) >= 2:
            if self.mode == "linear":
                fast_points = [to_linear_point(n, t) for n, t in zip(ns, t_fast)]
                self_points = [to_linear_point(n, t) for n, t in zip(ns, t_self)]
            else:
                fast_points = [to_log_point(n, t) for n, t in zip(ns, t_fast)]
                self_points = [to_log_point(n, t) for n, t in zip(ns, t_self)]

            pygame.draw.lines(self.screen, FAST_COLOR, False, fast_points, 3)
            pygame.draw.lines(self.screen, SELF_COLOR, False, self_points, 3)

            # Draw small markers for recent points
            for p in fast_points[-10:]:
                pygame.draw.circle(self.screen, FAST_COLOR, p, 3)
            for p in self_points[-10:]:
                pygame.draw.circle(self.screen, SELF_COLOR, p, 3)

        # Current status text
        latest = self.samples[-1]
        self.draw_text(
            f"Last n = {latest.n:,} | Self‑developed: {latest.t_self_ms:.4f} ms | Fast‑doubling: {latest.t_fast_ms:.4f} ms",
            (ml, mb + 40),
            self.font_small,
        )

        progress = len(self.samples) / len(self.n_values) if self.n_values else 1.0
        self.draw_text(
            f"Progress: {len(self.samples)}/{len(self.n_values)} points ({progress*100:.1f}%)",
            (WINDOW_WIDTH - 360, mb + 40),
            self.font_small,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        # Toggle between linear and log view
                        self.mode = "log" if self.mode == "linear" else "linear"
                    elif event.key == pygame.K_m:
                        # Toggle help overlay
                        self.show_info = not self.show_info

            # Compute next sample every few frames to keep UI responsive
            if self.current_index < len(self.n_values):
                self.compute_next_sample()

            self.draw_axes_and_grid()
            self.plot_samples()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    animator = FibPerformanceAnimator()
    animator.run()


