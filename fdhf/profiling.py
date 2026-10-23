"""Runtime-cost instrumentation.

Measures the three quantities reported in the manuscript's computational-cost
discussion: wall-clock training time, inference latency, peak GPU memory, and
the trainable / frozen parameter split of the LLM branch.
"""

from __future__ import annotations

import contextlib
import time


class Stopwatch:
    """Accumulates named wall-clock durations."""

    def __init__(self):
        self.timings = {}

    def time(self, name: str):
        return _TimerCtx(self, name)

    def add(self, name: str, seconds: float):
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def as_dict(self) -> dict:
        return dict(self.timings)

    def total(self) -> float:
        return float(sum(self.timings.values()))


@contextlib.contextmanager
def _TimerCtx(sw: Stopwatch, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sw.add(name, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #
def count_parameters(model) -> dict:
    """Total / trainable / frozen parameter counts of a torch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


# --------------------------------------------------------------------------- #
# Peak GPU memory
# --------------------------------------------------------------------------- #
def reset_peak_memory():
    """Reset the CUDA peak-memory counters (no-op without CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_memory() -> dict:
    """Peak allocated / reserved GPU memory in bytes (0 without CUDA)."""
    out = {"allocated_bytes": 0, "reserved_bytes": 0}
    try:
        import torch
        if torch.cuda.is_available():
            out["allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            out["reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass
    return out


def device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"
