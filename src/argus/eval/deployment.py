"""Deployment measurement: throughput, latency percentiles, peak memory, model size.

See docs/08_EVALUATION.md protocol (7) "flow-timeout stability" / deployment
measurement and docs/TODO.md P8. Drives `streaming.detector.StreamingDetector`
in the same push()-per-flow contract used in production, so the measured
latency includes graph-context rebuild cost, not just the model forward pass.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import torch

from argus.streaming.detector import StreamingDetector


@dataclass
class DeploymentReport:
    n_flows: int
    throughput_flows_per_sec: float
    latency_p50_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    peak_memory_mb: float
    model_size_params: int
    model_size_mb: float


def model_size(model: torch.nn.Module) -> tuple[int, float]:
    """Returns (parameter count, size in MB assuming each parameter's own dtype)."""
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_params, size_bytes / (1024**2)


def measure_streaming_throughput(
    detector: StreamingDetector,
    feature_rows: np.ndarray,
    times_ms: np.ndarray,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    batch_size: int = 1,
) -> DeploymentReport:
    """Push flows through `detector` in chronological order, `batch_size` at a
    time (1 = per-flow latency, the deployment-realistic case), aggregating
    throughput, latency percentiles, peak memory, and model size.
    """
    n = len(times_ms)
    per_flow_latency_ms: list[float] = []

    tracemalloc.start()
    t_start = time.perf_counter()
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        verdicts = detector.push(feature_rows[lo:hi], times_ms[lo:hi], src_ids[lo:hi], dst_ids[lo:hi])
        per_flow_latency_ms.extend(v.latency_ms for v in verdicts)
    elapsed_s = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_params, size_mb = model_size(detector.model)
    latencies = np.array(per_flow_latency_ms) if per_flow_latency_ms else np.zeros(0)
    return DeploymentReport(
        n_flows=n,
        throughput_flows_per_sec=(n / elapsed_s) if elapsed_s > 0 else float("inf"),
        latency_p50_ms=float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
        latency_p99_ms=float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
        latency_mean_ms=float(latencies.mean()) if len(latencies) else 0.0,
        peak_memory_mb=peak_bytes / (1024**2),
        model_size_params=n_params,
        model_size_mb=size_mb,
    )
