"""Multi-scale anchor-bin windowing.

See docs/04_GRAPH_CONSTRUCTION.md §2. Windows are backward-looking only.
"""

from __future__ import annotations

import numpy as np


def assign_anchor_bins(times_ms: np.ndarray, anchor_bin_seconds: int = 1) -> np.ndarray:
    """Assign each flow to an anchor bin index based on its start time.

    Args:
        times_ms: [N] flow start times in milliseconds, must be sorted ascending.
        anchor_bin_seconds: bin width in seconds.
    Returns:
        [N] integer bin id per flow.
    """
    bin_ms = anchor_bin_seconds * 1000
    t0 = times_ms[0]
    return ((times_ms - t0) // bin_ms).astype(np.int64)


def window_edge_ranges(
    times_ms: np.ndarray,
    bin_ids: np.ndarray,
    scale_seconds: int,
) -> list[tuple[int, int, int]]:
    """For each distinct anchor bin, compute the [lo, hi) index range of flows
    whose time falls within [bin_end - scale_seconds*1000, bin_end], using a
    sliding two-pointer scan over the sorted times array.

    Args:
        times_ms: [N] sorted ascending flow times (ms).
        bin_ids: [N] anchor bin id per flow (from assign_anchor_bins).
        scale_seconds: window duration for this scale.
    Returns:
        list of (bin_id, lo, hi) where flows[lo:hi] are inside the window ending
        at the end of that anchor bin. `hi` is also the first index of that bin's
        own target flows (flows with bin_id == bin_id start at some index <= hi).
    """
    n = len(times_ms)
    scale_ms = scale_seconds * 1000
    unique_bins = np.unique(bin_ids)
    results = []
    lo = 0
    hi = 0
    for b in unique_bins:
        # hi = first index with bin_id > b (exclusive end of this bin's own flows)
        while hi < n and bin_ids[hi] <= b:
            hi += 1
        bin_end_time = times_ms[hi - 1]  # last flow's time in this bin (approx bin end)
        window_start = bin_end_time - scale_ms
        while lo < hi and times_ms[lo] < window_start:
            lo += 1
        results.append((int(b), int(lo), int(hi)))
    return results
