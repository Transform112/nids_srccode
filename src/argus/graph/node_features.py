"""Per-node feature computation: local stats (Block 1) + TE7 spectral descriptor (Block 2).

See docs/04_GRAPH_CONSTRUCTION.md §3. F_v = 18 (12 + 6), or 12 if te7_enabled=false.
"""

from __future__ import annotations

import numpy as np


def compute_node_features_block1(
    node_ids: np.ndarray,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    times_ms: np.ndarray,
    byte_totals: np.ndarray,
    window_long_seconds: int,
    k_cap: int,
    prev_active_nodes: set[int] | None = None,
) -> np.ndarray:
    """Compute the 12 observer-derived local statistics per node.

    Args:
        node_ids: [V] node ids to compute features for (union of src/dst in window).
        src_ids, dst_ids, times_ms, byte_totals: [E] arrays for edges in the long window.
        window_long_seconds: D_L.
        k_cap: neighbour cap K, used to normalise degree features.
        prev_active_nodes: node ids seen in the previous window (for is_new_host).
    Returns:
        [V, 12] float32 array.
    """
    v = len(node_ids)
    out = np.zeros((v, 12), dtype=np.float32)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    prev_active_nodes = prev_active_nodes or set()
    d_l = max(window_long_seconds, 1)

    out_deg = np.zeros(v, dtype=np.int64)
    in_deg = np.zeros(v, dtype=np.int64)
    peers: list[set] = [set() for _ in range(v)]
    dst_ports_seen: list[set] = [set() for _ in range(v)]
    byte_vol = np.zeros(v, dtype=np.float64)
    gaps: list[list[float]] = [[] for _ in range(v)]
    last_time: dict[int, float] = {}
    short_scale_count = np.zeros(v, dtype=np.int64)
    mid_scale_count = np.zeros(v, dtype=np.int64)

    short_cutoff = times_ms.max() - 1000 if len(times_ms) else 0
    mid_cutoff = times_ms.max() - 30_000 if len(times_ms) else 0

    for i in range(len(src_ids)):
        s, d, t, b = src_ids[i], dst_ids[i], times_ms[i], byte_totals[i]
        si, di = node_index.get(s), node_index.get(d)
        if si is not None:
            out_deg[si] += 1
            peers[si].add(d)
            byte_vol[si] += b
            if t >= mid_cutoff:
                mid_scale_count[si] += 1
            if t >= short_cutoff:
                short_scale_count[si] += 1
            if s in last_time:
                gaps[si].append(float(t - last_time[s]))
            last_time[s] = t
        if di is not None:
            in_deg[di] += 1
            peers[di].add(s)
            dst_ports_seen[di].add(d)  # placeholder; real dst port tracked upstream

    for i, nid in enumerate(node_ids):
        od, idg = out_deg[i], in_deg[i]
        out[i, 0] = min(od, k_cap) / k_cap
        out[i, 1] = min(idg, k_cap) / k_cap
        out[i, 2] = np.log1p(len(peers[i])) / np.log1p(max(k_cap, 2))
        out[i, 3] = np.log1p(len(dst_ports_seen[i])) / np.log1p(max(k_cap, 2))
        out[i, 4] = len(peers[i]) / (od + 1e-6)
        out[i, 5] = np.log1p(od / d_l)
        g = gaps[i]
        if g:
            log_gaps = np.log1p(np.array(g) / 1000.0)
            out[i, 6] = float(log_gaps.mean())
            out[i, 7] = float(log_gaps.std())
        out[i, 8] = np.log1p(byte_vol[i]) / 20.0
        out[i, 9] = short_scale_count[i] / (mid_scale_count[i] / 30.0 + 1e-6)
        out[i, 10] = idg / (idg + od + 1e-6)
        out[i, 11] = 0.0 if nid in prev_active_nodes else 1.0

    return out


def compute_node_features_te7(
    node_ids: np.ndarray,
    src_or_dst_ids: np.ndarray,
    times_ms: np.ndarray,
    window_long_seconds: int,
    nbins: int = 64,
    min_flows: int = 8,
) -> np.ndarray:
    """TE7 spectral beaconing descriptor: 6 scalars per node from the FFT of its
    arrival-time histogram within the long window.
    """
    v = len(node_ids)
    out = np.zeros((v, 6), dtype=np.float32)
    if len(times_ms) == 0:
        return out
    t_min = times_ms.min()
    window_ms = window_long_seconds * 1000
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    for i, nid in enumerate(node_ids):
        mask = src_or_dst_ids == nid
        flow_times = times_ms[mask]
        if len(flow_times) < min_flows:
            continue
        rel = (flow_times - t_min) / max(window_ms, 1)
        counts, _ = np.histogram(rel, bins=nbins, range=(0.0, 1.0))
        counts = counts.astype(np.float64) - counts.mean()
        power = np.abs(np.fft.rfft(counts)) ** 2
        power = power[1:]  # discard DC
        if power.sum() <= 0 or len(power) == 0:
            continue
        p = power / power.sum()
        dominant_idx = int(np.argmax(power))
        out[i, 0] = dominant_idx / max(len(power), 1)
        out[i, 1] = float(power.max() / (power.sum() + 1e-9))
        entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(max(len(power), 2))
        out[i, 2] = float(entropy)
        geo_mean = np.exp(np.mean(np.log(power + 1e-12)))
        arith_mean = power.mean()
        out[i, 3] = float(geo_mean / (arith_mean + 1e-9))
        out[i, 4] = float(power.max() / (power.mean() + 1e-9))
        low_freq = power[: max(len(power) // 8, 1)]
        out[i, 5] = float(low_freq.sum() / (power.sum() + 1e-9))

    return out
