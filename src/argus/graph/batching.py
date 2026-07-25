"""Assemble anchor-bin batches into the tensors ArgusModel consumes.

See docs/04_GRAPH_CONSTRUCTION.md §5. This implementation processes one anchor
bin at a time; multi-bin BPTT batching (§5, T_bptt=8) is a straightforward
extension left for the training loop to iterate bin-by-bin while carrying the
per-node memory dict forward and detaching every `bptt_chunk` bins.
"""

from __future__ import annotations

import numpy as np
import torch

from argus.graph.node_features import compute_node_features_block1, compute_node_features_te7
from argus.graph.sampler import sample_neighbours
from argus.graph.windows import assign_anchor_bins, window_edge_ranges


class AnchorBinGraphSource:
    """Precomputes anchor-bin windows for S/M/L scales over a sorted flow table."""

    def __init__(
        self,
        times_ms: np.ndarray,
        edge_features: np.ndarray,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        labels: np.ndarray,
        anchor_bin_seconds: int = 1,
        window_short_seconds: int = 1,
        window_mid_seconds: int = 30,
        window_long_seconds: int = 300,
        neighbour_cap: int = 32,
        sampling: str = "recency_stratified",
        strata: int = 4,
        seed: int = 0,
        te7_enabled: bool = True,
        spectral_nbins: int = 64,
        spectral_min_flows: int = 8,
    ) -> None:
        order = np.argsort(times_ms, kind="stable")
        self.times_ms = times_ms[order]
        self.edge_features = edge_features[order]
        self.src_ids = src_ids[order]
        self.dst_ids = dst_ids[order]
        self.labels = labels[order]

        self.bin_ids = assign_anchor_bins(self.times_ms, anchor_bin_seconds)
        self.anchor_bin_seconds = anchor_bin_seconds
        self.scale_durations = {
            "short": window_short_seconds,
            "mid": window_mid_seconds,
            "long": window_long_seconds,
        }
        self.ranges = {
            name: window_edge_ranges(self.times_ms, self.bin_ids, dur)
            for name, dur in self.scale_durations.items()
        }
        self.neighbour_cap = neighbour_cap
        self.sampling = sampling
        self.strata = strata
        self.rng = np.random.default_rng(seed)
        self.unique_bins = sorted(set(self.bin_ids.tolist()))
        self.te7_enabled = te7_enabled
        self.spectral_nbins = spectral_nbins
        self.spectral_min_flows = spectral_min_flows
        self._prev_active_nodes: set[int] = set()

    def _cap_and_extract(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cap neighbours for edges[lo:hi]; return raw (uncapped-index) arrays."""
        if hi <= lo:
            empty_f = np.zeros((0, self.edge_features.shape[1]), dtype=np.float32)
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty_f, np.zeros(0, dtype=np.float32)
        src = self.src_ids[lo:hi]
        dst = self.dst_ids[lo:hi]
        feat = self.edge_features[lo:hi]
        window_end = self.times_ms[hi - 1]
        dt_seconds = (window_end - self.times_ms[lo:hi]) / 1000.0

        keep = sample_neighbours(
            dst, k=self.neighbour_cap, strategy=self.sampling, strata=self.strata, rng=self.rng
        )
        if len(keep) == 0:
            keep = np.arange(len(dst))
        return src[keep], dst[keep], feat[keep], dt_seconds[keep].clip(min=0.0)

    def _raw_long_window(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return uncapped (src, dst, times, byte-magnitude proxy) for node feature computation.

        Node features (docs/04_GRAPH_CONSTRUCTION.md §3) are computed over the
        full long window, not the degree-capped sampled context used for
        message passing.
        """
        if hi <= lo:
            return (
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
            )
        src = self.src_ids[lo:hi]
        dst = self.dst_ids[lo:hi]
        times = self.times_ms[lo:hi]
        # Byte-volume proxy: L1 norm of the (already-conditioned) edge feature
        # vector. Raw byte columns aren't threaded through this generic API;
        # this proxy preserves relative magnitude for the byte_volume feature.
        byte_proxy = np.abs(self.edge_features[lo:hi]).sum(axis=1)
        return src, dst, times, byte_proxy

    def build_bin_batch(self, bin_id: int, f_v: int = 18) -> dict | None:
        """Build the S/M/L graphs + target edges + labels for one anchor bin.

        Node index spaces are unified across all three scales, as required by
        docs/04_GRAPH_CONSTRUCTION.md §5.
        """
        ranges = {name: dict((b, (lo, hi)) for b, lo, hi in self.ranges[name]) for name in ("short", "mid", "long")}
        if bin_id not in ranges["short"]:
            return None

        raw = {}
        for name in ("short", "mid", "long"):
            if bin_id not in ranges[name]:
                return None
            lo, hi = ranges[name][bin_id]
            raw[name] = self._cap_and_extract(lo, hi)

        long_lo, long_hi = ranges["long"][bin_id]
        self._last_long_raw = self._raw_long_window(long_lo, long_hi)

        # Union node id space across all three scales.
        all_ids = np.unique(np.concatenate([np.concatenate([raw[n][0], raw[n][1]]) for n in raw]))
        local = {int(nid): i for i, nid in enumerate(all_ids)}

        def _remap(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
            if len(src) == 0:
                return torch.zeros(2, 0, dtype=torch.long)
            ls = np.array([local[int(s)] for s in src], dtype=np.int64)
            ld = np.array([local[int(d)] for d in dst], dtype=np.int64)
            return torch.tensor(np.stack([ls, ld]), dtype=torch.long)

        scale_tensors = {}
        for name in ("short", "mid", "long"):
            src, dst, feat, dt = raw[name]
            edge_index = _remap(src, dst)
            edge_attr = torch.tensor(feat, dtype=torch.float32)
            edge_dt = torch.tensor(dt, dtype=torch.float32)
            scale_tensors[name] = (edge_index, edge_attr, edge_dt)

        # Target flows: those whose bin_id == bin_id. They are guaranteed inside
        # the short-scale window by construction (docs/04_GRAPH_CONSTRUCTION.md §5).
        short_lo, short_hi = ranges["short"][bin_id]
        target_global_positions = np.nonzero(self.bin_ids[short_lo:short_hi] == bin_id)[0] + short_lo
        target_labels = self.labels[target_global_positions]
        target_feat = self.edge_features[target_global_positions]
        target_src = self.src_ids[target_global_positions]
        target_dst = self.dst_ids[target_global_positions]

        # Ensure target nodes are present in the shared node space; extend if needed.
        for nid in np.concatenate([target_src, target_dst]):
            if int(nid) not in local:
                local[int(nid)] = len(local)

        target_local_src = torch.tensor([local[int(s)] for s in target_src], dtype=torch.long)
        target_local_dst = torch.tensor([local[int(d)] for d in target_dst], dtype=torch.long)
        target_edge_index = torch.stack([target_local_src, target_local_dst])
        target_edge_attr = torch.tensor(target_feat, dtype=torch.float32)

        n_nodes = max(len(local), 1)
        node_feat = self._compute_node_features(local, n_nodes, f_v)

        return {
            "node_feat": node_feat,
            "scale_short": scale_tensors["short"],
            "scale_mid": scale_tensors["mid"],
            "scale_long": scale_tensors["long"],
            "target_edge_index": target_edge_index,
            "target_edge_attr": target_edge_attr,
            "target_labels": torch.tensor(target_labels, dtype=torch.long),
            "n_targets": len(target_global_positions),
        }

    def _compute_node_features(
        self, local: dict[int, int], n_nodes: int, f_v: int
    ) -> torch.Tensor:
        """Compute real Block-1 (+ optional TE7) node features for the shared node space.

        See docs/04_GRAPH_CONSTRUCTION.md §3. Falls back to zeros for any node
        not covered by the long window (should not normally happen since the
        long window is a superset of short/mid).
        """
        node_ids = np.array(sorted(local, key=lambda k: local[k]), dtype=np.int64)
        block1_dim = 12
        te7_dim = 6 if self.te7_enabled else 0
        out = np.zeros((n_nodes, block1_dim + te7_dim), dtype=np.float32)

        src, dst, times, byte_proxy = self._last_long_raw
        if len(src) > 0:
            block1 = compute_node_features_block1(
                node_ids, src, dst, times, byte_proxy,
                window_long_seconds=self.scale_durations["long"],
                k_cap=self.neighbour_cap,
                prev_active_nodes=self._prev_active_nodes,
            )
            out[:, :block1_dim] = block1
            if self.te7_enabled:
                all_ids_for_te7 = np.concatenate([src, dst])
                all_times_for_te7 = np.concatenate([times, times])
                te7 = compute_node_features_te7(
                    node_ids, all_ids_for_te7, all_times_for_te7,
                    window_long_seconds=self.scale_durations["long"],
                    nbins=self.spectral_nbins, min_flows=self.spectral_min_flows,
                )
                out[:, block1_dim:block1_dim + te7_dim] = te7
            self._prev_active_nodes = set(node_ids.tolist())

        # Pad or truncate to the requested f_v (model's expected node feature dim).
        actual_dim = out.shape[1]
        if actual_dim < f_v:
            out = np.concatenate([out, np.zeros((n_nodes, f_v - actual_dim), dtype=np.float32)], axis=1)
        elif actual_dim > f_v:
            out = out[:, :f_v]
        return torch.tensor(out, dtype=torch.float32)
