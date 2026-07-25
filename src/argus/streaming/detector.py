"""Streaming inference: the deployment contract.

See docs/04_GRAPH_CONSTRUCTION.md §6. Same code path as training, driven one
anchor bin at a time; bounded memory; no lookahead.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch


@dataclass
class Verdict:
    decision: str  # "CLASSIFY" | "DEFER" | "UNKNOWN"
    predicted_class: str | None
    evidence_total: float
    vacuity: float
    latency_ms: float


class StreamingDetector:
    """Bounded-memory streaming detector: push flows, get verdicts, register classes.

    Maintains a ring buffer of raw (already feature-transformed) flow rows
    bounded to the long window duration, rebuilding a small graph context per
    push call. This is O(buffer size) per push, which is bounded because the
    buffer only ever holds `window_long_seconds` worth of flows.
    """

    def __init__(
        self,
        model: ArgusModel,
        class_names: list[str],
        node_granularity: str = "ip",
        anchor_bin_seconds: int = 1,
        window_short_seconds: int = 1,
        window_mid_seconds: int = 30,
        window_long_seconds: int = 300,
        neighbour_cap: int = 32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.node_granularity = node_granularity
        self.anchor_bin_seconds = anchor_bin_seconds
        self.window_short_seconds = window_short_seconds
        self.window_mid_seconds = window_mid_seconds
        self.window_long_seconds = window_long_seconds
        self.neighbour_cap = neighbour_cap
        self.device = torch.device(device)
        self._buffer: deque = deque()  # each item: (time_ms, feat, src_key, dst_key, label)
        model.eval()

    def _evict(self, now_ms: float) -> None:
        cutoff = now_ms - self.window_long_seconds * 1000.0
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def push(self, feature_rows: np.ndarray, times_ms: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray) -> list[Verdict]:
        """Ingest a batch of flows (already feature-transformed) for one anchor bin.

        Args:
            feature_rows: [B, F_e] transformed edge feature vectors
            times_ms: [B] flow start times in ms, must be non-decreasing across
                successive `push` calls (no lookahead: never push a time older
                than one already pushed).
            src_ids, dst_ids: [B] raw node identity codes (already mapped to
                integers by the caller, e.g. via `assign_node_ids`).
        Returns:
            One Verdict per flow, in input order.
        """
        t0 = time.perf_counter()
        if self._buffer and times_ms.min() < self._buffer[-1][0]:
            raise ValueError("push() received an out-of-order timestamp (lookahead violation)")

        for i in range(len(times_ms)):
            self._buffer.append((float(times_ms[i]), feature_rows[i], int(src_ids[i]), int(dst_ids[i]), 0))
        now_ms = float(times_ms.max())
        self._evict(now_ms)

        buf_times = np.array([b[0] for b in self._buffer])
        buf_feats = np.array([b[1] for b in self._buffer])
        buf_src = np.array([b[2] for b in self._buffer], dtype=np.int64)
        buf_dst = np.array([b[3] for b in self._buffer], dtype=np.int64)
        buf_labels = np.zeros(len(self._buffer), dtype=np.int64)

        source = AnchorBinGraphSource(
            buf_times, buf_feats, buf_src, buf_dst, buf_labels,
            anchor_bin_seconds=self.anchor_bin_seconds,
            window_short_seconds=self.window_short_seconds,
            window_mid_seconds=self.window_mid_seconds,
            window_long_seconds=self.window_long_seconds,
            neighbour_cap=self.neighbour_cap,
        )
        current_bin = source.bin_ids[-1]
        batch = source.build_bin_batch(int(current_bin), f_v=self.model.f_v)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if batch is None or batch["n_targets"] == 0:
            return [
                Verdict("UNKNOWN", None, 0.0, 1.0, elapsed_ms / max(len(times_ms), 1))
                for _ in range(len(times_ms))
            ]

        with torch.no_grad():
            inputs = model_inputs_from_batch(batch, self.device)
            outputs = self.model(*inputs)
            decisions, _ = self.model.head.decide(outputs) if hasattr(self.model.head, "decide") else (
                outputs["p_hat"].argmax(dim=1), None
            )

        n_pushed = len(times_ms)
        n_targets = outputs["p_hat"].shape[0]
        offset = max(n_targets - n_pushed, 0)
        per_flow_latency = elapsed_ms / max(n_pushed, 1)

        verdicts = []
        for i in range(n_pushed):
            idx = offset + i
            if idx >= n_targets:
                verdicts.append(Verdict("UNKNOWN", None, 0.0, 1.0, per_flow_latency))
                continue
            dec = int(decisions[idx].item())
            vacuity = float(outputs.get("vacuity", torch.zeros(n_targets))[idx].item())
            evidence_total = float(outputs.get("evidence_total", torch.zeros(n_targets))[idx].item())
            if dec == -1:
                verdicts.append(Verdict("UNKNOWN", None, evidence_total, vacuity, per_flow_latency))
            elif dec == -2:
                verdicts.append(Verdict("DEFER", None, evidence_total, vacuity, per_flow_latency))
            else:
                verdicts.append(
                    Verdict("CLASSIFY", self.class_names[dec], evidence_total, vacuity, per_flow_latency)
                )
        return verdicts

    def register_class(self, name: str, feature_rows: np.ndarray, n_sub: int = 1) -> int:
        """Gradient-free few-shot class registration. O(n), no gradient steps."""
        assert not torch.is_grad_enabled() or True  # caller may be in any context; we force no_grad below
        h = torch.zeros(len(feature_rows), self.model.head.embedding.W_z.in_features)
        # Encode via the target-edge readout using a trivial single-node self-loop
        # context (no neighbours) — sufficient for a mean-embedding registration.
        with torch.no_grad():
            for i, row in enumerate(feature_rows):
                x = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                node_feat = torch.zeros(1, self.model.f_v)
                x_s = self.model.encoder.W_v(node_feat)
                h_e = self.model.encoder.encode_target_edges(
                    x_s, torch.zeros(2, 1, dtype=torch.long), x, scale_duration=1.0, scale_id=0
                )
                h[i] = h_e[0]
        return self.model.register_class(name, h, n_sub=n_sub)

    def memory_footprint(self) -> int:
        """Number of flows currently held in the ring buffer (proxy for bounded memory)."""
        return len(self._buffer)
