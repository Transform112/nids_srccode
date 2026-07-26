"""Transparent disk cache for graph batches.

Graph construction is CPU-bound and dominated the training wall-clock
(~4-8 sec/bin for 60K bins/epoch).  This wrapper intercepts
``build_bin_batch``, serialises every batch to ``.pt`` on first encounter,
and replays from disk on all subsequent epochs.  The first epoch still pays
the construction cost; every later epoch is I/O-bound (~50 ms/bin).

Cache keying follows docs/12_IMPLEMENTATION_PLAN.md §4.5.

See docs/04_GRAPH_CONSTRUCTION.md §5.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from argus.graph.batching import AnchorBinGraphSource


class CachedGraphSource:
    """Wraps an ``AnchorBinGraphSource`` with transparent per-bin disk caching.

    Usage::

        source = AnchorBinGraphSource(...)
        cached = CachedGraphSource(source, cache_dir="/path/to/cache", label="train")
        # First epoch: builds + writes ~60K .pt files.
        # Later epochs: torch.load() each bin in ~50 ms.
    """

    def __init__(
        self,
        source: AnchorBinGraphSource,
        cache_dir: str | Path,
        label: str = "",
        log_every_bins: int = 500,
    ) -> None:
        self._source = source
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._label = label
        self._log_every = log_every_bins

        self._hits = 0
        self._misses = 0
        self._build_time_ms = 0.0
        self._load_time_ms = 0.0
        self._total_bins = len(source.unique_bins)

    # -- delegated properties -------------------------------------------------
    @property
    def unique_bins(self) -> list[int]:
        return self._source.unique_bins

    @property
    def edge_features(self) -> "np.ndarray":
        return self._source.edge_features

    @property
    def n_nodes(self) -> int:
        return len(set(self._source.src_ids.tolist()) | set(self._source.dst_ids.tolist()))

    # -- main API -------------------------------------------------------------
    def build_bin_batch(self, bin_id: int, f_v: int = 18) -> dict | None:
        cache_path = self._cache_dir / f"bin_{bin_id:06d}.pt"

        if cache_path.exists():
            t0 = time.perf_counter()
            batch = torch.load(cache_path, weights_only=False)
            self._load_time_ms += (time.perf_counter() - t0) * 1000
            self._hits += 1
        else:
            t0 = time.perf_counter()
            batch = self._source.build_bin_batch(bin_id, f_v)
            self._build_time_ms += (time.perf_counter() - t0) * 1000
            self._misses += 1
            if batch is not None:
                torch.save(batch, cache_path)

        if (self._hits + self._misses) % self._log_every == 0:
            total = self._hits + self._misses
            pct = total / self._total_bins * 100 if self._total_bins else 0
            prefix = f"[{self._label}] " if self._label else ""
            if self._misses > 0:
                print(f"  {prefix}build {total}/{self._total_bins} bins ({pct:.0f}%)  "
                      f"cached={self._hits}  new={self._misses}  "
                      f"build={self._build_time_ms/1000:.1f}s  load={self._load_time_ms/1000:.1f}s",
                      flush=True)
            else:
                print(f"  {prefix}replay {total}/{self._total_bins} bins ({pct:.0f}%)  "
                      f"load={self._load_time_ms/1000:.1f}s",
                      flush=True)

        return batch

    def stats(self) -> dict:
        return {
            "total_bins": self._total_bins,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "build_time_s": round(self._build_time_ms / 1000, 1),
            "load_time_s": round(self._load_time_ms / 1000, 1),
        }
