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

import gzip
import json
import time
from pathlib import Path
from typing import Any

import torch

from argus.graph.batching import AnchorBinGraphSource

# Config keys that determine cached-batch *content*. Bins are keyed by a bare
# integer (`bin_{id:06d}`), so e.g. changing anchor_bin_seconds reuses the same
# filenames for a *different* time window — without a fingerprint check that
# mismatch loads silently instead of raising. See docs/04_GRAPH_CONSTRUCTION.md
# §5 and AGENT_GUIDE.md §7 (row 10: stale cache across resumed sessions).
CACHE_FINGERPRINT_GRAPH_KEYS = (
    "node_granularity", "anchor_bin_seconds", "window_short_seconds",
    "window_mid_seconds", "window_long_seconds", "neighbour_cap",
    "sampling", "strata",
)
# TE7/spectral settings also change the cached node-feature tensor's width
# and content (graph/node_features.py), even though they live under
# cfg.features, not cfg.graph.
CACHE_FINGERPRINT_FEATURE_KEYS = (
    "te7_enabled", "spectral_nbins", "spectral_min_flows",
)


def graph_config_fingerprint(cfg: Any) -> dict:
    """The subset of config that determines cached-batch content."""
    fp = {k: getattr(cfg.graph, k) for k in CACHE_FINGERPRINT_GRAPH_KEYS}
    fp.update({k: getattr(cfg.features, k) for k in CACHE_FINGERPRINT_FEATURE_KEYS})
    return fp


def verify_or_write_cache_meta(cache_dir: str | Path, cfg: Any) -> None:
    """Guard a per-split cache directory against silent config drift.

    First call for a fresh cache_dir writes ``_meta.json``; every later call
    (same run or a resumed one) must match it exactly, or raises rather than
    replaying graphs built under different windowing/sampling settings.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "_meta.json"
    current = graph_config_fingerprint(cfg)

    if not meta_path.is_file():
        with open(meta_path, "w") as f:
            json.dump(current, f, indent=2)
        return

    with open(meta_path) as f:
        stored = json.load(f)
    if stored != current:
        diff = {k: (stored.get(k), current[k]) for k in current if stored.get(k) != current[k]}
        raise ValueError(
            f"Graph cache at {cache_dir} was built with different graph.* "
            f"settings than the current config (stored -> current): {diff}. "
            f"Clear the cache directory or match the original settings."
        )


def _save_compressed(obj: dict, path: Path) -> None:
    """torch.save -> gzip for ~3x smaller cache files."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as fin, gzip.open(path, "wb", compresslevel=3) as fout:
        fout.write(fin.read())
    tmp.unlink()


def _load_compressed(path: Path) -> dict:
    with gzip.open(path, "rb") as fin:
        return torch.load(fin, weights_only=False)


class CachedGraphSource:
    """Wraps an ``AnchorBinGraphSource`` with transparent per-bin disk caching.

    Usage::

        source = AnchorBinGraphSource(...)
        cached = CachedGraphSource(source, cache_dir="/path/to/cache", cfg=cfg, label="train")
        # First epoch: builds + writes ~60K .pt.gz files.
        # Later epochs: torch.load() each bin in ~50 ms.
    """

    def __init__(
        self,
        source: AnchorBinGraphSource,
        cache_dir: str | Path,
        cfg: Any,
        label: str = "",
        log_every_bins: int = 500,
    ) -> None:
        self._source = source
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        verify_or_write_cache_meta(self._cache_dir, cfg)
        self._label = label
        self._log_every = log_every_bins

        self._hits = 0
        self._misses = 0
        self._build_time_ms = 0.0
        self._load_time_ms = 0.0
        self._write_time_ms = 0.0
        self._total_bins = len(source.unique_bins)

    def reset_epoch_state(self) -> None:
        """Delegate epoch-boundary state reset to the wrapped source.

        Only matters for cache-miss builds; cached batches were built in one
        fresh sequential pass and already reflect reset-at-split-start
        semantics.
        """
        self._source.reset_epoch_state()

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
        # Prefer compressed cache (the on-disk format both this class and
        # 03_cache_graphs.py write); fall back to reading legacy uncompressed
        # files left over from before compression was added.
        cache_gz = self._cache_dir / f"bin_{bin_id:06d}.pt.gz"
        cache_pt = self._cache_dir / f"bin_{bin_id:06d}.pt"

        if cache_gz.exists():
            t0 = time.perf_counter()
            batch = _load_compressed(cache_gz)
            self._load_time_ms += (time.perf_counter() - t0) * 1000
            self._hits += 1
        elif cache_pt.exists():
            t0 = time.perf_counter()
            batch = torch.load(cache_pt, weights_only=False)
            self._load_time_ms += (time.perf_counter() - t0) * 1000
            self._hits += 1
        else:
            t0 = time.perf_counter()
            batch = self._source.build_bin_batch(bin_id, f_v)
            self._build_time_ms += (time.perf_counter() - t0) * 1000
            self._misses += 1
            if batch is not None:
                # Timed separately: torch.save + gzip is real per-bin cost that
                # used to be invisible in the progress line, which made a slow
                # first epoch impossible to attribute (build? write? host I/O?).
                t1 = time.perf_counter()
                _save_compressed(batch, cache_gz)
                self._write_time_ms += (time.perf_counter() - t1) * 1000

        if (self._hits + self._misses) % self._log_every == 0:
            total = self._hits + self._misses
            pct = total / self._total_bins * 100 if self._total_bins else 0
            prefix = f"[{self._label}] " if self._label else ""
            if self._misses > 0:
                print(f"  {prefix}build {total}/{self._total_bins} bins ({pct:.0f}%)  "
                      f"cached={self._hits}  new={self._misses}  "
                      f"build={self._build_time_ms/1000:.1f}s  "
                      f"write={self._write_time_ms/1000:.1f}s  "
                      f"load={self._load_time_ms/1000:.1f}s",
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
            "write_time_s": round(self._write_time_ms / 1000, 1),
            "load_time_s": round(self._load_time_ms / 1000, 1),
        }
