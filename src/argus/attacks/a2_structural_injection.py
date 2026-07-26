"""A2 — structural node/edge injection: the headline C2 attack.

See docs/10_ADVERSARIAL.md §3. Operates in the already-transformed feature
space consumed by `AnchorBinGraphSource`; injected flows are timed so they
never become prediction targets themselves (only context for the victim's
neighbourhood).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch


def sample_benign_like_flows(
    benign_pool: np.ndarray, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap-sample `m` feature vectors from a pool of real benign flows.

    A practical stand-in for the KDE sampler in docs/10_ADVERSARIAL.md §3 —
    both draw synthetic-but-realistic benign feature vectors; bootstrapping the
    empirical distribution is a defensible, simpler substitute.
    """
    if len(benign_pool) == 0:
        raise ValueError("benign_pool is empty")
    idx = rng.choice(len(benign_pool), size=m, replace=True)
    return benign_pool[idx]


def _injection_times(
    window_start_ms: float,
    target_time_ms: float,
    m: int,
    strata: int,
    spread: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place `m` injected timestamps strictly before `target_time_ms`.

    spread='single_stratum': all within the most recent stratum (cheap attack).
    spread='all_strata': spread evenly across all `strata` sub-intervals of the
    window (strong attack; must sustain injection for the whole window).
    """
    span = max(target_time_ms - window_start_ms, 1.0)
    strata = max(strata, 1)
    if spread == "single_stratum":
        lo = target_time_ms - span / strata
        hi = target_time_ms - 1.0
        return rng.uniform(max(lo, window_start_ms), max(hi, lo), size=m)
    if spread == "all_strata":
        edges = np.linspace(window_start_ms, target_time_ms - 1.0, strata + 1)
        per = m // strata
        remainder = m - per * strata
        times = []
        for i in range(strata):
            n = per + (1 if i < remainder else 0)
            if n > 0:
                times.append(rng.uniform(edges[i], max(edges[i + 1], edges[i] + 1), size=n))
        return np.concatenate(times) if times else np.zeros(0)
    raise ValueError(f"Unknown spread strategy: {spread}")


@dataclass
class InjectionResult:
    budget: int
    clean_decision: int
    attacked_decision: int
    clean_evidence_total: float
    attacked_evidence_total: float
    evaded: bool


def build_injected_source(
    source: AnchorBinGraphSource,
    bin_id: int,
    victim_node_id: int,
    injection_host_id: int,
    benign_pool: np.ndarray,
    budget: int,
    spread: str = "all_strata",
    strata: int = 4,
    seed: int = 0,
    benign_class_id: int = 0,
) -> AnchorBinGraphSource:
    """Return a new source with `budget` synthetic flows injected into the
    victim's long window, ending strictly before the target bin.
    """
    rng = np.random.default_rng(seed)
    long_seconds = source.scale_durations["long"]
    ranges = dict((b, (lo, hi)) for b, lo, hi in source.ranges["long"])
    if bin_id not in ranges:
        raise ValueError(f"bin_id {bin_id} has no long-window range")
    lo, hi = ranges[bin_id]
    if hi <= lo:
        window_start_ms = source.times_ms[max(lo - 1, 0)]
    else:
        window_start_ms = source.times_ms[hi - 1] - long_seconds * 1000.0

    # The injected flows must land strictly before the target ANCHOR BIN's
    # start, not merely before the victim's raw timestamp — otherwise a flow
    # placed a few milliseconds earlier can still fall in the same 1-second
    # bin and be mis-scored as an extra target.
    t0 = float(source.times_ms[0])
    bin_start_ms = t0 + bin_id * source.anchor_bin_seconds * 1000.0
    target_time_ms = bin_start_ms

    # Never place an injected timestamp before the dataset's true start: doing
    # so would shift t0 on re-sort and silently renumber every anchor bin.
    window_start_ms = max(window_start_ms, t0)
    target_time_ms = max(target_time_ms, window_start_ms + 1.0)

    if budget <= 0:
        return source

    inj_times = _injection_times(window_start_ms, target_time_ms, budget, strata, spread, rng)
    inj_feats = sample_benign_like_flows(benign_pool, budget, rng)
    inj_src = np.full(budget, injection_host_id, dtype=np.int64)
    inj_dst = np.full(budget, victim_node_id, dtype=np.int64)
    inj_labels = np.full(budget, benign_class_id, dtype=source.labels.dtype)  # benign context only

    new_times = np.concatenate([source.times_ms, inj_times.astype(source.times_ms.dtype)])
    new_feats = np.concatenate([source.edge_features, inj_feats.astype(source.edge_features.dtype)])
    new_src = np.concatenate([source.src_ids, inj_src])
    new_dst = np.concatenate([source.dst_ids, inj_dst])
    new_labels = np.concatenate([source.labels, inj_labels])
    # Provenance mask for F7: carries any pre-existing injected flags through
    # (stacked attacks) and marks the new rows. The constructor re-sorts by
    # time, so this must be passed in rather than inferred positionally.
    new_injected = np.concatenate([source.is_injected, np.ones(budget, dtype=bool)])

    return AnchorBinGraphSource(
        new_times, new_feats, new_src, new_dst, new_labels,
        # Must match the source this was built from: evaluate_flow() looks up
        # the *same* bin_id in both clean_source and this injected source, and
        # bin_id is only comparable across the two if anchor_bin_seconds
        # agrees (bin_id = floor(t/anchor_bin_seconds) — a mismatch silently
        # points evaluate_flow at an unrelated time window post-injection).
        anchor_bin_seconds=source.anchor_bin_seconds,
        window_short_seconds=source.scale_durations["short"],
        window_mid_seconds=source.scale_durations["mid"],
        window_long_seconds=source.scale_durations["long"],
        neighbour_cap=source.neighbour_cap,
        sampling=source.sampling,
        strata=source.strata,
        seed=seed,
        te7_enabled=source.te7_enabled,
        spectral_nbins=source.spectral_nbins,
        spectral_min_flows=source.spectral_min_flows,
        is_injected=new_injected,
    )


def _find_target_bin_for_flow(
    source: AnchorBinGraphSource, flow_position: int
) -> int:
    """Given a position in the (time-sorted) source arrays, return its anchor bin id."""
    return int(source.bin_ids[flow_position])


@torch.no_grad()
def evaluate_flow(
    model: ArgusModel,
    source: AnchorBinGraphSource,
    bin_id: int,
    target_index_within_bin: int,
    device: torch.device,
) -> tuple[int, float]:
    """Run the model for `bin_id` and return (decision, evidence_total) for the
    `target_index_within_bin`-th target flow in that bin.
    """
    model.eval()
    batch = source.build_bin_batch(bin_id, f_v=model.f_v)
    if batch is None or batch["n_targets"] == 0:
        raise ValueError(f"bin {bin_id} has no targets after injection")
    inputs = model_inputs_from_batch(batch, device)
    outputs = model(*inputs)
    idx = min(target_index_within_bin, outputs["p_hat"].shape[0] - 1)
    if hasattr(model.head, "decide"):
        decisions, _ = model.head.decide(outputs)
        decision = int(decisions[idx].item())
    else:
        decision = int(outputs["p_hat"][idx].argmax().item())
    evidence_total = float(outputs.get("evidence_total", outputs["p_hat"].max(dim=1).values)[idx].item())
    return decision, evidence_total


def run_a2_budget_sweep(
    model: ArgusModel,
    clean_source: AnchorBinGraphSource,
    bin_id: int,
    target_index_within_bin: int,
    victim_node_id: int,
    injection_host_id: int,
    benign_pool: np.ndarray,
    device: torch.device,
    budgets: list[int] = (0, 1, 2, 4, 8, 16, 32, 64),
    spread: str = "all_strata",
    strata: int = 4,
    benign_class_id: int = 0,
) -> list[InjectionResult]:
    """Sweep injection budgets and record whether the target flow evades detection.

    "Evaded" means the attacked decision is the benign class (or UNKNOWN, which
    is also counted as an evasion of the *known-attack* alert per docs' ASR
    definition — UNKNOWN still alerts, so this is a conservative choice: only
    a benign classification counts as a full evasion here).
    """
    clean_decision, clean_evidence = evaluate_flow(
        model, clean_source, bin_id, target_index_within_bin, device
    )
    results = []
    for m in budgets:
        if m == 0:
            attacked_decision, attacked_evidence = clean_decision, clean_evidence
        else:
            injected_source = build_injected_source(
                clean_source, bin_id, victim_node_id, injection_host_id,
                benign_pool, budget=m, spread=spread, strata=strata,
                benign_class_id=benign_class_id,
            )
            attacked_decision, attacked_evidence = evaluate_flow(
                model, injected_source, bin_id, target_index_within_bin, device
            )
        results.append(
            InjectionResult(
                budget=m,
                clean_decision=clean_decision,
                attacked_decision=attacked_decision,
                clean_evidence_total=clean_evidence,
                attacked_evidence_total=attacked_evidence,
                evaded=(attacked_decision == benign_class_id),
            )
        )
    return results
