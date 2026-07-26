"""Class-imbalance mechanisms for Stage-1 training (docs/06_TRAINING.md §4).

CICIDS2018's Protocol-A training split spans 34,014:1 — `ddos_hoic` at 476,204
rows against `brute_xss` at 14. The design specifies three mechanisms applied
together; this module implements the two that are pure functions of the label
distribution, plus the minimum-count guard's class selection:

1. **Class-balanced subsampling of loss targets.** Every flow in an anchor bin
   is still embedded (the graph needs them), but the loss is computed on at
   most `n_per_class` targets per class. This decouples graph density from
   loss balance — see `balanced_target_indices`.
2. **Effective-number class weights** in `L_am`: ``w_c = (1 - v) / (1 - v^n_c)``
   with ``v = 0.999`` — see `effective_number_weights`.
3. **Minimum-count guard.** Classes below `min_count_for_prototype` training
   rows are excluded from Stage-1 prototype training and evaluated instead as
   few-shot registration targets — see `min_count_excluded_classes`.

Deliberately *not* implemented: duplication oversampling. At n_c = 14 that
memorises rather than generalises (docs/06_TRAINING.md §4).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch


def effective_number_weights(
    class_names: Sequence[str],
    class_counts: dict[str, int],
    nu: float = 0.999,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-class loss weights from the effective number of samples.

    ``w_c = (1 - nu) / (1 - nu ** n_c)`` (Cui et al., 2019, "Class-Balanced Loss
    Based on Effective Number of Samples"), as specified in
    docs/06_TRAINING.md §4 with ``nu = 0.999``.

    The point of this form over plain inverse frequency is saturation: samples
    of a large class overlap, so the *marginal* information a new one adds
    decays geometrically. On this split it compresses a raw 34,014:1 frequency
    ratio to a 71.9:1 weight ratio — enough to stop the tail being ignored,
    not so much that fourteen `brute_xss` rows dominate the gradient.

    Weights are rescaled so the mean over classes present in training is 1.0,
    which keeps the loss on the same scale as the unweighted version — the
    tuned `stage1_lr` and `grad_clip` stay valid.

    Classes with a count of 0 (absent from training) get weight 0.

    Args:
        class_names: ordered vocabulary; index i is class i.
        class_counts: name -> training row count. Missing names count as 0.
        nu: the ``nu`` hyperparameter, in [0, 1). ``nu = 0`` gives uniform
            weights (the unweighted case).
    Returns:
        [C] tensor of weights, suitable as `F.cross_entropy(weight=...)`.
    """
    if not 0.0 <= nu < 1.0:
        raise ValueError(f"effective_number_nu must be in [0, 1), got {nu}")
    # float64 throughout: nu ** n_c underflows for large n_c (0.999 ** 476204
    # is ~1e-207), and doing it in float32 would flush to zero *before* the
    # subtraction rather than after. The limit is the same either way here, but
    # mid-sized classes near the underflow boundary would quantise badly.
    counts = torch.tensor(
        [float(class_counts.get(name, 0)) for name in class_names], dtype=torch.float64
    )
    present = counts > 0
    if not bool(present.any()):
        raise ValueError("no class has a positive training count")
    if nu == 0.0:
        weights = present.to(torch.float64)
    else:
        effective_num = 1.0 - torch.pow(torch.tensor(nu, dtype=torch.float64), counts)
        weights = torch.where(
            present, (1.0 - nu) / effective_num, torch.zeros_like(effective_num)
        )
    weights = weights * (present.sum() / weights.sum())
    return weights.to(device=device, dtype=dtype)


def effective_target_counts(
    label_ids: np.ndarray,
    bin_ids: np.ndarray,
    num_classes: int,
    n_per_class: int | None,
) -> list[int]:
    """Per-class loss-target counts *after* class-balanced capping.

    The weights in `effective_number_weights` must describe the distribution
    the loss actually sees, and on this dataset capping changes it beyond
    recognition. With `anchor_bin_seconds = 10` and `n_per_class = 32`,
    measured over the real CICIDS2018 training split:

        ddos_hoic     27.35% -> 3.61%   (a burst: huge counts in few bins)
        infiltration   7.56% -> 30.82%  (low and slow: spread over many bins)

    Weighting by *raw* counts would hand those two the same weight, when after
    capping one is 8.5x more prevalent than the other — the correction would
    point the wrong way. Capping alone takes the imbalance from 1546:1 to
    175:1 (excluding the min-count class); these counts are what the residual
    weighting should then work on.

    This is the expected count under capping, not a sample of it: each bin
    contributes `min(count, n_per_class)` per class, which is exactly what
    `balanced_target_indices` draws every epoch regardless of which rows it
    picks.

    Args:
        label_ids: [N] class id per flow.
        bin_ids: [N] anchor bin id per flow (`graph.windows.assign_anchor_bins`).
        num_classes: vocabulary size.
        n_per_class: the cap; None or <= 0 returns plain per-class totals.
    Returns:
        list of length `num_classes`.
    """
    labels = np.asarray(label_ids, dtype=np.int64)
    if n_per_class is None or n_per_class <= 0:
        return np.bincount(labels, minlength=num_classes)[:num_classes].tolist()
    bins = np.asarray(bin_ids, dtype=np.int64)
    keys, counts = np.unique(bins * num_classes + labels, return_counts=True)
    out = np.zeros(num_classes, dtype=np.int64)
    np.add.at(out, keys % num_classes, np.minimum(counts, n_per_class))
    return out.tolist()


def min_count_excluded_classes(
    class_names: Sequence[str],
    class_counts: dict[str, int],
    min_count: int,
) -> list[int]:
    """Class indices below `min_count` training rows (docs/06_TRAINING.md §4.3).

    These are excluded from the Stage-1 loss rather than dropped from the
    vocabulary: their column stays in the prototype bank and in the confusion
    matrix, so few-shot registration (script 08) has an index to fill and the
    graph cache's label ids stay valid. What changes is that no AM-Softmax
    gradient ever pulls a prototype *toward* fourteen samples — which is what
    "excluded from Stage-1 prototype training" means in practice.

    A count of 0 does not qualify: a class absent from training entirely is
    already handled by its zero effective-number weight, and flagging it here
    would wrongly suppress it from the metric.
    """
    if min_count <= 0:
        return []
    return [
        i for i, name in enumerate(class_names)
        if 0 < class_counts.get(name, 0) < min_count
    ]


def balanced_target_indices(
    targets: torch.Tensor,
    n_per_class: int | None,
    exclude_class_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Indices of a class-balanced subsample of `targets`.

    Draws up to `n_per_class` targets per class present, uniformly without
    replacement, after removing `exclude_class_ids` entirely. Classes with
    fewer than `n_per_class` present are kept whole — this caps the head rather
    than inflating the tail, so no row is ever duplicated.

    Returned indices are sorted, so the subsample keeps the batch's original
    ordering. Sampling uses the global torch RNG, which checkpoint/resume
    already saves and restores.

    Args:
        targets: [B] long tensor of class ids.
        n_per_class: cap per class; None or <= 0 disables capping.
        exclude_class_ids: class ids to drop from the loss entirely.
    Returns:
        [B'] long tensor of indices into `targets`.
    """
    keep = torch.ones_like(targets, dtype=torch.bool)
    excluded = sorted({int(c) for c in exclude_class_ids})
    if excluded:
        keep &= ~torch.isin(targets, torch.tensor(excluded, device=targets.device))
    idx = keep.nonzero(as_tuple=True)[0]
    if n_per_class is None or n_per_class <= 0 or idx.numel() == 0:
        return idx

    kept: list[torch.Tensor] = []
    for c in torch.unique(targets[idx]):
        c_idx = idx[targets[idx] == c]
        if c_idx.numel() > n_per_class:
            perm = torch.randperm(c_idx.numel(), device=c_idx.device)
            c_idx = c_idx[perm[:n_per_class]]
        kept.append(c_idx)
    return torch.cat(kept).sort().values
