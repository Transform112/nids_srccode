"""Shared train/eval loop over anchor-bin graph batches.

Processes anchor bins in truncated-BPTT chunks of `bptt_chunk` bins
(docs/06_TRAINING.md §5, docs/05_ARCHITECTURE.md §4): per-bin losses are
accumulated across a chunk and a single `backward()` + `optimizer.step()`
runs at each chunk boundary, after which every tensor in `memory_state` is
detached. Within a chunk the TE6 per-node GRU memory (docs/05_ARCHITECTURE.md
§4) is read and written *without* detaching, so gradient genuinely flows
back through the recurrence — the GRU's own weights are trained. Stepping
once per chunk (not per bin) is what makes this safe: no parameter is
mutated in place while a graph that references it is still pending backward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel


@dataclass
class EpochResult:
    loss: float
    n_batches: int
    n_targets: int
    correct: int = 0
    #: [C, C] int64 counts, rows = true class, cols = predicted class. Present
    #: whenever the loss function returned predictions (see `run_epoch`).
    confusion: torch.Tensor | None = None
    #: Class ids excluded from `macro_f1` — the minimum-count classes that
    #: Stage 1 does not train (docs/06_TRAINING.md §4.3). They stay in
    #: `confusion` so they remain visible; they just do not steer selection.
    metric_ignore_class_ids: tuple[int, ...] = ()

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_targets if self.n_targets else 0.0

    @property
    def per_class_f1(self) -> dict[int, float]:
        """F1 per class id, over classes with non-zero support in this pass."""
        if self.confusion is None:
            return {}
        cm = self.confusion.to(torch.float64)
        tp = cm.diagonal()
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        denom = 2 * tp + (predicted - tp) + (support - tp)
        f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
        return {
            i: float(f1[i]) for i in range(cm.shape[0])
            if support[i] > 0 and i not in self.metric_ignore_class_ids
        }

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1 over classes present in this pass.

        This — not `accuracy` — is the metric docs/06_TRAINING.md §5 specifies
        for early stopping, and the difference is not cosmetic on a 34,014:1
        split: a model that predicts only `ddos_hoic` and `benign` scores 0.54
        accuracy and 0.09 macro-F1. Selecting on accuracy actively rewards
        ignoring the tail (docs/BUGS.md #52).

        Classes with no support in this pass are skipped rather than scored 0,
        so the number does not depend on how a rare class happened to fall
        across bins. False positives *onto* an absent class still register, via
        the precision of the classes they were taken from.
        """
        per_class = self.per_class_f1
        if not per_class:
            return 0.0
        return sum(per_class.values()) / len(per_class)


def model_inputs_from_batch(batch: dict, device: torch.device) -> tuple:
    """Convert a build_bin_batch() dict into ArgusModel.forward() positional args."""
    ei_s, ea_s, edt_s = batch["scale_short"]
    ei_m, ea_m, edt_m = batch["scale_mid"]
    ei_l, ea_l, edt_l = batch["scale_long"]
    node_feat = batch["node_feat"].to(device)
    return (
        node_feat,
        ei_s.to(device), ea_s.to(device), edt_s.to(device),
        ei_m.to(device), ea_m.to(device), edt_m.to(device),
        ei_l.to(device), ea_l.to(device), edt_l.to(device),
        batch["target_edge_index"].to(device),
        batch["target_edge_attr"].to(device),
    )


def _detach_memory(memory_state: dict) -> None:
    for k in memory_state:
        memory_state[k] = memory_state[k].detach()


def run_epoch(
    source: AnchorBinGraphSource,
    model: ArgusModel,
    optimizer: torch.optim.Optimizer | None,
    loss_fn,
    device: torch.device,
    train: bool = True,
    grad_clip: float = 1.0,
    memory_state: dict | None = None,
    bptt_chunk: int = 8,
    max_bins: int | None = None,
    channel_penalty: dict | None = None,
    label: str = "",
    log_every_bins: int = 50,
    shuffle_chunks: bool = True,
    shuffle_group_chunks: int = 1,
    num_classes: int | None = None,
    metric_ignore_class_ids: Sequence[int] = (),
) -> EpochResult:
    """Run one epoch (or eval pass) over all anchor bins in `source`.

    `loss_fn(outputs, targets, model)` computes the stage-specific loss and
    returns `(loss_tensor, correct_count)`, optionally with a third element:
    the [B] predicted class ids. When predictions are returned, a full
    confusion matrix is accumulated and `EpochResult.macro_f1` becomes
    available — which is the metric training actually selects on
    (docs/06_TRAINING.md §5). Two-element returns still work; they just leave
    `EpochResult.confusion` at None.

    `num_classes` sizes that confusion matrix; it defaults to
    `model.num_classes`. `metric_ignore_class_ids` are excluded from macro-F1
    (not from the matrix) — the minimum-count classes Stage 1 leaves to
    few-shot registration would otherwise contribute a constant 0.

    `channel_penalty`, if given, is a dict with keys `module` (a
    `ChannelPenaltyLoss` instance), `lambda_ch`, `stride`, `a_idx`, `b_idx`.
    Applied every `stride`-th training batch, with `lambda_ch` scaled by
    `stride` to keep the expected penalty magnitude constant
    (docs/06_TRAINING.md §2.4).

    In training mode, gradient steps happen once per `bptt_chunk` bins on the
    mean of the chunk's per-bin losses; `memory_state` is detached at every
    chunk boundary (truncated BPTT). In eval mode no graph is built at all.

    `shuffle_chunks` (training only) visits the BPTT chunks in random order
    instead of walking the timeline start-to-finish. This is not an
    optimisation — without it training is simply wrong on this data.
    CICIDS2018's bins are time-ordered and each attack campaign occupies a
    contiguous block, so the last 10% of the training timeline is 100% `bot`:
    the final ~900 optimizer steps of every epoch see one class and the model
    ends each epoch as a near-constant `bot` predictor. Measured effect:
    validation accuracy of exactly 0.0000 across the 85% of val bins where
    `bot` is rare (docs/BUGS.md #49).

    Shuffling is safe precisely because chunks are already independent units —
    `memory_state` is detached at every boundary, so no gradient crosses one.
    Bins stay in true temporal order *within* a chunk, so the TE6 recurrence and
    its BPTT window are unchanged; only the order in which those windows are
    visited changes, exactly like shuffling minibatches in ordinary SGD. Memory
    is cleared (not merely detached) at each chunk start when shuffling, since
    carrying a node's state backwards in time across a shuffled boundary would
    be meaningless — the same convention as sampling random segments in
    truncated-BPTT sequence models. Evaluation always runs sequentially with
    continuous memory, so inference-time long-horizon context is unaffected.

    `shuffle_group_chunks` sets how many *consecutive* BPTT chunks stay
    together as one shuffled segment. Shuffling clears TE6 memory at every
    segment boundary, so at the default of 1 a node's memory never spans more
    than `bptt_chunk` bins during training — while evaluation runs sequentially
    and accumulates memory over the whole split. That asymmetry is a real cost
    of the #49 fix: with `bptt_chunk=2` the GRU is trained on 20-second
    histories and evaluated on hours of them. Grouping restores the span
    (`shuffle_group_chunks * bptt_chunk` bins) without changing memory cost,
    because gradient truncation still happens every `bptt_chunk` bins — only
    the memory *values* carry across the inner boundaries. Class mixing is
    unaffected as long as a segment stays well short of a campaign block.

    Note: shuffling assumes node features come from the graph cache (built in
    one sequential pass, so the `is_new_host` feature is already correct).
    Training uncached with shuffling would compute that feature against a
    shuffled "previous window"; pass `shuffle_chunks=False` in that case.
    """
    model.train(train)
    if hasattr(source, "reset_epoch_state"):
        source.reset_epoch_state()
    total_loss = 0.0
    n_batches = 0
    n_targets = 0
    correct = 0
    memory_state = memory_state if memory_state is not None else {}

    n_cls = num_classes if num_classes is not None else getattr(model, "num_classes", None)
    confusion = (
        torch.zeros(n_cls, n_cls, dtype=torch.long, device=device) if n_cls else None
    )

    chunk_loss: torch.Tensor | None = None
    chunk_bins = 0

    def _chunk_step() -> None:
        nonlocal chunk_loss, chunk_bins
        if chunk_loss is None or optimizer is None:
            return
        optimizer.zero_grad()
        (chunk_loss / chunk_bins).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if hasattr(model.head, "prototype_bank"):
            model.head.prototype_bank.post_step_normalize()
        _detach_memory(memory_state)
        chunk_loss = None
        chunk_bins = 0

    bins = source.unique_bins[:max_bins] if max_bins else source.unique_bins
    total = len(bins)
    prefix = f"[{label}] " if label else ""
    mode_str = "train" if train else "eval"

    # Split the timeline into contiguous BPTT chunks, then group consecutive
    # chunks into segments and choose the order in which to visit the segments.
    # Sequential for eval; shuffled for training (see the docstring — a
    # time-ordered stream is also a class-ordered stream here).
    chunks = [bins[s:s + bptt_chunk] for s in range(0, total, bptt_chunk)]
    do_shuffle = bool(train and shuffle_chunks and optimizer is not None)
    if do_shuffle:
        g = max(1, shuffle_group_chunks)
        segments = [chunks[i:i + g] for i in range(0, len(chunks), g)]
        # torch's generator so the epoch order is covered by the RNG state that
        # checkpoint/resume already saves and restores.
        order = torch.randperm(len(segments)).tolist()
        segments = [segments[j] for j in order]
    else:
        segments = [[c] for c in chunks]

    i = -1
    for segment in segments:
        if do_shuffle:
            # A shuffled boundary jumps in time, so a carried-over node state
            # would describe a different moment. Start each segment cold.
            memory_state.clear()
        for chunk in segment:
            for bin_id in chunk:
                i += 1
                batch = source.build_bin_batch(bin_id, f_v=model.f_v)
                if batch is None or batch["n_targets"] == 0:
                    continue
                inputs = model_inputs_from_batch(batch, device)
                target_edge_attr = inputs[-1]
                apply_penalty = (
                    channel_penalty is not None and train and (i % channel_penalty["stride"] == 0)
                )
                if apply_penalty:
                    target_edge_attr.requires_grad_(True)
                targets = batch["target_labels"].to(device)
                node_ids = batch["node_ids"].to(device)

                with torch.set_grad_enabled(train):
                    outputs = model(*inputs, memory=memory_state, node_ids=node_ids)
                    returned = loss_fn(outputs, targets, model)
                    loss, n_correct = returned[0], returned[1]
                    preds = returned[2] if len(returned) > 2 else None
                    if apply_penalty:
                        penalty = channel_penalty["module"](
                            loss, target_edge_attr,
                            channel_penalty["a_idx"], channel_penalty["b_idx"],
                        )
                        loss = loss + channel_penalty["stride"] * channel_penalty["lambda_ch"] * penalty

                if train and optimizer is not None:
                    chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                    chunk_bins += 1

                total_loss += float(loss.item())
                n_batches += 1
                n_targets += len(targets)
                correct += n_correct

                if preds is not None and confusion is not None:
                    with torch.no_grad():
                        t = targets.reshape(-1)
                        p = preds.reshape(-1).to(t.device)
                        # A label outside the vocabulary means the cache and the
                        # class vocab disagree — count it nowhere rather than
                        # letting bincount silently overflow into another cell.
                        ok = (t >= 0) & (t < n_cls) & (p >= 0) & (p < n_cls)
                        if bool(ok.any()):
                            flat = torch.bincount(
                                (t[ok] * n_cls + p[ok]).to(torch.long), minlength=n_cls * n_cls
                            )
                            confusion += flat.view(n_cls, n_cls).to(confusion.device)

                if (i + 1) % log_every_bins == 0:
                    pct = (i + 1) / total * 100
                    avg_loss = total_loss / n_batches
                    acc = correct / n_targets if n_targets else 0.0
                    # Running macro-F1 alongside accuracy: on this split the two
                    # diverge hard, and accuracy alone hid a total tail collapse
                    # for three runs (docs/BUGS.md #49, #52).
                    f1_str = ""
                    if confusion is not None:
                        running = EpochResult(0.0, 0, 0, confusion=confusion.cpu(),
                                              metric_ignore_class_ids=tuple(metric_ignore_class_ids))
                        f1_str = f"  macroF1={running.macro_f1:.4f}"
                    print(f"  {prefix}{mode_str} {i + 1}/{total} bins ({pct:.0f}%)  "
                          f"loss={avg_loss:.4f}  acc={acc:.4f}{f1_str}", flush=True)

            # Step at the chunk boundary — driven by the chunk structure
            # itself, so a chunk whose trailing bins were empty still steps on
            # what it did see. Gradient truncation therefore still happens
            # every `bptt_chunk` bins even when a segment spans several chunks;
            # only the memory *values* survive across the inner boundaries.
            if train and optimizer is not None:
                _chunk_step()

    if train and optimizer is not None:
        _chunk_step()

    return EpochResult(
        loss=total_loss / max(n_batches, 1),
        n_batches=n_batches,
        n_targets=n_targets,
        correct=correct,
        confusion=confusion.cpu() if confusion is not None else None,
        metric_ignore_class_ids=tuple(int(c) for c in metric_ignore_class_ids),
    )
