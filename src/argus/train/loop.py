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

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_targets if self.n_targets else 0.0


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
) -> EpochResult:
    """Run one epoch (or eval pass) over all anchor bins in `source`.

    `loss_fn(outputs, targets, model) -> (loss_tensor, correct_count)` computes
    the stage-specific loss and returns the number of correctly classified
    targets for accuracy tracking.

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

    # Split the timeline into contiguous BPTT chunks, then choose the order in
    # which to visit them. Sequential for eval; shuffled for training (see the
    # docstring — a time-ordered stream is also a class-ordered stream here).
    chunks = [bins[s:s + bptt_chunk] for s in range(0, total, bptt_chunk)]
    do_shuffle = bool(train and shuffle_chunks and optimizer is not None)
    if do_shuffle:
        # torch's generator so the epoch order is covered by the RNG state that
        # checkpoint/resume already saves and restores.
        order = torch.randperm(len(chunks)).tolist()
        chunks = [chunks[j] for j in order]

    i = -1
    for chunk in chunks:
        if do_shuffle:
            # A shuffled boundary jumps in time, so a carried-over node state
            # would describe a different moment. Start each chunk cold.
            memory_state.clear()
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
                loss, n_correct = loss_fn(outputs, targets, model)
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

            if (i + 1) % log_every_bins == 0:
                pct = (i + 1) / total * 100
                avg_loss = total_loss / n_batches
                acc = correct / n_targets if n_targets else 0.0
                print(f"  {prefix}{mode_str} {i + 1}/{total} bins ({pct:.0f}%)  "
                      f"loss={avg_loss:.4f}  acc={acc:.4f}", flush=True)

        # Step at the chunk boundary — driven by the chunk structure itself, so
        # a chunk whose trailing bins were empty still steps on what it did see.
        if train and optimizer is not None:
            _chunk_step()

    if train and optimizer is not None:
        _chunk_step()

    return EpochResult(loss=total_loss / max(n_batches, 1), n_batches=n_batches, n_targets=n_targets, correct=correct)
