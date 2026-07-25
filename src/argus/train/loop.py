"""Shared train/eval loop over anchor-bin graph batches.

Processes one anchor bin at a time (see docs/04_GRAPH_CONSTRUCTION.md §5 for the
full multi-bin BPTT batching spec; this loop carries the node-memory dict
forward across bins and detaches it every `bptt_chunk` bins, which is the core
of that spec applied one bin at a time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    n_targets = 0
    correct = 0
    memory_state = memory_state if memory_state is not None else {}

    bins = source.unique_bins[:max_bins] if max_bins else source.unique_bins
    for i, bin_id in enumerate(bins):
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

        with torch.set_grad_enabled(train):
            outputs = model(*inputs)
            loss, n_correct = loss_fn(outputs, targets, model)
            if apply_penalty:
                penalty = channel_penalty["module"](
                    loss, target_edge_attr, channel_penalty["a_idx"], channel_penalty["b_idx"]
                )
                loss = loss + channel_penalty["stride"] * channel_penalty["lambda_ch"] * penalty

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if model.head.__class__.__name__ == "EPCHead":
                model.head.prototype_bank.post_step_normalize()

        if (i + 1) % bptt_chunk == 0:
            memory_state.clear()  # detach: our loop doesn't carry raw tensors, so clear is safe

        total_loss += float(loss.item())
        n_batches += 1
        n_targets += len(targets)
        correct += n_correct

    return EpochResult(loss=total_loss / max(n_batches, 1), n_batches=n_batches, n_targets=n_targets, correct=correct)
