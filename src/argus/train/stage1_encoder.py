"""Stage 1: encoder + prototype geometry training.

See docs/06_TRAINING.md §1, §2.1, §2.2, §2.4, §2.7.
"""

from __future__ import annotations

import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.losses.am_softmax import AMSoftmaxLoss, CompactnessLoss
from argus.losses.channel_penalty import ChannelPenaltyLoss
from argus.models.argus import ArgusModel
from argus.train.loop import EpochResult, run_epoch


def make_stage1_loss_fn(
    am_loss: AMSoftmaxLoss,
    compact_loss: CompactnessLoss,
    lambda_cmp: float,
    lambda_div: float,
    epoch: int,
):
    """Build the Stage-1 loss closure: L_am + lambda_cmp*L_compact + lambda_div*L_div."""

    def loss_fn(outputs: dict, targets: torch.Tensor, model: ArgusModel) -> tuple[torch.Tensor, int]:
        cos_c = outputs["cos_c"]
        z = outputs["z"]
        prototypes_per_class = _class_prototype_centroid(model.head.prototype_bank, cos_c.shape[1])
        loss = am_loss(cos_c, targets, epoch=epoch)
        loss = loss + lambda_cmp * compact_loss(z, prototypes_per_class, targets)
        loss = loss + lambda_div * model.head.prototype_bank.diversity_loss()
        pred = cos_c.argmax(dim=1)
        correct = int((pred == targets).sum().item())
        return loss, correct

    return loss_fn


def _class_prototype_centroid(prototype_bank, num_classes: int) -> torch.Tensor:
    """Return a [C, d_z] tensor of mean-normalised sub-prototype centroids per class,
    for use as the compactness-loss target (docs/06_TRAINING.md §2.2 uses the nearest
    prototype in practice; the mean centroid is a stable, differentiable proxy).
    """
    import torch.nn.functional as F

    d_z = prototype_bank.bank.shape[1]
    centroids = torch.zeros(num_classes, d_z, device=prototype_bank.bank.device)
    for ci in range(num_classes):
        mask = torch.tensor(
            [i == ci for i in prototype_bank.class_of], device=prototype_bank.bank.device
        )
        if mask.any():
            centroids[ci] = prototype_bank.bank[mask].mean(dim=0)
    return F.normalize(centroids, dim=-1)


def train_stage1(
    model: ArgusModel,
    train_source: AnchorBinGraphSource,
    val_source: AnchorBinGraphSource,
    cfg,
    device: torch.device,
    max_bins: int | None = None,
) -> dict:
    """Full Stage-1 training with early stopping on validation macro-F1 proxy (accuracy)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.stage1_lr, weight_decay=cfg.regularisation.weight_decay
    )
    am_loss = AMSoftmaxLoss(
        margin=cfg.head.am_softmax_margin,
        scale_start=cfg.head.am_softmax_scale_start,
        scale_final=cfg.head.am_softmax_scale_final,
        warmup_epochs=cfg.head.am_softmax_warmup_epochs,
        label_smoothing=cfg.regularisation.label_smoothing,
    )
    compact_loss = CompactnessLoss()
    channel_penalty_loss = ChannelPenaltyLoss(rho=cfg.loss.channel_ratio_tolerance)
    channel_penalty_cfg = {
        "module": channel_penalty_loss,
        "lambda_ch": cfg.loss.lambda_channel,
        "stride": cfg.loss.channel_penalty_stride,
        "a_idx": model.encoder.a_idx,
        "b_idx": model.encoder.b_idx,
    }

    best_val = -1.0
    patience_left = cfg.train.stage1_patience
    history = []

    for epoch in range(cfg.train.stage1_epochs):
        loss_fn = make_stage1_loss_fn(
            am_loss, compact_loss, cfg.loss.lambda_compact, cfg.loss.lambda_div, epoch
        )
        train_result = run_epoch(
            train_source, model, optimizer, loss_fn, device, train=True,
            grad_clip=cfg.train.grad_clip, bptt_chunk=cfg.train.bptt_chunk, max_bins=max_bins,
            channel_penalty=channel_penalty_cfg,
        )
        val_result = run_epoch(
            val_source, model, None, loss_fn, device, train=False, max_bins=max_bins,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_result.loss, "train_acc": train_result.accuracy,
             "val_loss": val_result.loss, "val_acc": val_result.accuracy}
        )
        if val_result.accuracy > best_val:
            best_val = val_result.accuracy
            patience_left = cfg.train.stage1_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return {"history": history, "best_val_acc": best_val}
