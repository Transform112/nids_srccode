"""Stage 1: encoder + prototype geometry training.

See docs/06_TRAINING.md §1, §2.1, §2.2, §2.4, §2.7.
"""

from __future__ import annotations

from pathlib import Path

import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.losses.am_softmax import AMSoftmaxLoss, CompactnessLoss
from argus.losses.channel_penalty import ChannelPenaltyLoss
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint, save_checkpoint
from argus.train.gates import (
    gate_g1_encoder_learning,
    gate_g2_prototype_collapse,
    gate_g2b_subprototype_collapse,
    gate_g5_channel_reliance,
    gate_g6_numerical_health,
    gate_g7_overfitting,
    prototype_gate_stats,
    record_gate,
)
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
    run_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    """Full Stage-1 training with early stopping on validation macro-F1 proxy (accuracy).

    If `run_dir` is given, a resumable checkpoint (`stage1_ckpt_last.pt`, real
    optimizer + RNG + early-stop state) is written every
    `cfg.train.checkpoint_every_epoch` epochs and the best-val-metric weights
    are kept in `stage1_ckpt_best.pt`; `resume=True` continues from the last
    checkpoint if one exists (docs/06_TRAINING.md §6). On return the model
    carries the *best* epoch's weights, not the last epoch's — early stopping
    with patience deliberately trains past the best epoch, so returning the
    final weights would ship a model known to be worse.

    TE6 memory is not part of the checkpoint: it re-initialises to zeros at
    split start (docs/05_ARCHITECTURE.md §4), which happens at the top of
    every epoch, so an epoch-boundary checkpoint has no memory state to carry.
    """
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
    best_epoch = -1
    patience_left = cfg.train.stage1_patience
    history = []
    start_epoch = 0

    ckpt_last = run_dir / "stage1_ckpt_last.pt" if run_dir is not None else None
    ckpt_best = run_dir / "stage1_ckpt_best.pt" if run_dir is not None else None
    if resume and ckpt_last is not None and ckpt_last.exists():
        state = load_checkpoint(ckpt_last, model, optimizer, map_location=str(device))
        extra = state.get("extra", {})
        start_epoch = state["epoch"] + 1
        best_val = extra.get("best_val", -1.0)
        best_epoch = extra.get("best_epoch", -1)
        patience_left = extra.get("patience_left", cfg.train.stage1_patience)
        history = extra.get("history", [])
        print(f"[04] Resumed from {ckpt_last} — continuing at epoch {start_epoch} "
              f"(best={best_val:.4f} @ epoch {best_epoch}, patience={patience_left})",
              flush=True)

    ckpt_every_epoch = bool(getattr(cfg.train, "checkpoint_every_epoch", True))

    for epoch in range(start_epoch, cfg.train.stage1_epochs):
        print(f"[04] === Epoch {epoch}/{cfg.train.stage1_epochs} (best={best_val:.4f}) ===",
              flush=True)
        loss_fn = make_stage1_loss_fn(
            am_loss, compact_loss, cfg.loss.lambda_compact, cfg.loss.lambda_div, epoch
        )
        train_result = run_epoch(
            train_source, model, optimizer, loss_fn, device, train=True,
            grad_clip=cfg.train.grad_clip, bptt_chunk=cfg.train.bptt_chunk, max_bins=max_bins,
            channel_penalty=channel_penalty_cfg,
            label=f"S1 e{epoch}",
        )
        val_result = run_epoch(
            val_source, model, None, loss_fn, device, train=False, max_bins=max_bins,
            label=f"S1 e{epoch} val",
        )
        history.append(
            {"epoch": epoch, "train_loss": train_result.loss, "train_acc": train_result.accuracy,
             "val_loss": val_result.loss, "val_acc": val_result.accuracy}
        )

        # Gates (docs/06_TRAINING.md §8). G1/G7 use the val-accuracy proxy the
        # rest of this trainer already early-stops on (macro-F1 per doc — a
        # disclosed proxy, see docs/BUGS.md). G6 failure hard-stops: a NaN/Inf
        # loss means every further step is wasted compute.
        gates_report = run_dir / "gates_report.json" if run_dir is not None else None
        g6_ok, g6_msg = gate_g6_numerical_health(torch.tensor(train_result.loss))
        if not g6_ok:
            record_gate(gates_report, "G6", g6_ok, g6_msg + f" (epoch {epoch})")
            raise RuntimeError(f"Gate G6 failed at epoch {epoch}: non-finite training loss")
        if epoch == 5:
            g1_ok, g1_msg = gate_g1_encoder_learning(
                val_result.accuracy, getattr(cfg.gates, "g1_min_val_f1_epoch5", 0.5))
            record_gate(gates_report, "G1", g1_ok, g1_msg + " (val-acc proxy)")
        g7_ok, g7_msg = gate_g7_overfitting(
            train_result.accuracy, val_result.accuracy,
            getattr(cfg.gates, "g7_max_train_val_gap", 0.10))
        if not g7_ok:
            record_gate(gates_report, "G7", g7_ok, g7_msg + f" (epoch {epoch}, acc proxy)")
        if channel_penalty_loss.last_ratio is not None:
            g5_ok, g5_msg = gate_g5_channel_reliance(
                channel_penalty_loss.last_ratio, getattr(cfg.gates, "g5_max_channel_ratio", 0.8))
            if not g5_ok:
                record_gate(gates_report, "G5", g5_ok, g5_msg + f" (epoch {epoch})")

        improved = val_result.accuracy > best_val
        print(f"[04] Epoch {epoch:2d}/{cfg.train.stage1_epochs}  "
              f"train={train_result.loss:.4f}  val={val_result.loss:.4f}  "
              f"acc={val_result.accuracy:.4f}  "
              f"{'NEW BEST' if improved else f'patience={patience_left}'}",
              flush=True)
        if improved:
            best_val = val_result.accuracy
            best_epoch = epoch
            patience_left = cfg.train.stage1_patience
            if ckpt_best is not None:
                save_checkpoint(ckpt_best, model, optimizer, epoch=epoch,
                                extra={"best_val": best_val, "best_epoch": best_epoch})
        else:
            patience_left -= 1

        if ckpt_last is not None and (ckpt_every_epoch or patience_left <= 0
                                      or epoch == cfg.train.stage1_epochs - 1):
            save_checkpoint(
                ckpt_last, model, optimizer, epoch=epoch,
                extra={"best_val": best_val, "best_epoch": best_epoch,
                       "patience_left": patience_left, "history": history, "stage": "stage1"},
            )

        if patience_left <= 0:
            print(f"[04] Early stop at epoch {epoch} — val_acc={best_val:.4f}")
            break

    if ckpt_best is not None and ckpt_best.exists() and best_epoch >= 0:
        state = torch.load(ckpt_best, map_location=str(device), weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"[04] Restored best weights (epoch {best_epoch}, val_acc={best_val:.4f})")

    # Post-training prototype geometry gates (docs/06_TRAINING.md §8).
    if hasattr(model.head, "prototype_bank"):
        gates_report = run_dir / "gates_report.json" if run_dir is not None else None
        inter, intra = prototype_gate_stats(model.head.prototype_bank)
        g2_ok, g2_msg = gate_g2_prototype_collapse(
            inter, getattr(cfg.gates, "g2_max_proto_cosine", 0.8))
        record_gate(gates_report, "G2", g2_ok, g2_msg)
        g2b_ok, g2b_msg = gate_g2b_subprototype_collapse(intra)
        record_gate(gates_report, "G2b", g2b_ok, g2b_msg)

    return {"history": history, "best_val_acc": best_val, "best_epoch": best_epoch,
            "optimizer": optimizer}
