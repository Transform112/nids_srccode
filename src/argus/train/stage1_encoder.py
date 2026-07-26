"""Stage 1: encoder + prototype geometry training.

See docs/06_TRAINING.md §1, §2.1, §2.2, §2.4, §2.7.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from argus.graph.batching import AnchorBinGraphSource
from argus.losses.am_softmax import AMSoftmaxLoss, CompactnessLoss
from argus.losses.channel_penalty import ChannelPenaltyLoss
from argus.losses.class_balance import (
    balanced_target_indices,
    effective_number_weights,
    min_count_excluded_classes,
)
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint, save_checkpoint
from argus.train.gates import (
    gate_g1_encoder_learning,
    gate_g2_prototype_collapse,
    gate_g2b_subprototype_collapse,
    gate_g5_channel_reliance,
    gate_g6_numerical_health,
    gate_g7_overfitting,
    gate_g8_tail_collapse,
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
    class_weights: torch.Tensor | None = None,
    n_per_class: int | None = None,
    exclude_class_ids: Sequence[int] = (),
):
    """Build the Stage-1 loss closure: L_am + lambda_cmp*L_compact + lambda_div*L_div.

    The three class-imbalance mechanisms of docs/06_TRAINING.md §4 enter here:

    - `class_weights` — effective-number weights passed through to AM-Softmax.
    - `n_per_class` — the loss is computed on a class-balanced subsample of the
      bin's targets. Every target is still embedded and still scored for the
      metrics; capping only changes which ones produce gradient, which is what
      "decouples graph density from loss balance" means.
    - `exclude_class_ids` — minimum-count classes, dropped from the loss so no
      gradient ever pulls a prototype toward a 14-sample class.

    Pass `n_per_class=None` and `exclude_class_ids=()` for evaluation passes:
    subsampling an eval pass would throw away the very tail targets the metric
    exists to measure.

    Returns `(loss, correct, predictions)`; `run_epoch` uses the predictions to
    accumulate the confusion matrix behind `EpochResult.macro_f1`.
    """

    def loss_fn(
        outputs: dict, targets: torch.Tensor, model: ArgusModel
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        cos_c = outputs["cos_c"]
        z = outputs["z"]
        # Accuracy and the confusion matrix are computed over *all* targets,
        # never the subsample — otherwise the reported metric would measure a
        # distribution the model is not evaluated on.
        pred = cos_c.argmax(dim=1)
        correct = int((pred == targets).sum().item())

        # Diversity is a property of the bank alone, so it applies to every bin
        # regardless of which targets survive the class balancing.
        loss = lambda_div * model.head.prototype_bank.diversity_loss()

        keep = balanced_target_indices(targets, n_per_class, exclude_class_ids)
        if keep.numel() == 0:
            # Every flow in this bin belongs to an excluded class. Keep the
            # graph connected to cos_c with a zero-valued term so the chunk's
            # backward still runs on whatever its other bins contributed.
            return loss + cos_c.sum() * 0.0, correct, pred

        cos_sel, z_sel, tgt_sel = cos_c[keep], z[keep], targets[keep]
        prototypes_per_class = _class_prototype_centroid(model.head.prototype_bank, cos_c.shape[1])
        loss = loss + am_loss(cos_sel, tgt_sel, class_weights=class_weights, epoch=epoch)
        loss = loss + lambda_cmp * compact_loss(z_sel, prototypes_per_class, tgt_sel)
        return loss, correct, pred

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
    class_counts: dict[str, int] | None = None,
    weight_counts: dict[str, int] | None = None,
) -> dict:
    """Full Stage-1 training with early stopping on validation macro-F1.

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

    `class_counts` (name -> training rows) enables the class-imbalance
    mechanisms of docs/06_TRAINING.md §4: effective-number loss weights, the
    class-balanced target subsample, and the minimum-count guard. Without it
    training falls back to the unweighted, unbalanced behaviour — which on
    CICIDS2018's 34,014:1 split is not a neutral default (docs/BUGS.md #50).

    `weight_counts` overrides the counts the *weights* are derived from, while
    `class_counts` still drives the minimum-count guard. Pass the post-cap
    counts from `losses.class_balance.effective_target_counts` here: the two
    differ by up to 7.6x per class once `n_per_class` capping is active
    (docs/BUGS.md #51). Defaults to `class_counts`.
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

    # --- Class imbalance (docs/06_TRAINING.md §4) -------------------------
    class_weights: torch.Tensor | None = None
    excluded_ids: tuple[int, ...] = ()
    n_per_class = getattr(cfg.train, "n_per_class", None)
    if class_counts is not None:
        nu = getattr(cfg.loss, "effective_number_nu", 0.0)
        excluded_ids = tuple(min_count_excluded_classes(
            model.class_names, class_counts,
            getattr(cfg.train, "min_count_for_prototype", 0),
        ))
        if nu:
            counts_for_weights = dict(weight_counts or class_counts)
            # An excluded class never appears as a target, so leaving its
            # (large) weight in the vector only distorts the normalisation of
            # the classes that do train.
            for i in excluded_ids:
                counts_for_weights[model.class_names[i]] = 0
            class_weights = effective_number_weights(
                model.class_names, counts_for_weights, nu=nu, device=device
            )
        # Report the spread over classes that actually train: excluded and
        # absent classes carry weight 0 by design, and including them would
        # make the ratio meaninglessly large.
        if class_weights is not None:
            live = class_weights[class_weights > 0]
            w_lo, w_hi = float(live.min()), float(live.max())
            w_desc = f"nu={nu} range=[{w_lo:.3f}, {w_hi:.3f}] ({w_hi / w_lo:.1f}x spread)"
        else:
            w_desc = "disabled (nu=0)"
        print(f"[04] Imbalance: n_per_class={n_per_class}  effective-number weights {w_desc}  "
              f"min-count excluded={[model.class_names[i] for i in excluded_ids]}",
              flush=True)
    else:
        print("[04] Imbalance: class_counts not supplied — training UNWEIGHTED "
              "and unbalanced (docs/06_TRAINING.md §4 disabled)", flush=True)

    best_val = -1.0
    best_epoch = -1
    best_per_class_f1: dict[int, float] = {}
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
        best_per_class_f1 = {int(k): v for k, v in extra.get("best_per_class_f1", {}).items()}
        patience_left = extra.get("patience_left", cfg.train.stage1_patience)
        history = extra.get("history", [])
        print(f"[04] Resumed from {ckpt_last} — continuing at epoch {start_epoch} "
              f"(best={best_val:.4f} @ epoch {best_epoch}, patience={patience_left})",
              flush=True)

    ckpt_every_epoch = bool(getattr(cfg.train, "checkpoint_every_epoch", True))

    for epoch in range(start_epoch, cfg.train.stage1_epochs):
        print(f"[04] === Epoch {epoch}/{cfg.train.stage1_epochs} (best={best_val:.4f}) ===",
              flush=True)
        # Training balances and reweights the loss; evaluation must not — an
        # eval pass that subsampled would discard the tail targets macro-F1
        # exists to measure.
        train_loss_fn = make_stage1_loss_fn(
            am_loss, compact_loss, cfg.loss.lambda_compact, cfg.loss.lambda_div, epoch,
            class_weights=class_weights, n_per_class=n_per_class,
            exclude_class_ids=excluded_ids,
        )
        eval_loss_fn = make_stage1_loss_fn(
            am_loss, compact_loss, cfg.loss.lambda_compact, cfg.loss.lambda_div, epoch,
            class_weights=class_weights,
        )
        train_result = run_epoch(
            train_source, model, optimizer, train_loss_fn, device, train=True,
            grad_clip=cfg.train.grad_clip, bptt_chunk=cfg.train.bptt_chunk, max_bins=max_bins,
            channel_penalty=channel_penalty_cfg,
            shuffle_group_chunks=getattr(cfg.train, "shuffle_group_chunks", 1),
            label=f"S1 e{epoch}", metric_ignore_class_ids=excluded_ids,
        )
        val_result = run_epoch(
            val_source, model, None, eval_loss_fn, device, train=False, max_bins=max_bins,
            label=f"S1 e{epoch} val", metric_ignore_class_ids=excluded_ids,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_result.loss, "train_acc": train_result.accuracy,
             "train_macro_f1": train_result.macro_f1,
             "val_loss": val_result.loss, "val_acc": val_result.accuracy,
             "val_macro_f1": val_result.macro_f1,
             "val_per_class_f1": {model.class_names[i]: f1
                                  for i, f1 in val_result.per_class_f1.items()}}
        )

        # Gates (docs/06_TRAINING.md §8). G1/G7 now use real macro-F1 rather
        # than the accuracy proxy they used to: on this split accuracy passes
        # both gates while the model ignores every tail class (docs/BUGS.md
        # #52). G6 failure hard-stops: a NaN/Inf loss means every further step
        # is wasted compute.
        gates_report = run_dir / "gates_report.json" if run_dir is not None else None
        g6_ok, g6_msg = gate_g6_numerical_health(torch.tensor(train_result.loss))
        if not g6_ok:
            record_gate(gates_report, "G6", g6_ok, g6_msg + f" (epoch {epoch})")
            raise RuntimeError(f"Gate G6 failed at epoch {epoch}: non-finite training loss")
        if epoch == 5:
            g1_ok, g1_msg = gate_g1_encoder_learning(
                val_result.macro_f1, getattr(cfg.gates, "g1_min_val_f1_epoch5", 0.5))
            record_gate(gates_report, "G1", g1_ok, g1_msg)
        g7_ok, g7_msg = gate_g7_overfitting(
            train_result.macro_f1, val_result.macro_f1,
            getattr(cfg.gates, "g7_max_train_val_gap", 0.10))
        if not g7_ok:
            record_gate(gates_report, "G7", g7_ok, g7_msg + f" (epoch {epoch})")
        if channel_penalty_loss.last_ratio is not None:
            g5_ok, g5_msg = gate_g5_channel_reliance(
                channel_penalty_loss.last_ratio, getattr(cfg.gates, "g5_max_channel_ratio", 0.8))
            if not g5_ok:
                record_gate(gates_report, "G5", g5_ok, g5_msg + f" (epoch {epoch})")

        # Selection metric: val macro-F1 (docs/06_TRAINING.md §5).
        improved = val_result.macro_f1 > best_val
        weakest = sorted(val_result.per_class_f1.items(), key=lambda kv: kv[1])[:3]
        print(f"[04] Epoch {epoch:2d}/{cfg.train.stage1_epochs}  "
              f"train={train_result.loss:.4f}  val={val_result.loss:.4f}  "
              f"macroF1={val_result.macro_f1:.4f}  acc={val_result.accuracy:.4f}  "
              f"{'NEW BEST' if improved else f'patience={patience_left}'}",
              flush=True)
        if weakest:
            print("[04]   weakest classes: "
                  + ", ".join(f"{model.class_names[i]}={f1:.3f}" for i, f1 in weakest),
                  flush=True)
        if improved:
            best_val = val_result.macro_f1
            best_epoch = epoch
            best_per_class_f1 = dict(val_result.per_class_f1)
            patience_left = cfg.train.stage1_patience
            if ckpt_best is not None:
                save_checkpoint(ckpt_best, model, optimizer, epoch=epoch,
                                extra={"best_val": best_val, "best_epoch": best_epoch,
                                       "best_per_class_f1": best_per_class_f1})
        else:
            patience_left -= 1

        if ckpt_last is not None and (ckpt_every_epoch or patience_left <= 0
                                      or epoch == cfg.train.stage1_epochs - 1):
            save_checkpoint(
                ckpt_last, model, optimizer, epoch=epoch,
                extra={"best_val": best_val, "best_epoch": best_epoch,
                       "best_per_class_f1": best_per_class_f1,
                       "patience_left": patience_left, "history": history, "stage": "stage1"},
            )

        if patience_left <= 0:
            print(f"[04] Early stop at epoch {epoch} — val_macro_f1={best_val:.4f}")
            break

    if ckpt_best is not None and ckpt_best.exists() and best_epoch >= 0:
        state = torch.load(ckpt_best, map_location=str(device), weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"[04] Restored best weights (epoch {best_epoch}, val_macro_f1={best_val:.4f})")

    gates_report = run_dir / "gates_report.json" if run_dir is not None else None

    # G8 tail collapse, on the selected epoch. G0 cannot see this failure —
    # it trains on a class-balanced subset, so it is structurally blind to a
    # model that has simply stopped predicting the tail (docs/BUGS.md #49).
    if best_per_class_f1:
        g8_ok, g8_msg = gate_g8_tail_collapse(
            {model.class_names[i]: f1 for i, f1 in best_per_class_f1.items()},
            max_collapsed=getattr(cfg.gates, "g8_max_collapsed_classes", 0),
        )
        record_gate(gates_report, "G8", g8_ok, g8_msg + f" (epoch {best_epoch})")

    # Post-training prototype geometry gates (docs/06_TRAINING.md §8).
    if hasattr(model.head, "prototype_bank"):
        inter, intra = prototype_gate_stats(model.head.prototype_bank)
        g2_ok, g2_msg = gate_g2_prototype_collapse(
            inter, getattr(cfg.gates, "g2_max_proto_cosine", 0.8))
        record_gate(gates_report, "G2", g2_ok, g2_msg)
        g2b_ok, g2b_msg = gate_g2b_subprototype_collapse(intra)
        record_gate(gates_report, "G2b", g2b_ok, g2b_msg)

    return {"history": history, "best_val_macro_f1": best_val,
            "best_val_per_class_f1": {model.class_names[i]: f1
                                      for i, f1 in best_per_class_f1.items()},
            "excluded_classes": [model.class_names[i] for i in excluded_ids],
            "best_epoch": best_epoch, "optimizer": optimizer}
