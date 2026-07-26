"""Stage 2: evidential calibration training.

See docs/06_TRAINING.md §1, §2.3, §2.5, §2.6. Encoder is frozen; only the
embedding projection, prototype bank, and evidential temperature train.

Scope note: this implementation includes the embedding-space mixup synthetic
unknown generator (§3.1). The structural pseudo-unknown generator (§3.2)
requires rebuilding graph context at a different host and re-running the
encoder; it is a natural extension once the graph cache (P1/P3) is wired to
support arbitrary context rebuilding, and is intentionally out of scope here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from argus.graph.batching import AnchorBinGraphSource
from argus.losses.evidential import EvidentialLoss
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint, save_checkpoint
from argus.train.gates import (
    gate_g3_evidence_collapse,
    gate_g4_unknown_carving,
    gate_g6_numerical_health,
    record_gate,
)
from argus.train.loop import model_inputs_from_batch, run_epoch


def embedding_mixup_unknowns(
    z: torch.Tensor,
    targets: torch.Tensor,
    prototype_bank,
    mu_low: float = 0.35,
    mu_high: float = 0.65,
    cos_reject: float = 0.90,
    n_synth: int | None = None,
) -> torch.Tensor:
    """Generate synthetic-unknown embeddings via mid-band interpolation.

    See docs/06_TRAINING.md §3.1. Rejects interpolations landing inside a real
    class cone (cosine to nearest prototype > cos_reject).
    """
    device = z.device
    bsz = z.shape[0]
    n_synth = n_synth or max(bsz // 5, 1)  # gamma ~= 0.2 of batch size
    if bsz < 2:
        return torch.zeros(0, z.shape[1], device=device)

    idx_a = torch.randint(0, bsz, (n_synth,), device=device)
    idx_b = torch.randint(0, bsz, (n_synth,), device=device)
    mu = torch.empty(n_synth, 1, device=device).uniform_(mu_low, mu_high)
    z_synth = F.normalize(mu * z[idx_a] + (1 - mu) * z[idx_b], dim=-1)

    cos_to_protos = torch.matmul(z_synth, prototype_bank.bank.T)
    max_cos = cos_to_protos.max(dim=1).values
    keep = max_cos <= cos_reject
    return z_synth[keep]


def evaluate_vacuity(
    model: ArgusModel,
    val_source: AnchorBinGraphSource,
    cfg,
    device: torch.device,
    max_bins: int | None = None,
) -> tuple[float, float]:
    """Mean vacuity on known val flows and on synthetic-mixup unknowns.

    Inputs for gates G3/G4 (docs/06_TRAINING.md §8). Synthetic-unknown vacuity
    is computed through the same evidence path the Stage-2 unknown loss uses.
    """
    model.eval()
    known_vac: list[float] = []
    synth_vac: list[float] = []
    head = model.head
    bins = val_source.unique_bins[:max_bins] if max_bins else val_source.unique_bins
    with torch.no_grad():
        for bin_id in bins:
            batch = val_source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            inputs = model_inputs_from_batch(batch, device)
            outputs = model(*inputs)
            known_vac.extend(outputs["vacuity"].flatten().tolist())

            synth_z = embedding_mixup_unknowns(
                outputs["z"], batch["target_labels"].to(device), head.prototype_bank,
                mu_low=cfg.loss.mixup_mu_low, mu_high=cfg.loss.mixup_mu_high,
                cos_reject=cfg.loss.cos_reject,
            )
            if synth_z.shape[0] > 0:
                cos_c = head.prototype_bank.cosine_to_classes(synth_z)
                d_c = 1.0 - cos_c
                log_e = torch.clamp(
                    -(d_c - head.margin_m) / head.tau.clamp_min(head.tau_min),
                    -head.log_evidence_clamp, head.log_evidence_clamp,
                )
                e = torch.exp(log_e)
                c = e.shape[1]
                s = e.sum(dim=1) + c
                synth_vac.extend((c / s).flatten().tolist())

    mean_known = float(sum(known_vac) / len(known_vac)) if known_vac else 0.0
    mean_synth = float(sum(synth_vac) / len(synth_vac)) if synth_vac else 0.0
    return mean_known, mean_synth


def make_stage2_loss_fn(evid_loss: EvidentialLoss, cfg, epoch: int):
    kl_weight = min(1.0, epoch / max(cfg.loss.kl_anneal_epochs, 1)) * cfg.loss.lambda_kl_max

    def loss_fn(
        outputs: dict, targets: torch.Tensor, model: ArgusModel
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        head = model.head
        alpha = outputs["alpha"]
        loss = evid_loss(alpha, targets, kl_anneal_weight=kl_weight)

        synth_z = embedding_mixup_unknowns(
            outputs["z"], targets, head.prototype_bank,
            mu_low=cfg.loss.mixup_mu_low, mu_high=cfg.loss.mixup_mu_high,
            cos_reject=cfg.loss.cos_reject,
        )
        if synth_z.shape[0] > 0:
            cos_c = head.prototype_bank.cosine_to_classes(synth_z)
            d_c = 1.0 - cos_c
            log_e = torch.clamp(
                -(d_c - head.margin_m) / head.tau.clamp_min(head.tau_min),
                -head.log_evidence_clamp, head.log_evidence_clamp,
            )
            e = torch.exp(log_e)
            total = e.sum(dim=1)
            unk_loss = (total + F.relu(total - 0.01)).mean()
            loss = loss + cfg.loss.lambda_unknown * unk_loss

        pred = outputs["p_hat"].argmax(dim=1)
        correct = int((pred == targets).sum().item())
        return loss, correct, pred

    return loss_fn


def train_stage2(
    model: ArgusModel,
    train_source: AnchorBinGraphSource,
    val_source: AnchorBinGraphSource,
    cfg,
    device: torch.device,
    max_bins: int | None = None,
    run_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    """Full Stage-2 training: encoder frozen, head parameters trained at low LR.

    Checkpoint/resume semantics mirror `train_stage1` (see its docstring):
    `stage2_ckpt_last.pt` every `checkpoint_every_epoch` epochs,
    `stage2_ckpt_best.pt` on val improvement, best weights restored on return.
    """
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    if model.memory is not None:
        for p in model.memory.parameters():
            p.requires_grad_(False)
    for p in model.fusion.parameters():
        p.requires_grad_(False)

    # tau_hat is excluded: it must follow anneal_tau()'s mandated schedule
    # exactly (docs/06_TRAINING.md §2.3); letting AdamW also gradient-step it
    # every batch fights that schedule between the once-per-epoch resets.
    head_params = [
        p for name, p in model.head.named_parameters()
        if p.requires_grad and name != "tau_hat"
    ]
    optimizer = torch.optim.AdamW(
        head_params, lr=cfg.train.stage2_lr, weight_decay=cfg.regularisation.weight_decay
    )
    evid_loss = EvidentialLoss()

    best_val = -1.0
    best_epoch = -1
    patience_left = cfg.train.stage2_patience
    history = []
    start_epoch = 0

    ckpt_last = run_dir / "stage2_ckpt_last.pt" if run_dir is not None else None
    ckpt_best = run_dir / "stage2_ckpt_best.pt" if run_dir is not None else None
    if resume and ckpt_last is not None and ckpt_last.exists():
        state = load_checkpoint(ckpt_last, model, optimizer, map_location=str(device))
        extra = state.get("extra", {})
        start_epoch = state["epoch"] + 1
        best_val = extra.get("best_val", -1.0)
        best_epoch = extra.get("best_epoch", -1)
        patience_left = extra.get("patience_left", cfg.train.stage2_patience)
        history = extra.get("history", [])
        print(f"[05] Resumed from {ckpt_last} — continuing at epoch {start_epoch} "
              f"(best={best_val:.4f} @ epoch {best_epoch}, patience={patience_left})",
              flush=True)

    ckpt_every_epoch = bool(getattr(cfg.train, "checkpoint_every_epoch", True))

    for epoch in range(start_epoch, cfg.train.stage2_epochs):
        model.head.anneal_tau(epoch)
        print(f"[05] === Epoch {epoch}/{cfg.train.stage2_epochs} "
              f"(tau={model.head.tau.item():.3f} best={best_val:.4f}) ===",
              flush=True)
        loss_fn = make_stage2_loss_fn(evid_loss, cfg, epoch)
        train_result = run_epoch(
            train_source, model, optimizer, loss_fn, device, train=True,
            grad_clip=cfg.train.grad_clip, bptt_chunk=cfg.train.bptt_chunk, max_bins=max_bins,
            label=f"S2 e{epoch}",
        )
        val_result = run_epoch(
            val_source, model, None, loss_fn, device, train=False, max_bins=max_bins,
            label=f"S2 e{epoch} val",
        )
        history.append(
            {"epoch": epoch, "train_loss": train_result.loss, "train_acc": train_result.accuracy,
             "train_macro_f1": train_result.macro_f1,
             "val_loss": val_result.loss, "val_acc": val_result.accuracy,
             "val_macro_f1": val_result.macro_f1}
        )

        gates_report = run_dir / "gates_report.json" if run_dir is not None else None
        g6_ok, g6_msg = gate_g6_numerical_health(torch.tensor(train_result.loss))
        if not g6_ok:
            record_gate(gates_report, "G6", g6_ok, g6_msg + f" (stage2 epoch {epoch})")
            raise RuntimeError(f"Gate G6 failed at stage2 epoch {epoch}: non-finite training loss")

        # Selection on macro-F1. docs/06_TRAINING.md §5 specifies val OpenAUC
        # for Stage 2; that needs the open-set eval path and remains a
        # disclosed deviation, but macro-F1 is strictly closer to it than
        # accuracy — accuracy on this split is ~54% for a two-class predictor.
        improved = val_result.macro_f1 > best_val
        print(f"[05] Epoch {epoch:2d}/{cfg.train.stage2_epochs}  "
              f"train={train_result.loss:.4f}  val={val_result.loss:.4f}  "
              f"macroF1={val_result.macro_f1:.4f}  acc={val_result.accuracy:.4f}  "
              f"tau={model.head.tau.item():.3f}  "
              f"{'NEW BEST' if improved else f'patience={patience_left}'}",
              flush=True)
        if improved:
            best_val = val_result.macro_f1
            best_epoch = epoch
            patience_left = cfg.train.stage2_patience
            if ckpt_best is not None:
                save_checkpoint(ckpt_best, model, optimizer, epoch=epoch,
                                extra={"best_val": best_val, "best_epoch": best_epoch})
        else:
            patience_left -= 1

        if ckpt_last is not None and (ckpt_every_epoch or patience_left <= 0
                                      or epoch == cfg.train.stage2_epochs - 1):
            save_checkpoint(
                ckpt_last, model, optimizer, epoch=epoch,
                extra={"best_val": best_val, "best_epoch": best_epoch,
                       "patience_left": patience_left, "history": history, "stage": "stage2"},
            )

        if patience_left <= 0:
            print(f"[05] Early stop at epoch {epoch} — val_macro_f1={best_val:.4f}")
            break

    if ckpt_best is not None and ckpt_best.exists() and best_epoch >= 0:
        state = torch.load(ckpt_best, map_location=str(device), weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"[05] Restored best weights (epoch {best_epoch}, val_macro_f1={best_val:.4f})")

    # Post-training evidential gates G3/G4 (docs/06_TRAINING.md §8). Only
    # meaningful for the evidential head — the distance_threshold fallback has
    # no evidence/vacuity path to gate.
    if not hasattr(model.head, "log_evidence_clamp"):
        return {"history": history, "best_val_macro_f1": best_val, "best_epoch": best_epoch,
                "optimizer": optimizer}
    gates_report = run_dir / "gates_report.json" if run_dir is not None else None
    mean_known_vac, mean_synth_vac = evaluate_vacuity(model, val_source, cfg, device,
                                                      max_bins=max_bins)
    g3_ok, g3_msg = gate_g3_evidence_collapse(
        mean_known_vac, getattr(cfg.gates, "g3_max_known_vacuity", 0.7))
    record_gate(gates_report, "G3", g3_ok, g3_msg)
    g4_ok, g4_msg = gate_g4_unknown_carving(
        mean_synth_vac, getattr(cfg.gates, "g4_min_unknown_vacuity", 0.5))
    record_gate(gates_report, "G4", g4_ok, g4_msg)

    return {"history": history, "best_val_macro_f1": best_val, "best_epoch": best_epoch,
            "optimizer": optimizer}
