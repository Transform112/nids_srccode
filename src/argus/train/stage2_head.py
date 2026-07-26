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

import torch
import torch.nn.functional as F

from argus.graph.batching import AnchorBinGraphSource
from argus.losses.evidential import EvidentialLoss
from argus.models.argus import ArgusModel
from argus.train.loop import run_epoch


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


def make_stage2_loss_fn(evid_loss: EvidentialLoss, cfg, epoch: int):
    kl_weight = min(1.0, epoch / max(cfg.loss.kl_anneal_epochs, 1)) * cfg.loss.lambda_kl_max

    def loss_fn(outputs: dict, targets: torch.Tensor, model: ArgusModel) -> tuple[torch.Tensor, int]:
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
        return loss, correct

    return loss_fn


def train_stage2(
    model: ArgusModel,
    train_source: AnchorBinGraphSource,
    val_source: AnchorBinGraphSource,
    cfg,
    device: torch.device,
    max_bins: int | None = None,
) -> dict:
    """Full Stage-2 training: encoder frozen, head parameters trained at low LR."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    if model.memory is not None:
        for p in model.memory.parameters():
            p.requires_grad_(False)
    for p in model.fusion.parameters():
        p.requires_grad_(False)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params, lr=cfg.train.stage2_lr, weight_decay=cfg.regularisation.weight_decay
    )
    evid_loss = EvidentialLoss()

    best_val = -1.0
    patience_left = cfg.train.stage2_patience
    history = []

    for epoch in range(cfg.train.stage2_epochs):
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
             "val_loss": val_result.loss, "val_acc": val_result.accuracy}
        )
        improved = val_result.accuracy > best_val
        print(f"[05] Epoch {epoch:2d}/{cfg.train.stage2_epochs}  "
              f"train={train_result.loss:.4f}  val={val_result.loss:.4f}  "
              f"acc={val_result.accuracy:.4f}  tau={model.head.tau.item():.3f}  "
              f"{'NEW BEST' if improved else f'patience={patience_left}'}",
              flush=True)
        if val_result.accuracy > best_val:
            best_val = val_result.accuracy
            patience_left = cfg.train.stage2_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[05] Early stop at epoch {epoch} — val_acc={best_val:.4f}")
                break

    return {"history": history, "best_val_acc": best_val}
