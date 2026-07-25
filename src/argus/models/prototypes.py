"""Multi-prototype bank with few-shot class registration.

See docs/05_ARCHITECTURE.md §6.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Unit-norm prototype bank with sub-prototypes per class.

    Trained prototypes are nn.Parameter; registered prototypes are frozen buffers
    appended at runtime with zero gradient steps.
    """

    def __init__(
        self,
        d_z: int,
        class_names: list[str],
        class_counts: dict[str, int] | None = None,
        sub_prototypes_benign: int = 4,
        sub_prototypes_attack_large: int = 2,
        sub_prototypes_attack_small: int = 1,
        benign_name: str = "Benign",
        large_threshold: int = 10_000,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.class_names = list(class_names)
        self.benign_name = benign_name
        self.sub_benign = sub_prototypes_benign
        self.sub_attack_large = sub_prototypes_attack_large
        self.sub_attack_small = sub_prototypes_attack_small
        self.large_threshold = large_threshold

        if class_counts is None:
            class_counts = {name: large_threshold + 1 for name in class_names}

        self.class_of: list[int] = []
        self.sub_counts: list[int] = []
        for ci, name in enumerate(self.class_names):
            count = class_counts.get(name, large_threshold + 1)
            if name == benign_name:
                n_sub = self.sub_benign
            elif count >= large_threshold:
                n_sub = self.sub_attack_large
            else:
                n_sub = self.sub_attack_small
            self.sub_counts.append(n_sub)
            self.class_of.extend([ci] * n_sub)

        total_protos = sum(self.sub_counts)
        bank = torch.randn(total_protos, d_z)
        bank = F.normalize(bank, dim=-1)
        self.bank = nn.Parameter(bank)
        self._num_trained = total_protos

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def post_step_normalize(self) -> None:
        """Re-normalise all trainable prototypes to unit norm."""
        with torch.no_grad():
            self.bank.data[: self._num_trained] = F.normalize(
                self.bank.data[: self._num_trained], dim=-1
            )

    def cosine_to_classes(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: [B, d_z] unit-norm embeddings
        Returns:
            [B, C] max cosine per class over sub-prototypes.
        """
        # [B, P] cosine to every sub-prototype
        cos = torch.matmul(z, self.bank.T)
        bsz = z.shape[0]
        c = self.num_classes
        out = torch.full((bsz, c), -2.0, device=z.device, dtype=z.dtype)
        for ci in range(c):
            mask = torch.tensor(
                [i == ci for i in self.class_of], device=z.device, dtype=torch.bool
            )
            if mask.any():
                out[:, ci] = cos[:, mask].max(dim=1).values
        return out

    def diversity_loss(self, max_cosine: float = 0.8) -> torch.Tensor:
        """Penalty encouraging sub-prototypes of the same class to stay apart."""
        loss = torch.tensor(0.0, device=self.bank.device)
        for ci in range(self.num_classes):
            mask = torch.tensor(
                [i == ci for i in self.class_of], device=self.bank.device, dtype=torch.bool
            )
            if mask.sum() < 2:
                continue
            protos = self.bank[mask]
            sim = torch.matmul(protos, protos.T)
            # Zero diagonal
            sim = sim - torch.eye(sim.shape[0], device=sim.device)
            loss = loss + F.relu(sim - max_cosine).pow(2).sum()
        return loss

    def register_class(
        self,
        name: str,
        embeddings: torch.Tensor,
        n_sub: int = 1,
    ) -> int:
        """Append a new class to the bank with zero gradient steps.

        Args:
            name: new class name
            embeddings: [n, d_z] unit-norm embeddings of labelled samples
            n_sub: number of sub-prototypes; >1 only if enough samples.
        Returns:
            class index of the newly registered class
        """
        assert not torch.is_grad_enabled(), "register_class must be called under torch.no_grad()"
        with torch.no_grad():
            return self._register_class_impl(name, embeddings, n_sub)

    def _register_class_impl(
        self,
        name: str,
        embeddings: torch.Tensor,
        n_sub: int = 1,
    ) -> int:
        embeddings = embeddings.to(self.bank.device)
        if n_sub > 1 and embeddings.shape[0] >= 4 * n_sub:
            # Simple k-means-ish via random init + one refinement step
            indices = torch.randperm(embeddings.shape[0])[:n_sub]
            centres = embeddings[indices]
            for _ in range(5):
                sim = torch.matmul(embeddings, centres.T)
                labels = sim.argmax(dim=1)
                new_centres = []
                for j in range(n_sub):
                    mask = labels == j
                    if mask.any():
                        new_centres.append(embeddings[mask].mean(0, keepdim=True))
                    else:
                        new_centres.append(centres[j : j + 1])
                centres = torch.cat(new_centres, dim=0)
        else:
            centres = embeddings.mean(0, keepdim=True)
        centres = F.normalize(centres, dim=-1)

        new_bank = torch.cat([self.bank.detach(), centres], dim=0)
        # Convert old Parameter to buffer + new Parameter? Simpler: keep Parameter
        # but freeze the registered rows by excluding them from optimiser via param groups.
        self.bank = nn.Parameter(new_bank, requires_grad=False)
        new_idx = self.num_classes
        self.class_names.append(name)
        self.sub_counts.append(centres.shape[0])
        self.class_of.extend([new_idx] * centres.shape[0])
        return new_idx

    def get_trainable_mask(self) -> torch.Tensor:
        """Return bool mask of trainable prototype rows."""
        mask = torch.zeros(self.bank.shape[0], dtype=torch.bool, device=self.bank.device)
        mask[: self._num_trained] = True
        return mask

    def ema_update(self, class_idx: int, embeddings: torch.Tensor, momentum: float) -> None:
        """EMA-update the nearest sub-prototype of `class_idx` toward the mean
        of `embeddings`, with zero gradient steps.

        See docs/05_ARCHITECTURE.md §6.2 "Optional: EMA prototype drift
        correction" and docs/07_HYPERPARAMETERS.md `prototype_ema_momentum`
        (default 0.99, i.e. slow drift). The caller is responsible for gating
        `embeddings` on the evidence threshold *before* calling this — this
        method performs no gating itself (see docs/10_ADVERSARIAL.md §4, A3).
        """
        assert not torch.is_grad_enabled(), "ema_update must be called under torch.no_grad()"
        if embeddings.shape[0] == 0:
            return
        with torch.no_grad():
            mask = torch.tensor(
                [i == class_idx for i in self.class_of], device=self.bank.device, dtype=torch.bool
            )
            if not mask.any():
                return
            mean_embed = F.normalize(embeddings.mean(0, keepdim=True), dim=-1).to(self.bank.device)  # [1, d_z]
            protos = self.bank.data[mask]  # [n_sub, d_z]
            # Update whichever sub-prototype is nearest to the batch mean —
            # the common case (attack classes) has a single sub-prototype.
            sims = torch.matmul(protos, mean_embed.T).squeeze(-1)  # [n_sub]
            local_idx = int(sims.argmax().item())
            global_idx = mask.nonzero(as_tuple=True)[0][local_idx]
            # momentum close to 1 => slow drift (mostly retain the old prototype).
            updated = momentum * self.bank.data[global_idx] + (1.0 - momentum) * mean_embed.squeeze(0)
            self.bank.data[global_idx] = F.normalize(updated, dim=-1)
