"""UNKNOWN triage: per-decision reports + quantitative cluster validation.

See docs/11_XAI.md §5. Per-decision triage is a deterministic template filled
from the head's own quantities (no LLM, unlike XG-NID — auditable). Cluster
validation groups UNKNOWN embeddings and checks whether they organise into
coherent, discoverable clusters aligned with the true held-out classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from argus.models.epc import EPCHead


@dataclass
class TriageReport:
    verdict: str
    total_evidence: float
    vacuity: float
    nearest_prototypes: list[dict]
    scale_gate: dict[str, float] | None
    prose: str


def _nearest_prototypes(head: EPCHead, z: torch.Tensor, class_names: list[str], top_k: int = 2) -> list[dict]:
    with torch.no_grad():
        cos_c = head.prototype_bank.cosine_to_classes(z.unsqueeze(0)).squeeze(0)  # [C]
    order = torch.argsort(cos_c, descending=True)[:top_k]
    return [
        {"class": class_names[int(i)], "cosine": float(cos_c[i].item()), "distance": float(1.0 - cos_c[i].item())}
        for i in order
    ]


def render_triage_report(
    head: EPCHead,
    outputs: dict,
    idx: int,
    class_names: list[str],
    gate: dict[str, float] | None = None,
) -> TriageReport:
    """Build a per-decision triage report for a single UNKNOWN-verdicted flow.

    Args:
        outputs: the head's forward-pass output dict.
        idx: row within `outputs` for this flow.
        gate: optional {"short": g_S, "mid": g_M, "long": g_L} multi-scale
            fusion gate values for this flow (docs §1.2, temporal attribution).
    """
    z = outputs["z"][idx]
    total_evidence = float(outputs["evidence_total"][idx].item())
    vacuity = float(outputs["vacuity"][idx].item())
    nearest = _nearest_prototypes(head, z, class_names)

    top = nearest[0] if nearest else None
    scale_note = ""
    if gate:
        dominant_scale = max(gate, key=gate.get)
        scale_note = f" Evidence is concentrated at the {dominant_scale} time scale (gate={gate[dominant_scale]:.2f})."
    prose = (
        f"UNKNOWN verdict (total evidence {total_evidence:.3f}, vacuity {vacuity:.2f}). "
        + (f"Nearest known class is '{top['class']}' (cosine {top['cosine']:.2f}), "
           f"but the total evidence is too low to classify confidently." if top else "")
        + scale_note
    )
    return TriageReport(
        verdict="UNKNOWN",
        total_evidence=total_evidence,
        vacuity=vacuity,
        nearest_prototypes=nearest,
        scale_gate=gate,
        prose=prose,
    )


@dataclass
class ClusterValidationResult:
    purity: float
    nmi: float
    ari: float
    n_clusters: int
    nearest_prototype_correspondence: dict[str, str] = field(default_factory=dict)


def _cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    total = len(true_labels)
    if total == 0:
        return 0.0
    correct = 0
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if not mask.any():
            continue
        values, counts = np.unique(true_labels[mask], return_counts=True)
        correct += counts.max()
    return float(correct / total)


def validate_unknown_clusters(
    embeddings: np.ndarray,
    true_held_out_labels: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    seed: int = 0,
) -> ClusterValidationResult:
    """Cluster UNKNOWN-verdicted embeddings and validate against the true
    (held-out, ground-truth) class labels the model never trained on.

    Args:
        embeddings: [n, d_z] unit-norm embeddings of flows verdicted UNKNOWN.
        true_held_out_labels: [n] ground-truth held-out class id per flow
            (available only for offline evaluation; never used at inference).
        n_clusters: number of held-out classes (for k-means); ignored for hdbscan.
        method: "kmeans" (fixed k) or "hdbscan" (automatic k).
    """
    if len(embeddings) == 0:
        return ClusterValidationResult(purity=0.0, nmi=0.0, ari=0.0, n_clusters=0)

    if method == "kmeans":
        model = KMeans(n_clusters=max(n_clusters, 1), random_state=seed, n_init=10)
        cluster_labels = model.fit_predict(embeddings)
    elif method == "hdbscan":
        model = HDBSCAN(min_cluster_size=max(len(embeddings) // 20, 2))
        cluster_labels = model.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    purity = _cluster_purity(true_held_out_labels, cluster_labels)
    nmi = float(normalized_mutual_info_score(true_held_out_labels, cluster_labels))
    ari = float(adjusted_rand_score(true_held_out_labels, cluster_labels))
    n_found = len(set(cluster_labels.tolist()) - {-1})  # exclude HDBSCAN noise label

    return ClusterValidationResult(purity=purity, nmi=nmi, ari=ari, n_clusters=n_found)


def nearest_prototype_correspondence(
    head: EPCHead,
    held_out_embeddings: dict[str, np.ndarray],
    known_class_names: list[str],
) -> dict[str, str]:
    """For each held-out class, the modal "nearest known prototype" among its
    UNKNOWN-verdicted flows — validates that unknowns land near semantically
    related known classes (e.g. held-out DDoS -> nearest known DoS).
    """
    correspondence = {}
    for held_out_name, embeds in held_out_embeddings.items():
        if len(embeds) == 0:
            continue
        z = torch.as_tensor(embeds, dtype=torch.float32)
        with torch.no_grad():
            cos_c = head.prototype_bank.cosine_to_classes(z)  # [n, C]
        nearest_idx = cos_c.argmax(dim=1)
        modal = int(torch.mode(nearest_idx).values.item())
        correspondence[held_out_name] = known_class_names[modal]
    return correspondence
