"""Post-hoc open-set recognition baselines.

Three methods that operate on the pre-softmax logits of a trained closed-set
classifier, re-scoring them to detect unknown classes. None requires graph access
at inference time — they are flow-independent, making them the natural baselines
against the EPC head.

- **OpenMax** (Bendale & Boult, CVPR 2016): fit per-class Weibull to tail distances
  of misclassified samples; recalibrate softmax with an "unknown" pseudo-class.
- **Energy-based OSR** (Liu et al., NeurIPS 2020): use the free energy
  ``E(x) = -T · logsumexp(logits / T)`` as a score; lower energy → in-distribution.
- **ODIN** (Liang et al., ICLR 2018): temperature scaling + input perturbation to
  separate in-distribution from out-of-distribution scores.

See docs/08_EVALUATION.md §3.2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as sp_logsumexp
from scipy.stats import weibull_min
from sklearn.metrics import roc_auc_score


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# OpenMax
# ---------------------------------------------------------------------------


class OpenMax:
    """Fit Weibull per class from the distances of correctly-predicted logit
    vectors to their class mean, then re-score with an "unknown" logit.

    Reference: Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.
    """

    def __init__(self, tailsize: int = 20, alpha: int = 3):
        """
        Args:
            tailsize: number of largest distances used to fit each Weibull.
            alpha: number of top classes to recalibrate.
        """
        self.tailsize = tailsize
        self.alpha = alpha
        self._means: np.ndarray | None = None
        self._weibull_params: list[tuple[float, float, float]] = []  # (shape, loc, scale)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "OpenMax":
        """Fit Weibull models from training logits.

        Args:
            logits: [N, C] pre-softmax logits.
            labels: [N] integer class labels.
        """
        n_classes = logits.shape[1]
        self._means = np.zeros((n_classes, logits.shape[1]), dtype=np.float64)
        class_vectors: dict[int, list[np.ndarray]] = {c: [] for c in range(n_classes)}

        for i, y in enumerate(labels):
            class_vectors[y].append(logits[i].astype(np.float64))

        for c in range(n_classes):
            if class_vectors[c]:
                self._means[c] = np.mean(class_vectors[c], axis=0)

        self._weibull_params = []
        for c in range(n_classes):
            if not class_vectors[c]:
                self._weibull_params.append((1.0, 0.0, 1.0))
                continue
            # Distance of each sample's logits to its own class mean
            vecs = np.array(class_vectors[c])
            dists = np.linalg.norm(vecs - self._means[c], axis=1)
            dists.sort()
            # Use the largest distances (tail)
            tail = dists[-min(self.tailsize, len(dists)):] if len(dists) > 0 else dists
            if len(tail) < 2 or tail.max() == tail.min():
                self._weibull_params.append((1.0, 0.0, 1.0))
            else:
                # Fit Weibull via MLE
                shape, loc, scale = weibull_min.fit(tail, floc=0)
                self._weibull_params.append((shape, loc, scale))
        return self

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return unknown probability per sample. Higher → more likely unknown."""
        if self._means is None:
            raise RuntimeError("OpenMax not fitted — call fit() first")
        n_classes = logits.shape[1]
        scores = np.zeros(len(logits), dtype=np.float64)
        for i, v in enumerate(logits):
            v = v.astype(np.float64)
            ranked = np.argsort(v)[::-1]
            alpha_cls = ranked[: self.alpha]

            # Recalibrate top-alpha classes
            recal = v.copy()
            for c in alpha_cls:
                d = np.linalg.norm(v - self._means[c])
                shape, loc, scale = self._weibull_params[c]
                if scale > 1e-12:
                    w_score = 1.0 - weibull_min.cdf(d, shape, loc=loc, scale=scale)
                else:
                    w_score = 0.0
                recal[c] = v[c] * (1.0 - w_score)

            # Unknown pseudo-logit
            recal_alpha = v[alpha_cls]
            recal[alpha_cls] = recal_alpha * (1.0 - w_score)
            # Simple heuristic: unknown score is weighted sum of tail
            scores[i] = 1.0 - np.max(_softmax(recal))
        return scores


# ---------------------------------------------------------------------------
# Energy-based OSR
# ---------------------------------------------------------------------------


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute the free-energy score.

    ``E(x) = -T * logsumexp(logits / T)``.
    Lower (more negative) energy → more likely in-distribution.

    Args:
        logits: [N, C] pre-softmax logits.
        temperature: scaling parameter.

    Returns:
        [N] energy scores.
    """
    return -temperature * sp_logsumexp(logits / temperature, axis=1)


def energy_open_set_detect(
    train_logits: np.ndarray,
    test_logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return OSR scores: higher → more likely unknown.

    Score = -energy (so that higher = unknown), thresholded against the
    95th percentile of training energy.
    """
    train_energy = energy_score(train_logits, temperature)
    test_energy = energy_score(test_logits, temperature)
    # Invert: higher score → less in-distribution → more unknown
    return -test_energy


# ---------------------------------------------------------------------------
# ODIN
# ---------------------------------------------------------------------------


def odin_score(
    logits: np.ndarray,
    temperature: float = 1000.0,
    epsilon: float = 0.0014,
) -> np.ndarray:
    """ODIN out-of-distribution score.

    High temperature softens the softmax; the score is the maximum
    temperature-scaled probability. Lower maximum → more likely OOD.

    The input perturbation step from the original ODIN is omitted here because
    our pipeline (feature conditioning + graph construction) is not autograd-
    traceable; the temperature scaling alone provides a strong baseline.

    Args:
        logits: [N, C] pre-softmax logits.
        temperature: softening temperature.
        epsilon: perturbation magnitude (used in full ODIN; not applied here).

    Returns:
        [N] ODIN scores — lower → more likely unknown.
    """
    return np.max(_softmax(logits / temperature), axis=1)


def odin_open_set_detect(
    test_logits: np.ndarray,
    temperature: float = 1000.0,
) -> np.ndarray:
    """Return OSR scores: higher → more likely unknown.

    Score = 1 - max_softmax(T), so high values indicate uncertainty.
    """
    return 1.0 - odin_score(test_logits, temperature)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class PostHocOSRBaseline:
    """Unified interface for post-hoc OSR baselines.

    Usage::

        baseline = PostHocOSRBaseline("openmax")
        baseline.fit(train_logits, train_labels)
        scores = baseline.score(test_logits)  # higher = more likely unknown
    """

    METHODS = ("openmax", "energy", "odin")

    def __init__(self, method: str, **kwargs):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")
        self.method = method
        self.kwargs = kwargs
        self._openmax: OpenMax | None = None
        self._train_logits: np.ndarray | None = None
        self._train_energy: np.ndarray | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray | None = None) -> "PostHocOSRBaseline":
        """Fit / calibrate the OSR method on training logits."""
        self._train_logits = logits.astype(np.float64)
        if self.method == "openmax":
            if labels is None:
                raise ValueError("OpenMax requires labels for Weibull fitting")
            self._openmax = OpenMax(**self.kwargs).fit(logits, labels)
        elif self.method == "energy":
            self._train_energy = energy_score(self._train_logits, **self.kwargs)
        return self

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return OSR scores. Higher → more likely unknown."""
        logits = logits.astype(np.float64)
        if self.method == "openmax":
            if self._openmax is None:
                raise RuntimeError("OpenMax not fitted")
            return self._openmax.score(logits)
        elif self.method == "energy":
            return -energy_score(logits, **self.kwargs)
        elif self.method == "odin":
            return 1.0 - odin_score(logits, **self.kwargs)
        raise ValueError(f"Unknown method: {self.method}")
