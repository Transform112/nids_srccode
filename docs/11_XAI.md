# 11 — Explainability (XAI)

The evidence base for **C4**. Module map: `src/argus/xai/` —
`evidence_attrib.py`, `explainers.py`, `metrics.py`, `triage.py`.

**Principle: XAI here is evidence, not decoration.** Most NIDS papers append a
SHAP plot and assert interpretability. This section instead uses explanation to
*verify a robustness claim* and to *make the UNKNOWN verdict actionable*, and
measures explanation quality quantitatively.

---

## 1. Native evidence attribution

The EPC head is natively decomposable, which post-hoc explainers on a softmax
head are not. Exploit that.

### 1.1 Why decomposition is available

Evidence for class `k` is `e_k = exp(−(d_k − m)/τ)` with `d_k = 1 − ⟨z_e, p_k⟩`.
So `log e_k = (⟨z_e, p_k⟩ − 1 + m)/τ`, and the inner product decomposes
coordinate-wise:

```
log e_k  =  (1/τ) · Σ_{j=1}^{d_z} z_e[j] · p_k[j]  +  const
```

Each embedding dimension `j` contributes `z_e[j] · p_k[j] / τ` to the log-evidence
for class `k`. This is exact, not an approximation, and requires no surrogate
model or sampling.

### 1.2 Attribution to inputs

Two levels, both wanted:

**Feature-level.** Propagate the embedding-dimension attributions back to input
features with Integrated Gradients along the path from a benign baseline:

```
attr_feature[i, k] = (x[i] − x_base[i]) · ∫_0^1 ∂ log e_k / ∂x[i] |_{x_base + α(x − x_base)} dα
```
50 integration steps. Baseline `x_base` = the median benign training flow.
Report the top 15 features per decision.

**Neighbour-level (graph attribution).** For target flow `e`, the contribution of
each sampled neighbour edge `n`:

```
attr_edge[n] = α_{n,v} · || ∂ h_e / ∂ m_{n→v} ||_2
```
i.e. attention weight × gradient sensitivity of the target's representation to
that neighbour's message. Normalise so `Σ_n attr_edge[n] = 1`, giving an
**attribution mass distribution over the neighbourhood**. This quantity is what
§3 uses to verify C2. Report the top 10 edges.

**Temporal attribution.** Because the multi-scale gate `g` is explicit, report
`g_S, g_M, g_L` alongside every explanation. This says *at which time scale* the
evidence was found — directly serving C3.

---

## 2. Baseline explainers

For comparison, all applied to the same decisions:

| Explainer | Applies to | Notes |
|---|---|---|
| **GNNExplainer** | Graph structure + features | 200 epochs; standard mask-learning baseline |
| **PGExplainer** | Graph structure | 30 epochs; amortised, faster at inference |
| **KernelSHAP** | Features only | 200 samples; flow-independent view |
| **Attention weights alone** | Graph structure | The naive baseline; attention is *not* explanation, and showing where it disagrees with attribution is a useful minor result |
| **Counterfactual (ProvX-style)** | Graph structure | Minimal edge subset whose removal flips the prediction; gives the *necessity* metric |

---

## 3. Robustness verification — the C4 figure ⭐

This is the contribution that makes XAI load-bearing rather than ornamental.

**Claim to verify:** robust aggregation works because it denies adversarially
injected edges influence over the aggregate.

**Experiment.**

```
1. Take malicious flows correctly detected under clean conditions.
2. Run attack A2 (structural injection) at budget m, for m in {4, 8, 16, 32}.
3. For each configuration in {mean, trimmed(β=0.25), soft_medoid}:
     a. compute attr_edge over the neighbourhood
     b. sum attribution mass falling on INJECTED edges
     c. record injected_mass_fraction = Σ_{n ∈ injected} attr_edge[n]
4. Plot injected_mass_fraction vs m, one line per aggregation type. (Figure F7)
```

**Expected result.** Under `mean`, injected mass fraction rises roughly linearly
with `m/(m + n_benign)` — the attacker buys influence in proportion to injection.
Under `trimmed`, it stays near zero until the injection share approaches `β`,
then rises. The knee should sit at approximately `β`.

**If the knee appears at `β`, the explanation has causally confirmed the
breakdown-point mechanism**, and F6 (ASR vs budget) and F7 (attribution vs
budget) tell one coherent story from two independent measurements. That pairing
is stronger than either figure alone.

If the knee does *not* appear at `β`, that is an important negative finding about
the interaction between attention weighting and coordinate-wise trimming — report
it and investigate rather than suppressing it.

---

## 4. Explanation quality metrics

Quantitative, not illustrative. All computed over ≥1,000 sampled decisions.

| Metric | Definition | Direction |
|---|---|---|
| **Fidelity+** | Δ in predicted-class probability when the top-k explanation elements are **removed**. Large = the explanation identified what mattered. | higher better |
| **Fidelity−** | Δ when everything **except** the top-k is removed. Small = the explanation is sufficient. | lower better |
| **Sparsity** | `1 − k / total_elements` at fixed fidelity | higher better |
| **Necessity** (ProvX) | Fraction of explanations whose removal flips the prediction | higher better |
| **Stability** | Mean rank correlation (Spearman) of attributions across 20 small input perturbations that do not change the prediction | higher better |
| **Adversarial stability** | Same, but under A1 perturbations at small ε that do not flip the prediction | higher better |
| **Runtime** | ms per explanation | lower better |

**Stability is the metric that separates useful from decorative explanations.**
An explainer whose output reshuffles under an imperceptible input change cannot
support analyst trust. Native evidence attribution should beat GNNExplainer here
substantially, because it is a closed-form decomposition rather than a learned
mask — that is a concrete, defensible advantage worth stating.

---

## 5. UNKNOWN triage

Makes the UNKNOWN verdict actionable, which is what turns C1 from a metric into
an operational capability.

### 5.1 Per-decision triage report

For each UNKNOWN flow, emit:

```
{
  "verdict": "UNKNOWN",
  "total_evidence": 0.031,
  "vacuity": 0.94,
  "nearest_prototypes": [
      {"class": "dos",      "cosine": 0.61, "distance": 0.39},
      {"class": "scanning", "cosine": 0.44, "distance": 0.56}
  ],
  "low_evidence_drivers": [
      {"feature": "iat_cv_fwd",   "contribution": -0.28},
      {"feature": "duty_cycle",   "contribution": -0.19},
      {"feature": "spectral_flatness", "contribution": -0.14}
  ],
  "scale_gate": {"short": 0.11, "mid": 0.24, "long": 0.65},
  "top_neighbour_edges": [ ... ]
}
```

Rendered in prose: *"Resembles DoS, but the inter-arrival rhythm is far more
irregular than any known DoS variant, and the evidence is concentrated at the
300-second scale."* That is a statement an analyst can act on, and it is derived
mechanically from the head — no LLM required, unlike XG-NID.

### 5.2 Quantitative validation of triage

Do not merely show examples. Validate:

```
1. Collect all flows the model verdicts UNKNOWN on the P-OS test split.
2. Cluster their embeddings z_e (k-means, k = number of held-out classes;
   also report HDBSCAN with automatic k).
3. Compare clusters against ground-truth held-out class labels.
4. Report: cluster purity, Normalised Mutual Information, Adjusted Rand Index.
5. Report nearest-prototype accuracy: for each held-out class, does the modal
   "nearest known prototype" correspond to a semantically related known class?
   (e.g. held-out DDoS → nearest known prototype DoS)
```

High purity means the model is not merely rejecting unknowns but **organising
them into coherent, discoverable groups** — which is precisely what makes
few-shot registration practical, because an analyst can label one cluster rather
than a stream of individual alerts. This links §5 directly back to C1 and is the
strongest operational argument in the paper.

---

## 6. Deliverables

| Artifact | Content | Claim |
|---|---|---|
| **F7** | Injected-edge attribution mass vs budget, by aggregation type | **C4 / C2** |
| **T11** | Explanation quality metrics, all explainers | C4 |
| **F11** | Example explanation panel: one true positive, one UNKNOWN, one adversarial | C4 |
| **T12** | UNKNOWN triage: purity, NMI, ARI, nearest-prototype correspondence | C1 / C4 |
| Appendix | Temporal attribution by class (gate values) | C3 |
| `results/tables/xai_metrics.csv` | Machine-readable | — |

Runner: `scripts/11_run_xai.py`, config `config/experiment/xai.yaml`.

---

## 7. What we do not claim

- We do not claim explanations are *causal* in the network sense. They are
  faithful to the model, which is a different and weaker statement. Say so.
- We do not claim the UNKNOWN triage identifies the attack *by name*. It groups
  and characterises; naming requires an analyst or a threat-intelligence lookup.
- We do not use an LLM to generate explanation text. XG-NID does; the output is
  unverifiable and unmetriced. Our triage prose is a deterministic template
  filled from the head's own quantities, which is auditable.
