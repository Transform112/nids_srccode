# 00 — Overview

## 1. Problem

Machine-learning network intrusion detection systems (NIDS) are trained on a
fixed, closed set of attack classes. In deployment three things happen that the
training assumption does not cover:

1. **New attacks appear.** A closed-set classifier must assign every input to a
   known class; a zero-day is silently absorbed into whatever class is nearest,
   usually "benign".
2. **The model must be updated without being rebuilt.** Full retraining is
   operationally expensive and risks regressing on previously-handled attacks.
3. **The detector itself becomes a target.** Graph-based NIDS aggregate over a
   host's neighbourhood, and an attacker frequently *controls part of that
   neighbourhood* — they can emit additional flows at will.

Graph neural networks are the current state of the art for flow-based NIDS
because network traffic is natively relational. But essentially every published
GNN-NIDS is closed-set, is evaluated on random splits, and has been shown to
degrade sharply under realistic drift and structural attacks.

## 2. The gap (verified, not assumed)

Two arXiv full-text queries were run on 2026-07-25:

| Query | Total results | Finding |
|---|---|---|
| `"graph neural network" AND "intrusion detection" AND ("open-set" OR "zero-day" OR "continual")` | **3** | None performs principled open-world recognition |
| `"intrusion detection" AND ("open-set recognition" OR "prototype learning" OR "evidential")` | **9** | **All flow-independent / tabular. None is graph-based.** |

Meanwhile the GNN-NIDS literature is dense with accuracy-oriented work
(E-GraphSAGE, E-ResGAT, EDGMAT, Anomal-E, STEG, CAGN-GAT, GTCN-G, PPT-GNN,
GraphIDS, XG-NID, X-CBA). See `01_RELATED_WORK.md`.

Two recent papers supply the vulnerability evidence:

- **Venturi et al.** (arXiv:2403.11830) formalise *problem-space structural*
  adversarial attacks on GNN-NIDS and show that models which resist
  feature-space attacks are still broken by structure-space attacks.
- **REAL-IoT** (arXiv:2507.10836) shows GNN-NIDS performance drops substantially
  under distribution drift and realistic attacks relative to benchmark numbers,
  i.e. **the literature systematically overestimates GNN-NIDS resilience.**

So: open-world recognition and structural robustness are each missing from
GNN-NIDS, and they are *the same operational problem* — the model must behave
sanely on inputs it was not traine/d to expect, whether those inputs are a novel
attack or an adversarially manipulated neighbourhood.

## 3. Contributions

**C1 — First open-world GNN-NIDS.**
An evidential-prototype head replaces the softmax. A single forward pass yields
(a) a known-class prediction, (b) a calibrated *defer* signal when evidence is
high but ambiguous, and (c) an *UNKNOWN* verdict when total evidence is low.
New attack classes are registered by appending a prototype computed as the mean
embedding of `n` labelled samples — **zero gradient steps, therefore zero
catastrophic forgetting by construction**, which is a structural guarantee
rather than an empirical observation.

**C2 — Structure-robust message passing for NIDS.**
Degree-capped neighbourhoods with trimmed-mean / soft-medoid aggregation, edge
features partitioned into attacker-controllable and observer-derived channels
with a gradient penalty against relying on the former, and time-decayed
attention that down-weights stale injected edges. This yields a stated
*breakdown point*: an adversary must inject more than `β·K` flows into a victim's
neighbourhood before the aggregate moves at all.

**C3 — A temporal representation that demonstrably contributes.**
Multi-scale windows (1 s / 30 s / 300 s), functional Δt encoding, derived
intra-flow rhythm descriptors, and a per-host spectral beaconing descriptor —
validated by a seven-rung ablation ladder reporting **per-class** deltas. The
L0→L2 rung alone is the first published measurement of what NF-v3's temporal
features are worth to a graph model.

**C4 — Explanation-verified robustness.**
XAI is used as evidence, not decoration: adversarially injected edges receive
near-zero attribution mass under robust aggregation and high mass under mean
aggregation, causally explaining *why* C2 works.

## 4. Why these four cohere

C1 and C2 are the two coupled mechanisms. They share a mechanism: both are about
**not over-committing on low-evidence input**. The evidential head refuses to
classify when evidence is thin; the robust aggregator refuses to let a minority
of neighbours dominate. C3 supplies the signal that makes both possible — and
notably, the temporal features are the *hardest for an attacker to forge*,
because altering inter-packet timing degrades the attack's own effectiveness.
C4 verifies C2 rather than merely illustrating it.

## 5. Scope

**In scope**
- Flow-level detection on NetFlow-v3 (NF3) datasets, 53 features.
- Multi-class classification with an explicit UNKNOWN class and a DEFER action.
- Few-shot registration of new attack classes at inference time.
- Problem-space adversarial evaluation (feature, structural, poisoning,
  adaptive, temporal).
- Post-hoc and native explanation, with quantitative explanation metrics.
- Deployment measurement: throughput, latency percentiles, memory.

**Out of scope** (do not add these; they were considered and rejected)
- Continuous-time TGN memory — dropped in a prior iteration of this project;
  expensive, hard to reproduce, and superseded by windowed GRU memory.
- Packet-level or payload modelling — NF-v3 is flow-level only.
- Federated learning, LLM-based components, reinforcement-learning training.
- Reproducing FastFlow's RL loop.
- Collecting a real testbed capture.

## 6. Success criteria

The paper is publishable if all of the following hold:

1. Closed-set macro-F1 is at parity with or above re-run GNN baselines.
   *Parity is sufficient — accuracy is not the contribution.*
2. Open-set unknown detection beats softmax-threshold, OpenMax, and energy-score
   baselines on OpenAUC, averaged over ≥5 random class holdouts.
3. Few-shot registration reaches usable F1 on a novel class at n ≤ 20 with
   exactly zero degradation on previously known classes.
4. Under structural attack A2, ARGUS retains substantially more macro-F1 than
   E-GraphSAGE at equal injection budget.
5. The temporal ladder shows a measurable, per-class-attributable gain.
6. Expected Calibration Error is materially lower than a softmax baseline.

## 7. Glossary

| Term | Meaning |
|---|---|
| **ARGUS** | This system. Working name; not an acronym requiring expansion in the paper. |
| **SR-TEG** | Structure-Robust Temporal Edge GNN — the encoder. |
| **EPC** | Evidential Prototype Classifier — the head. |
| **NF-v3 / NF3** | NetFlow version 3 NIDS datasets, 53 features, 10 temporal. |
| **Flow** | One NetFlow record = one graph edge. |
| **Host** | A graph node; IP or IP:port depending on dataset. |
| **Window** | A time slice of flows from which one graph is built. |
| **Scale** | One of the three window durations (1 s / 30 s / 300 s). |
| **Prototype** | A unit-norm embedding vector representing one class. |
| **Evidence** | Non-negative Dirichlet concentration mass assigned to a class. |
| **DEFER** | Output meaning "evidence is high but ambiguous; escalate to analyst". |
| **UNKNOWN** | Output meaning "total evidence too low; likely novel class". |
| **Openness** | Standard open-set difficulty measure; see `08_EVALUATION.md` §3.2. |
| **Breakdown point** | Fraction of neighbours an adversary must control to move a robust aggregate. |
| **Controllable feature** | A feature the attacker can set freely at negligible cost. |
| **Observer-derived feature** | A feature that is costly or self-defeating for the attacker to forge. |
