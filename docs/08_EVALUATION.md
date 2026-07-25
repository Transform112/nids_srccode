# 08 — Evaluation

Module map: `src/argus/eval/` — `metrics.py`, `openset.py`, `calibration.py`,
`selective.py`, `continual.py`, `deployment.py`, `report.py`.

**Rule: accuracy is never reported as a headline number.** At 0.3%–5% attack
prevalence it is uninformative. Macro-F1, per-class F1, and PR-AUC are the
primary closed-set metrics.

---

## 1. Protocols

| ID | Protocol | Purpose | Claim |
|---|---|---|---|
| **P-CS** | Closed-set, per-class stratified temporal split | Parity with baselines | C1 (prerequisite) |
| **P-OS** | Open-set, leave-attack-classes-out, 5 random holdouts | Unknown detection | **C1** |
| **P-OS2** | **Open-set within-family** (CICIDS2018 only) — hold out one variant, keep its siblings | Unknown detection, hard case | **C1** |
| **P-FS** | Few-shot registration, n ∈ {1,5,10,20,50}, 5 seeds | Extensibility without retraining | **C1** |
| **P-TR** | Cross-dataset transfer, CICIDS2018 ↔ ToN-IoT | Deployability / drift | C1, C3 |
| **P-ID** | **Identity-leakage audit + host-disjoint split** | Behaviour vs identity memorisation | methodological |
| **P-ADV** | Adversarial A1–A5, budget sweeps | Robustness | **C2** |
| **P-TEMP** | Temporal ladder L0→L7 | Temporal contribution | **C3** |
| **P-XAI** | Explanation quality + robustness verification | Interpretability | **C4** |
| **P-DEP** | Throughput, latency, memory | Deployability | C1, C2 |
| **P-TO** | Flow-timeout stability ablation | Robustness to exporter config | secondary |

---

## 2. Closed-set metrics (P-CS)

| Metric | Definition | Notes |
|---|---|---|
| **Macro-F1** | Unweighted mean of per-class F1 | **Primary metric.** |
| **Per-class F1** | Standard | Always report the full table; the story is in the tail classes |
| **Weighted F1** | Support-weighted | Report for comparability with prior work only |
| **PR-AUC (per class)** | Area under precision–recall, one-vs-rest | Preferred over ROC-AUC under extreme imbalance |
| **MCC** | Matthews correlation coefficient, multi-class | Robust to imbalance in a way F1 is not; cheap to add and rarely reported in NIDS |
| **Balanced accuracy** | Mean per-class recall | Secondary |
| **Confusion matrix** | Normalised by true class | Figure F2 |

With a 39,806:1 head-to-tail ratio on the primary dataset (`02_DATASETS.md`
§6), report **per-tier macro-F1** as well: head / body / tail / extreme. A
single macro-F1 hides whether gains come from the classes that matter.

Report as **mean ± std over 5 seeds**. A single-seed number is not acceptable for
any headline claim.

---

## 3. Open-set metrics (P-OS)

### 3.1 Setup

Hold out `C_h = 3` attack classes; their test flows carry ground-truth label
UNKNOWN. Repeat over `R = 5` distinct random holdout sets. **Report mean ± std
across holdout sets** — prior work almost universally reports a single holdout,
which is cherry-pickable.

Holdout sets must be sampled without replacement from the attack classes and
recorded in the run manifest so they are reproducible.

### 3.2 Metrics

| Metric | Definition |
|---|---|
| **Unknown TPR** | Fraction of true-unknown flows assigned UNKNOWN |
| **Unknown FPR** | Fraction of known-class flows wrongly assigned UNKNOWN |
| **AUROC (known vs unknown)** | Using `−E_total` (or vacuity `u`) as the unknown score |
| **OpenAUC** | Joint measure of closed-set accuracy and unknown detection; the standard open-set metric. Report as primary. |
| **Known macro-F1** | Macro-F1 restricted to known classes, computed on flows not rejected as UNKNOWN |
| **Openness sweep** | Vary `C_h ∈ {1,2,3,4}`; openness = `1 − sqrt(2·C_train / (C_test + C_target))`. Plot metric vs openness. |

The unknown score is `u = C / S` (vacuity). Report the score distribution for
known vs unknown flows as a histogram (figure F3) — separation is the visual
argument for C1.

### 3.3 Open-set baselines

Softmax + max-probability threshold, OpenMax, energy score, ODIN, plus the
non-graph open-set systems MSPL and CLOSR. Every one uses the identical splits
and holdout sets.

---

## 4. Additional metric families

### 4.1 Calibration

Rarely reported in NIDS, and mandatory here because the word "evidential" is a
calibration claim.

| Metric | Definition |
|---|---|
| **ECE** | Expected Calibration Error, 15 equal-mass bins: `Σ_b (n_b/N)·|acc(b) − conf(b)|` |
| **MCE** | Maximum Calibration Error over bins |
| **Brier score** | Multi-class, on `p̂` |
| **Reliability diagram** | Figure F4; ARGUS vs softmax baseline overlaid |
| **NLL** | Negative log-likelihood on `p̂` |

Use equal-mass (not equal-width) bins — with confidence concentrated near 1,
equal-width bins put nearly all mass in one bin and understate ECE.

### 4.2 Selective prediction (the DEFER action)

| Metric | Definition |
|---|---|
| **Risk–coverage curve** | Error rate vs fraction of flows not deferred; sweep `θ_defer` |
| **AURC** | Area under the risk–coverage curve; lower is better |
| **E-AURC** | Excess AURC over the optimal oracle ranking |
| **Risk @ 90% coverage** | Single-number operating point |
| **Deferral precision** | Fraction of deferred flows that the non-deferring model would have got wrong |

Deferral precision is the metric that says whether the DEFER action is *useful*
to an analyst rather than merely conservative. Report it.

### 4.3 Few-shot registration (P-FS)

Procedure: train with `C_h` classes held out (as in P-OS), then register each
held-out class from `n` labelled samples, then re-evaluate on the full test set.

| Metric | Definition |
|---|---|
| **New-class F1 @ n** | F1 on the registered class, `n ∈ {1,5,10,20,50}` |
| **Old-class macro-F1 delta** | Macro-F1 on previously known classes, before vs after registration. **Must be exactly 0.000 for ARGUS.** |
| **Forgetting measure** | `max_t acc_t − acc_final` per old class; zero by construction |
| **Backward transfer (BWT)** | Standard continual-learning metric |
| **Registration wall-clock** | Seconds to register; expected O(n), milliseconds |

Baseline comparison: fine-tuning the softmax baseline on the same `n` samples.
Expect visible catastrophic forgetting, which is the contrast that makes C1
land. Also compare against full retraining as an upper bound on new-class F1 and
a lower bound on cost.

### 4.4 Cross-dataset transfer (P-TR)

Train on NF-CICIDS2018, test on NF-ToN-IoT (and the reverse). Schemas are
identical so no feature alignment is needed — but the **label vocabularies are
disjoint**, so evaluation is by family mapping (`dos`, `ddos`, `brute`, `web`,
`other`) plus unknown detection. Classes present in the target but absent from
the source count as UNKNOWN.

Transfer is also the **ultimate control for host identity leakage** (TRAP 3): no
host in ToN-IoT appears in CICIDS2018, so any transfer performance cannot be
identity memorisation. Report it alongside the identity floor from
`02_DATASETS.md` §7.1.

Report: known macro-F1 (family-mapped), unknown TPR/FPR, OpenAUC, ECE, and the
`L7_PROTO` `OTHER` rate (a drift indicator). Also report zero-shot performance
and performance after few-shot registration of `n = 20` samples per target-only
class — that pairing is the strongest deployability evidence in the paper.

### 4.5 Adversarial (P-ADV)

See `10_ADVERSARIAL.md` for attack definitions.

| Metric | Definition |
|---|---|
| **Robust macro-F1 @ budget** | Macro-F1 under attack, as a function of attack budget |
| **Attack success rate (ASR)** | Fraction of attack flows that evade detection after perturbation |
| **Robustness curve** | ASR vs budget; the money plot (figure F6) |
| **Empirical breakdown point** | Injection budget `m` at which robust macro-F1 drops below 90% of clean |
| **Unknown-evasion rate** | Fraction of adversarial flows classified as a *known benign* class rather than UNKNOWN |

The last metric matters: a good open-world model should route successful evasions
to UNKNOWN rather than to "benign". Report it.

### 4.6 Deployment (P-DEP)

Measured through `StreamingDetector` (`04_GRAPH_CONSTRUCTION.md` §6), not through
the training loop.

| Metric | Notes |
|---|---|
| **Throughput** | Flows/second, sustained over ≥10 min of test stream |
| **Latency p50 / p95 / p99** | Per-flow, from `push()` entry to verdict |
| **Peak GPU memory** | MB |
| **Peak host memory** | MB, including window buffers and node-memory dict |
| **Model size** | Parameters and MB on disk |
| **Registration latency** | ms per `register_class` call |

Report on the local machine and note the hardware. Compare against E-GraphSAGE
under identical conditions. If ARGUS is slower, report honestly and price the
temporal/robustness gain against it — the cost table is more credible than a
claim of free improvement.

### 4.7 Flow-timeout stability (P-TO)

Borrowed from the flow-timeout paper in `papers/` (up to 8.77% F1 swing from
exporter timeout choice; tree ensembles most stable).

ARGUS's windowed graph does not depend on exporter idle/active timeout, so vary
the effective timeout by re-aggregating flow records and compare F1 variance
against a flow-only baseline (Extra Trees) and a single-scale GNN. Cheap to run;
strengthens the deployability argument.

---

## 5. Baselines — re-run, never quoted

**Never copy numbers from other papers.** Splits, preprocessing, subsampling, and
class vocabularies all differ; a quoted number is not a comparison.

| Baseline | Family | Isolates |
|---|---|---|
| Extra Trees | Non-DL, flow-independent | Whether the graph helps at all |
| Random Forest | Non-DL | Same |
| MLP on `F_e` | DL, flow-independent | Whether the graph helps, DL-controlled |
| E-GraphSAGE | GNN, mean agg, constant node init | The published GNN-NIDS bar |
| EGATv2 / E-ResGAT | GNN, attention | Whether robust aggregation beats attention |
| Anomal-E | GNN, self-supervised, binary | Unsupervised comparison |
| GraphIDS | GNN, masked AE | If reproducible within budget; else omit and say so |
| ARGUS − EPC (softmax head) | Ablation | **Isolates C1** |
| ARGUS − robust agg (mean) | Ablation | **Isolates C2** |
| ARGUS − temporal (L0) | Ablation | **Isolates C3** |
| OpenMax / Energy / ODIN | Post-hoc OSR on ARGUS encoder | Whether prototype geometry is needed |
| MSPL | Non-graph prototypical OSR | Closest non-graph mechanism |
| CLOSR | Non-graph contrastive OSR | Published OpenAUC comparator |

---

## 6. Ablation grid

| ID | Ablation | Values | Claim |
|---|---|---|---|
| E-ABL-AGG | Aggregation | mean / trimmed(β=0.1,0.25,0.4) / soft_medoid | C2 |
| E-ABL-K | Neighbour cap | 8 / 16 / 32 / 64 | C2 |
| E-ABL-SAMP | Sampling strategy | recent / uniform / recency_stratified | C2 |
| E-ABL-CH | Channel penalty `λ_ch` | 0 / 0.02 / 0.05 / 0.2 | C2 |
| E-ABL-EPC | Head type | epc / softmax / openmax / energy / distance_threshold | C1 |
| E-ABL-UNK | Synthetic unknowns | none / mixup / structural / both | C1 |
| E-ABL-T\* | Temporal ladder | L0 … L7 | **C3**, see `09_TEMPORAL_STUDY.md` |
| E-ABL-W | Window sizes | 9 combinations | C3 |
| E-ABL-DIM | `d_h`, `d_z` | 32 / 64 / 128 | — |
| E-ABL-DEPTH | Layers | 1 / 2 / 3 | — |

---

## 7. Statistical reporting

- 5 seeds minimum for every headline number; report mean ± std.
- For open-set, 5 seeds × 5 holdout sets = 25 runs; report mean ± std across the
  25 and note both sources of variance.
- Paired comparisons (ARGUS vs a baseline on identical splits/seeds) use a
  **paired bootstrap** over test flows, 10,000 resamples, reporting the 95% CI of
  the difference. State significance in terms of that CI, not a p-value.
- Where a difference is within noise, **say so**. A parity result on closed-set
  accuracy is the expected and honest outcome; overclaiming it invites rejection.

---

## 8. Figures and tables for the paper

| ID | Content | Claim |
|---|---|---|
| **T1** | Dataset statistics, class counts, unique-IP counts, split sizes | Setup |
| **T2** | Closed-set macro-F1 + per-class F1, ARGUS vs all baselines | C1 prereq |
| **T3** | Open-set: unknown TPR/FPR, AUROC, OpenAUC, known macro-F1, mean ± std over 5 holdouts | **C1** |
| **T3b** | **Within-family open-set (P-OS2): unknown TPR and sibling-absorption rate per held-out variant** | **C1** |
| **T4** | Few-shot: new-class F1 @ n, old-class delta (0.000), registration latency | **C1** |
| **T5** | Adversarial: robust macro-F1 and ASR at selected budgets | **C2** |
| **T6** | Temporal ladder L0–L7: macro-F1 + per-class deltas + latency cost | **C3** |
| **T7** | Cross-dataset transfer, zero-shot and after 20-shot registration | C1, C3 |
| **T8** | Calibration: ECE, MCE, Brier, NLL | C1 |
| **T9** | Deployment: throughput, latency percentiles, memory, model size | Deployability |
| **T10** | Ablation grid summary | C1, C2, C3 |
| **T13** | **Identity-leakage audit: identity floor, unseen-pair rate, identity-reliance gap** | methodological |
| **F1** | Architecture diagram | — |
| **F2** | Confusion matrix, primary dataset | C1 |
| **F3** | Vacuity histogram, known vs unknown | **C1** |
| **F4** | Reliability diagram, ARGUS vs softmax | C1 |
| **F5** | Multi-scale gate `g` by attack class | **C3** |
| **F6** | ASR vs injection budget, by aggregation type | **C2** |
| **F7** | Attribution mass on injected edges, robust vs mean agg | **C4** |
| **F8** | Risk–coverage curve | C1 |
| **F9** | Openness sweep | C1 |
| **F10** | Temporal ladder per-class heatmap | **C3** |

Bold rows are the claim-critical evidence. If schedule slips, protect these
first; drop T7, T9, F8, F9 before anything bold.
