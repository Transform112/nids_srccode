# AGENT_GUIDE.md — Master brief for an implementing agent

**Read this file first and completely.** It is the single entry point for any AI
agent or engineer picking up this project. It tells you what the project is, what
already exists, what to build, in what order, what will go wrong, and how to
diagnose it. Everything referenced here is in `docs/`.

If you follow this guide you should not need to search the web, guess at a
design decision, or invent a number. Every value has already been chosen and
justified.

---

## 0. Thirty-second summary

Build **ARGUS**: a graph neural network for network intrusion detection whose
classifier head is a *prototype bank + evidential (Dirichlet) uncertainty layer*
instead of a softmax.

From one forward pass it produces: a known-class prediction, a calibrated
*defer* signal, or an *UNKNOWN* verdict for a novel attack. New attack classes
are added by appending a prototype computed from a handful of labelled samples —
**zero gradient steps, therefore provably zero forgetting**. The encoder
underneath is hardened against structural adversarial attacks, and the temporal
representation is measured rather than assumed.

Target output: an 8–10 page IEEE/ACM conference paper (NOMS / TrustCom / ICC /
EuroS&P workshop).

**Status: no code exists yet.** `docs/` and `dataset/` are complete. You are
starting at phase P0.

---

## 1. Why this project exists — the gap

Two arXiv full-text searches were run on 2026-07-25 and are recorded in
`docs/00_OVERVIEW.md` §2:

| Query | Results | Finding |
|---|---:|---|
| GNN + intrusion detection + (open-set OR zero-day OR continual) | **3** | none does principled open-world recognition |
| intrusion detection + (open-set recognition OR prototype learning OR evidential) | **9** | **all flow-independent/tabular; none graph-based** |

So: open-world recognition and structural robustness are each absent from
GNN-NIDS. Two 2025 papers (Venturi et al. arXiv:2403.11830; REAL-IoT
arXiv:2507.10836) show existing GNN-NIDS collapse under structural attacks and
realistic drift.

**Do not weaken this framing.** "First open-world GNN-NIDS" is defensible and
verified. If you find contrary evidence while implementing, record it in
`docs/01_RELATED_WORK.md` rather than quietly softening the claim.

---

## 2. The four claims — everything traces to one of these

| ID | Claim | Primary evidence | Doc |
|---|---|---|---|
| **C1** | First open-world GNN-NIDS: classify / defer / unknown from one head, plus few-shot class registration with zero forgetting | T3, T4, F3 | `05` §6, `08` §3 |
| **C2** | Structure-robust message passing with a stated breakdown point | T5, F6 | `05` §3, `10` |
| **C3** | A temporal representation that demonstrably contributes, per class | T6, F10, F5 | `09` |
| **C4** | Explanation-verified robustness — injected edges get near-zero attribution | F7 | `11` §3 |

If a piece of work does not serve one of these four, it is out of scope.

---

## 3. What already exists

```
new_ids/
├── AGENT_GUIDE.md      ← you are here
├── README.md           index + non-negotiable rules
├── docs/               00–14 + TODO — the complete specification
└── dataset/            13.3 GB of raw NetFlow-v3 CSVs, READ-ONLY
```

### 3.1 The data

Four CSVs, **identical 55-column schema** (53 features + `Label` + `Attack`).
All statistics below were measured directly from these files, not copied from
the source paper.

| File | Size | Rows | src IPs | Attack classes | Role |
|---|---:|---:|---:|---:|---|
| `NF-CICIDS2018-v3.csv` | 4.03 GB | 20,115,529 | 181,876 | **14** | **PRIMARY** |
| `NF-ToN-IoT-v3.csv` | 5.06 GB | 27,520,260 | 15,270 | 9 | SECONDARY |
| `NF-UNSW-NB15-v3.csv` | 0.55 GB | 2,365,424 | **40** | 9 | tertiary |
| `NF-BoT-IoT-v3.csv` | 3.64 GB | 16,933,808 | **20** | 4 | degenerate |

Column order on disk starts with the two timestamps — **never index columns
positionally, always by name.**

### 3.2 Documents, in reading order

| Doc | Read when |
|---|---|
| `docs/00_OVERVIEW.md` | Now — problem, gap, claims, scope, glossary |
| `docs/02_DATASETS.md` | **Before any data code. Mandatory.** Three traps live here |
| `docs/03_FEATURE_ENGINEERING.md` | Building the feature pipeline |
| `docs/04_GRAPH_CONSTRUCTION.md` | Building windows/graphs/sampling |
| `docs/05_ARCHITECTURE.md` | Building the model. Every equation and shape |
| `docs/06_TRAINING.md` | Building the training loop |
| `docs/07_HYPERPARAMETERS.md` | Any time you need a number |
| `docs/12_IMPLEMENTATION_PLAN.md` | **Directory layout + full config schema** |
| `docs/08_EVALUATION.md` | Building metrics |
| `docs/09_TEMPORAL_STUDY.md` | Running the temporal ladder (C3) |
| `docs/10_ADVERSARIAL.md` | Building attacks (C2) |
| `docs/11_XAI.md` | Building explanations (C4) |
| `docs/13_EXPERIMENT_MATRIX.md` | Running experiments — 819 runs, all enumerated |
| `docs/14_PAPER_OUTLINE.md` | Writing |
| `docs/01_RELATED_WORK.md` | Writing — ~35 citations with IDs |
| `docs/TODO.md` | Tracking |

---

## 4. The three traps — read before writing data code

These were found by profiling the actual CSVs. Each one silently invalidates
results if missed. Full detail in `docs/02_DATASETS.md` §3, §4, §7.

### TRAP 1 — unique-IP degeneracy

`NF-UNSW-NB15` has **40 unique source IPs**; `NF-BoT-IoT` has **20**. An IP-level
graph over 40 nodes is near-complete: message passing degenerates to global
pooling, the degree cap becomes meaningless, and the C2 breakdown-point argument
collapses because "the attacker's share of a neighbourhood" is undefined.

**Rule:** IP nodes for CICIDS2018 and ToN-IoT; **IP:port composite nodes** for
UNSW-NB15. BoT-IoT is a documented negative result only. Enforce with a hard
error via `min_unique_src_ip: 1000` — fail, do not warn.

### TRAP 2 — attacks segregated by capture day

Measured mapping, e.g. CICIDS2018: `FTP-BruteForce` day 1 only, `Bot` day 11
only, `Infiltration` days 9–10 only. A chronological 70/15/15 split puts entire
classes on one side of the boundary and makes macro-F1 meaningless.

**Rule:** closed-set uses a **per-class stratified temporal split** (Protocol A).
Open-set *exploits* the day structure deliberately (Protocol B / B2).

### TRAP 3 — host identity leakage

Attacker hosts are a small fixed set. A model can score well by memorising
*which hosts are malicious* rather than *which behaviour is malicious*. Graph
models are **more** exposed than flow-independent ones, because message passing
and the per-node GRU give identity extra routes into the prediction.

**Rule:** run the identity-leakage audit (`docs/02_DATASETS.md` §7.1) before any
modelling, report the identity floor in the paper, and use memory dropout
`p_mem = 0.1`.

---

## 5. Architecture in one page

Full spec with equations and tensor shapes: `docs/05_ARCHITECTURE.md`.

```
edge features x_e ─┬─ Channel A proj (controllable, d_A=48)  ─┐
                   └─ Channel B proj (observer,     d_B=80)  ─┤
node features x_v ────────────────────────────────────────────├─→ SR-TEG per scale
Δt ─→ Time2Vec φ(Δt) ─────────────────────────────────────────┘        │
                                                                        ↓
                          h_e^S, h_e^M, h_e^L ──→ gated fusion (1s/30s/300s)
                                                                        ↓
                                              projection + L2 norm → z_e ∈ S^63
                                                                        ↓
              multi-prototype bank → cosine → log-space Dirichlet evidence
                                                                        ↓
                                        {CLASSIFY c | DEFER | UNKNOWN}
```

**SR-TEG encoder** — provenance-partitioned edge channels; Time2Vec Δt encoding
with log-grid initialisation; time-decayed attention (decay as an additive
log-space bias inside the softmax); **trimmed-mean aggregation** (β = 0.20) plus
a multi-aggregator readout; per-node GRU memory with dropout and eviction;
weight-shared across three concurrent time scales with a learned fusion gate.

**EPC head** — unit-norm embedding; **multi-prototype bank** (4 sub-prototypes
for benign, 2 for large attack classes, 1 for small); cosine → evidence computed
**in log space**; three-way decision on total evidence and margin.

### 5.1 Five design decisions you must not silently reverse

1. **Never BatchNorm.** Five independent reasons in `docs/05_ARCHITECTURE.md`
   §9.1 — the decisive ones are that batch statistics under 87% class imbalance
   are effectively benign statistics, that single-flow streaming inference breaks
   the train/eval contract, and that BatchNorm is an *attack surface* (a sample's
   normalisation depends on other samples in its batch). Use LayerNorm /
   GraphNorm. The config validator rejects `batchnorm`.
2. **Evidence must be computed in log space.** `exp(−(d−m)/τ)` with `τ = 0.1` has
   an exponent range of `[−16.5, +3.5]` and multiplies every head gradient by 10.
   Naïve evaluation overflows fp16 and explodes fp32. Clamp `log e` to ±15, use
   `logsumexp` for `S`, keep the head in fp32 under AMP, and **anneal τ from 1.0
   to 0.1**.
3. **Multi-prototype is not optional.** Benign is 61–87% of traffic and strongly
   multi-modal. A single benign cone either swallows attacks (unknown detection
   collapses) or rejects normal traffic. This is the most likely cause of a
   disappointing OpenAUC.
4. **`recency_stratified` sampling, not "most recent K".** Pure recency sampling
   is trivially attackable — an adversary emits K flows just before the victim's
   and owns the entire neighbourhood regardless of the robust aggregator, which
   would void C2.
5. **Never a random split.** Protocol A or B only. See TRAP 2.

---

## 6. Build order

Do not begin a phase until the previous phase's acceptance criteria pass. Full
criteria in `docs/12_IMPLEMENTATION_PLAN.md` §3.

| Phase | Build | Gate to pass |
|---|---|---|
| **P0** | Repo skeleton, `config/default.yaml`, `constants.py` | `ruff check && pytest` runs clean |
| **P1** ⭐ | Data substrate: clean, canonicalise, subsample, split, audit, features | Split audit passes all 4 hard checks; feature manifest emitted |
| **P2** | Baselines (parallel with P3) | E-GraphSAGE reproduces on *our* split |
| **P3** ⭐ | SR-TEG encoder | Closed-set parity with E-GraphSAGE; G0–G2 pass |
| **P4** ⭐ | EPC head — **highest risk** | `register_class` changes zero parameters; OpenAUC beats softmax |
| **P5** | Adversarial A1–A5 | Every perturbed vector satisfies domain constraints |
| **P6** | XAI (parallel with P5) | Attribution sums to log-evidence within tolerance |
| **P7** | Temporal ladder (may start after P3) | All 8 rungs × 5 seeds, or partials clearly marked |
| **P8** | Deployment measurement + writing | Every number traces to a `run_id` |

**P2 is deliberately early.** If E-GraphSAGE cannot be reproduced on our split,
that is a data-pipeline bug, and it is far cheaper to find at P2 than at P4.

**P7 depends only on P3**, because the temporal ladder runs encoder-only. This
de-risks C3 against P4's failure — run it in parallel.

---

## 7. Where this will go wrong, and what to do

Ordered by likelihood. Each has a documented mitigation; none should be a
surprise.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | NaN loss in Stage 2 | Evidence expression overflow | Log space, clamp ±15, fp32 head, check τ anneal. `05` §6.3 |
| 2 | Stage 2 will not converge | Evidential head is genuinely finicky | **Documented fallback:** `--set head.type=distance_threshold`. Preserves all of C1's few-shot claim and most of the open-set claim. Decide by end of P4; **do not let this block the paper** |
| 3 | Model cannot beat E-GraphSAGE | Expressiveness lost to robust aggregation | Multi-aggregator readout (`05` §3.5b); lower `trim_beta`; raise `d_h`. Also: **parity is sufficient** — accuracy is not the contribution |
| 4 | Suspiciously high accuracy | Host identity memorisation | Run the identity audit (`02` §7.1). If the identity floor is high, disclose it |
| 5 | OpenAUC disappointing | Single benign prototype | Confirm multi-prototype is active and `L_div` is preventing collapse (gate G2b) |
| 6 | Prototypes collapse together | Margin/compactness too weak | Raise `m` or `λ_cmp` (gate G2) |
| 7 | Temporal features "don't help" | TE1 conditioning missing | IAT spans 6+ orders of magnitude; without signed-log + quantile the columns are numerically dead. This is rung L1→L2 of the ladder |
| 8 | Model can't overfit 1,000 samples | Bug, not capacity | **Gate G0.** Likely: constant columns from a broken pipeline, features silently dropped by an incomplete channel partition, or a detached tensor |
| 9 | Train/val gap widens | Overfitting | Raise DropEdge/dropout; check identity leakage (gate G7) |
| 10 | Kaggle session dies mid-run | 12 h limit | Per-epoch checkpointing + run registry; every runner skips completed runs |

---

## 8. Configuration

Everything tunable is in config; **no magic numbers in code**. Full key list in
`docs/12_IMPLEMENTATION_PLAN.md` §4.2, validation rules in §4.3.

Layering: `default.yaml` → `dataset/<name>.yaml` → `model/<name>.yaml` →
`experiment/<name>.yaml` → CLI `--set key=value`. The merged config is written
verbatim to `results/runs/<run_id>/config_resolved.yaml`.

The validator must **raise on unknown keys** — this catches typo'd overrides that
would otherwise silently no-op and waste a full training run.

```bash
# Smoke test on a laptop before anything else
--set data.subsample_target=200000 --set train.stage1_epochs=2 \
--set run.device=cpu --set graph.neighbour_cap=8

# Ablations that isolate each claim
--set head.type=softmax        # isolates C1
--set model.aggregation=mean   # isolates C2
# temporal ladder rungs: toggle model.te3..te6_enabled, features.te1/te2/te7_enabled  → C3
```

---

## 9. Standing rules

Violating any of these silently invalidates results.

1. Never a random train/test split — Protocol A or B only.
2. Never IP-level nodes on UNSW-NB15 or BoT-IoT.
3. Fit all scalers and encoders on **train only**; verify via the provenance record.
4. Never quote a baseline number from another paper — re-run on our splits.
5. Always report per-class F1 alongside macro-F1. **Never headline accuracy** —
   at 0.0022% prevalence for the rarest class it is meaningless.
6. Thresholds are selected on **validation only**; report sweeps as curves.
7. Recompute TE2 derived features after any adversarial perturbation — never
   perturb them independently, or the attacker gets physically impossible flows.
8. Every reported number traces to a `run_id` in `results/runs/registry.jsonl`.
9. Report mean ± std over ≥3 seeds. Single-seed numbers appear nowhere.
10. Never BatchNorm.

---

## 10. What "done" looks like

The paper is publishable when all six hold (`docs/00_OVERVIEW.md` §6):

1. Closed-set macro-F1 at **parity or better** vs re-run GNN baselines.
2. Open-set beats softmax-threshold / OpenMax / energy on OpenAUC, averaged over
   ≥5 random class holdouts.
3. Few-shot registration usable at n ≤ 20 with **exactly zero** degradation on
   known classes.
4. Under structural attack A2, ARGUS retains substantially more macro-F1 than
   E-GraphSAGE at equal injection budget.
5. Temporal ladder shows a measurable, per-class-attributable gain.
6. ECE materially below a softmax baseline.

After step 7 of the execution order in `docs/13_EXPERIMENT_MATRIX.md` §11, all
four claims have supporting evidence and a submittable draft exists.

---

## 11. Descoping order

If time runs short, drop in this order. Never drop anything below the line.

```
DROP FIRST  →  NF-BoT-IoT degenerate case
               Flow-timeout ablation (P-TO)
               A3 prototype poisoning, A4 adaptive attacker
               GraphIDS baseline
               Cross-dataset reverse direction
               NF-UNSW-NB15 entirely
──────────────────────────────────────────────────
NEVER DROP  →  Open-set holdout variance (5 holdouts, mean ± std)
               Calibration metrics (ECE + reliability diagram)
               Temporal ladder rungs L0–L3
               A2 structural injection
               Few-shot zero-forgetting demonstration
               Identity-leakage audit
               Re-running baselines on our splits
```

---

## 12. Prompt to hand to a fresh agent

Copy the block below verbatim when starting a new session.

> You are implementing **ARGUS**, an open-world, structure-robust, temporally
> grounded graph neural network for network intrusion detection, targeting an
> IEEE/ACM conference paper.
>
> **First action: read `AGENT_GUIDE.md` in the workspace root, then
> `docs/02_DATASETS.md` and `docs/12_IMPLEMENTATION_PLAN.md` in full.** Those
> three files contain the complete specification — architecture, equations,
> hyperparameters, directory layout, config schema, experiment matrix, and the
> three data traps. Do not search the web and do not invent design decisions;
> every choice has already been made and justified. If something appears
> underspecified, check `docs/07_HYPERPARAMETERS.md` before deciding anything
> yourself, and record any decision you do make.
>
> The raw data is in `dataset/` (four NetFlow-v3 CSVs, 13.3 GB, identical
> 55-column schema, read-only). No code exists yet; start at phase P0 in
> `docs/12_IMPLEMENTATION_PLAN.md` §3.
>
> Non-negotiable rules: never a random split; never IP-level nodes on
> UNSW-NB15 or BoT-IoT; fit scalers on train only; never BatchNorm; compute the
> Dirichlet evidence in log space; never headline accuracy. The full list is
> §9 of the guide.
>
> Work phase by phase. Do not begin a phase until the previous phase's
> acceptance criteria pass. Run gate G0 — the 1,000-sample overfit capacity
> check — before any full training run, and every time the architecture changes.
> Update `docs/TODO.md` as you complete items.
