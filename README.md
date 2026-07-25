# ARGUS — Open-World, Structure-Robust, Temporally-Grounded Graph NIDS

Research project targeting an IEEE/ACM conference paper (NOMS / TrustCom / ICC /
EuroS&P workshop), 8–10 pages.

**One-line thesis.** A graph neural network for network intrusion detection whose
classifier head is a *prototype bank + evidential (Dirichlet) uncertainty layer*
instead of a softmax — yielding known-class prediction, calibrated deferral, and
zero-day rejection from one forward pass, with new attack classes registered from
a handful of labelled samples using **zero gradient steps**; built on an encoder
hardened against problem-space structural adversarial attacks, and grounded in a
temporal representation whose contribution is measured, not assumed.

---

## Reading order

| # | Document | Purpose |
|---|---|---|
| — | [AGENT_GUIDE.md](AGENT_GUIDE.md) | **Master brief — read first** |
| 0 | [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md) | Problem, gap, contributions, scope, glossary |
| 1 | [docs/01_RELATED_WORK.md](docs/01_RELATED_WORK.md) | Citation table, competitor analysis, differentiation |
| 2 | [docs/02_DATASETS.md](docs/02_DATASETS.md) | Measured statistics, **the three traps**, split protocols, imbalance |
| 3 | [docs/03_FEATURE_ENGINEERING.md](docs/03_FEATURE_ENGINEERING.md) | TE1/TE2 transforms, provenance partition, exact formulas |
| 4 | [docs/04_GRAPH_CONSTRUCTION.md](docs/04_GRAPH_CONSTRUCTION.md) | Node granularity, multi-scale windows, sampling |
| 5 | [docs/05_ARCHITECTURE.md](docs/05_ARCHITECTURE.md) | SR-TEG encoder + EPC head, equations, normalisation, failure modes |
| 6 | [docs/06_TRAINING.md](docs/06_TRAINING.md) | Losses, two-stage schedule, regularisation, sanity gates |
| 7 | [docs/07_HYPERPARAMETERS.md](docs/07_HYPERPARAMETERS.md) | Every hyperparameter, default, range, guard values |
| 8 | [docs/08_EVALUATION.md](docs/08_EVALUATION.md) | 11 protocols, metric formulas, baselines, ablations |
| 9 | [docs/09_TEMPORAL_STUDY.md](docs/09_TEMPORAL_STUDY.md) | TE1–TE7 and the L0→L7 ablation ladder |
| 10 | [docs/10_ADVERSARIAL.md](docs/10_ADVERSARIAL.md) | Threat model, attacks A1–A5, pseudocode |
| 11 | [docs/11_XAI.md](docs/11_XAI.md) | Native attribution, explainer baselines, metrics |
| 12 | [docs/12_IMPLEMENTATION_PLAN.md](docs/12_IMPLEMENTATION_PLAN.md) | **Directory structure + full config schema**, build order |
| 13 | [docs/13_EXPERIMENT_MATRIX.md](docs/13_EXPERIMENT_MATRIX.md) | Every run as a row, mapped to paper tables |
| 14 | [docs/14_PAPER_OUTLINE.md](docs/14_PAPER_OUTLINE.md) | Section outline, claim→evidence map |
| — | [docs/TODO.md](docs/TODO.md) | Tracked checklist |

**If you are an AI agent or a new engineer, read
[AGENT_GUIDE.md](AGENT_GUIDE.md) first** — it is the single-file brief covering
the idea, the data, the architecture, the build order, the known failure modes,
and a ready-to-paste prompt for a fresh session. Then read 00 → 02 → 12, then
03–06. **Documents 02 and 12 are mandatory before writing any code.**

---

## The data

`dataset/` holds four NetFlow-v3 CSVs (13.3 GB), read-only, with an identical
55-column schema (53 features + `Label` + `Attack`).

| File | Rows | src IPs | Attack classes | Role |
|---|---:|---:|---:|---|
| `NF-CICIDS2018-v3.csv` | 20,115,529 | 181,876 | **14** | **PRIMARY** |
| `NF-ToN-IoT-v3.csv` | 27,520,260 | 15,270 | 9 | SECONDARY |
| `NF-UNSW-NB15-v3.csv` | 2,365,424 | **40** | 9 | tertiary (IP:port nodes) |
| `NF-BoT-IoT-v3.csv` | 16,933,808 | **20** | 4 | degenerate case only |

All statistics in the docs were **measured from these files**, not copied from
the source paper — which differs in four places, two of them consequential. See
[docs/02_DATASETS.md](docs/02_DATASETS.md) §2.

---

## Non-negotiable rules

These are recorded because violating them silently invalidates results.

1. **Never use a random train/test split.** Protocol A or B only. See
   [docs/02_DATASETS.md](docs/02_DATASETS.md) §4.
2. **Never use IP-level nodes on NF-UNSW-NB15 or NF-BoT-IoT.** They have 40 and
   20 unique source IPs. See §3 (TRAP 1).
3. **Never use a plain chronological split.** Attacks are segregated by capture
   day. See §4 (TRAP 2).
4. **Run the identity-leakage audit before modelling** and report the identity
   floor. See §7 (TRAP 3).
5. **Fit all scalers/encoders on the training split only.** See
   [docs/03_FEATURE_ENGINEERING.md](docs/03_FEATURE_ENGINEERING.md) §1.
6. **Never use BatchNorm.** Five reasons, one of them an attack surface. See
   [docs/05_ARCHITECTURE.md](docs/05_ARCHITECTURE.md) §9.1.
7. **Compute the Dirichlet evidence in log space.** Naive evaluation overflows
   and explodes. See [docs/05_ARCHITECTURE.md](docs/05_ARCHITECTURE.md) §6.3.
8. **Never quote baseline numbers from other papers.** Re-run on our splits.
9. **Always report per-class F1 alongside macro-F1; never headline accuracy.**
   The rarest class is 0.0022% of the primary dataset.
10. **Run gate G0** — the 1,000-sample overfit capacity check — before any full
    training run. See [docs/06_TRAINING.md](docs/06_TRAINING.md) §8.1.

---

## Status

Specification complete and validated against the real data. **Implementation
in progress**, spanning every phase P0–P8 at varying depth:

- **P0–P3** (repo scaffold, config system, data substrate, feature pipeline,
  graph construction incl. real node features + TE7 spectral descriptor,
  SR-TEG encoder): complete and tested.
- **P4** (EPC head, two-stage training, channel-penalty loss, validation-only
  threshold calibration): complete and tested.
- **P2** (baselines): Extra Trees, Random Forest, MLP, identity-only leakage
  floor, E-GraphSAGE, and EGATv2 (GATv2 attention + residual connections)
  implemented and verified end-to-end/unit-tested; Anomal-E and post-hoc OSR
  baselines not yet built.
- **P5** (adversarial): all five attacks implemented and tested — domain-constraint
  projection, the headline A2 structural-injection attack (budget sweep, both
  spread strategies), A1 feature-space evasion and A4 adaptive white-box
  attacker (both via SPSA-estimated ascent through the frozen, non-
  differentiable feature pipeline), A3 prototype poisoning (added
  `PrototypeBank.ema_update`, the drift-correction mechanism it targets), and
  A5 temporal jitter with attacker-cost accounting. `scripts/11_run_adversarial.py`
  orchestrates all five; full-scale sweeps on real data not yet run.
- **P6** (XAI): native evidence attribution (exact embedding-level
  decomposition + Integrated Gradients feature attribution), both verified
  against the completeness axiom; baseline explainers and UNKNOWN triage not
  yet built.
- **P7** (temporal ladder): not yet implemented.
- **P8** (deployment): `StreamingDetector` (`push()` / `register_class()`)
  implemented with no-lookahead and bounded-memory tests; aggregated
  throughput/latency reporting not yet built.

**77 tests passing** under `tests/`, plus an end-to-end laptop smoke test at
`scripts/00_smoke_test.py` that runs the full P1–P4 pipeline on a real slice of
`dataset/NF-CICIDS2018-v3.csv`. See `docs/TODO.md` for the exact per-item
checklist.

Run the smoke test:

```powershell
$env:PYTHONPATH = "src"
python scripts/00_smoke_test.py
```

Numbered scripts (`scripts/01_prepare_data.py` … `05_train_head.py`,
`scripts/14_run_baselines.py`) can be run individually; see each script's
docstring for laptop vs. Kaggle-scale usage (the `--nrows`, `--set
data.subsample_target=...`, and `--set run.device=cuda` overrides control
scale; `--max-bins` bounds anchor-bin processing for quick dev runs).

**Known limitation:** the current SR-TEG encoder computes per-node aggregation
with a Python loop over nodes per layer (`src/argus/models/srteg.py`), which is
correct but not yet vectorised for large graphs. On real timestamps spanning
hours, the number of anchor bins (one per second by default) makes a full
training epoch slow; `graph.anchor_bin_seconds`, `train.stage1_epochs`, and the
`max_bins` argument to `train_stage1`/`train_stage2` can bound this for
development. Full Kaggle-scale throughput work (vectorising the per-node loop,
precomputed graph batch caching per `docs/12_IMPLEMENTATION_PLAN.md` §4.5) is
the next milestone. The temporal ablation ladder (P7) and the remaining
adversarial/XAI/baseline modules are the next scope to close — see
`docs/TODO.md` for the itemised list.
