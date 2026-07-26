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

Specification complete and validated against the real data. Every phase
P0–P8 has a real, non-stub implementation; remaining work is running the
full-scale sweeps on Kaggle and writing the paper. `docs/TODO.md` is the
authoritative, actively-maintained checklist — this section is a summary of
it, and may lag; when in doubt, trust `docs/TODO.md`.

- **P0–P3** (repo scaffold, config system, data substrate, feature pipeline,
  graph construction incl. real node features + TE7 spectral descriptor,
  SR-TEG encoder): complete and tested. A local full-scale run of the real
  20.1M-row CICIDS2018 CSV through the data substrate (all 15 classes) has
  been executed successfully, including a precomputed graph-batch disk cache
  (`scripts/03_cache_graphs.py`, `src/argus/graph/cache.py`) that replays
  bins from disk instead of rebuilding them every epoch.
- **P4** (EPC head, two-stage training, channel-penalty loss, validation-only
  threshold calibration): complete and tested.
- **P2** (baselines): Extra Trees, Random Forest, MLP, identity-only leakage
  floor, E-GraphSAGE, EGATv2, Anomal-E, and post-hoc OSR (OpenMax/energy/ODIN)
  all implemented; tabular + identity-only + E-GraphSAGE verified end-to-end
  on real data, the rest unit-tested.
- **P5** (adversarial): all five attacks implemented and tested — domain-constraint
  projection, the headline A2 structural-injection attack (budget sweep, both
  spread strategies), A1 feature-space evasion and A4 adaptive white-box
  attacker (both via SPSA-estimated ascent through the frozen, non-
  differentiable feature pipeline), A3 prototype poisoning (added
  `PrototypeBank.ema_update`, the drift-correction mechanism it targets), and
  A5 temporal jitter with attacker-cost accounting. `scripts/11_run_adversarial.py`
  orchestrates all five; full-scale sweeps on real data not yet run.
- **P6** (XAI): native evidence attribution (exact embedding-level
  decomposition + Integrated Gradients feature attribution, both verified
  against the completeness axiom), baseline explainers (GNNExplainer,
  PGExplainer, KernelSHAP, attention-only, counterfactual necessity), and
  UNKNOWN-cluster triage all implemented and tested; not yet run on real data.
- **P7** (temporal ladder): implemented (`scripts/10_run_temporal_ladder.py`,
  `config/experiment/temporal_ladder.yaml`, 8 rungs × 5 seeds); not yet run.
- **P8** (deployment): `StreamingDetector` (`push()` / `register_class()`)
  with no-lookahead and bounded-memory tests, plus aggregated throughput/
  latency/model-size reporting (`src/argus/eval/deployment.py`,
  `scripts/13_measure_deployment.py`), implemented and tested.

**97 tests passing** under `tests/`, plus an end-to-end laptop smoke test at
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

**Known limitation:** graph construction (not the encoder's per-node
aggregation) is the throughput bottleneck — building one anchor bin takes a
few seconds, and at the documented default `anchor_bin_seconds=1` a full
CICIDS2018 epoch produces ~62,000 bins, which does not fit a single Kaggle
GPU session. Two mitigations are in place: `scripts/03_cache_graphs.py`
precomputes and gzip-compresses every bin once so later epochs replay from
disk in ~50ms/bin instead of rebuilding (`src/argus/graph/cache.py` guards
against silently reusing a cache built with different `graph.*` settings);
and `config/dataset/cicids2018.yaml` coarsens `anchor_bin_seconds`/
`window_short_seconds` to 10s as a compute-budget deviation from the
documented `{1,5}`/`{0.5,1,2}` ranges (`docs/07_HYPERPARAMETERS.md` §3,
tracked in `docs/TODO.md` "Open decisions"). Full Kaggle-scale training runs
(P3 parity check, P4 Stage-1/2 training, then P5–P8 sweeps) are the next
milestone — see `docs/TODO.md` for the itemised list.
