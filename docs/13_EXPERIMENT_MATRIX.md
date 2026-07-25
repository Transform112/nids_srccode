# 13 — Experiment Matrix

Every run as a row. `scripts/15_make_tables.py` reads
`results/runs/registry.jsonl` and assembles the paper artifacts from these IDs.

Runners skip any `run_id` already present in the registry, so every script is
resumable across interrupted Kaggle sessions.

**Run ID convention:** `<experiment_id>__<dataset>__<variant>__s<seed>`
e.g. `E-TEMP-L3__ton_iot__default__s2`.

Dataset codes: `CIC` = NF-CICIDS2018 (**primary**, 14 classes),
`TI` = NF-ToN-IoT (secondary, 9 classes), `UNSW` = NF-UNSW-NB15 (ip_port nodes),
`BOT` = NF-BoT-IoT (degenerate).

> **The primary dataset changed from ToN-IoT to CICIDS2018** after profiling the
> real CSVs: CICIDS2018 has 14 fine-grained attack classes (the paper reported 6
> aggregated families), 181,876 source IPs, 11 active days, and a **class
> hierarchy** that enables the within-family open-set protocol B2. Wherever a
> table below says `CIC`, that is now the headline dataset. See
> `02_DATASETS.md` §2.3.

---

## 1. Baselines — P-CS

| ID | Model | Dataset | Split | Seeds | Runs | Feeds |
|---|---|---|---|---:|---:|---|
| E-BL-ET | Extra Trees | TI, CIC | Protocol A | 5 | 10 | T2 |
| E-BL-RF | Random Forest | TI, CIC | Protocol A | 5 | 10 | T2 |
| E-BL-MLP | MLP on `F_e` | TI, CIC | Protocol A | 5 | 10 | T2 |
| E-BL-ESAGE | E-GraphSAGE | TI, CIC, UNSW | Protocol A | 5 | 15 | T2, F6, T9 |
| E-BL-EGAT | EGATv2 / E-ResGAT | TI, CIC | Protocol A | 5 | 10 | T2 |
| E-BL-ANOM | Anomal-E | TI | Protocol A | 5 | 5 | T2 (binary) |
| E-BL-GIDS | GraphIDS | TI | Protocol A | 3 | 3 | T2 — *drop if not reproducible* |
| **Subtotal** | | | | | **63** | |

## 2. ARGUS closed-set — P-CS

| ID | Variant | Dataset | Seeds | Runs | Feeds |
|---|---|---|---:|---:|---|
| E-CS-FULL | Full ARGUS (L7) | TI, CIC, UNSW | 5 | 15 | T2, F2 |
| E-CS-DEGEN | Full ARGUS | BOT | 3 | 3 | Negative-result note |
| **Subtotal** | | | | **18** | |

## 3. Open-set — P-OS ⭐

5 holdout sets × 5 seeds = 25 runs per configuration.

| ID | Configuration | Dataset | Runs | Feeds |
|---|---|---|---:|---|
| E-OS-1 | ARGUS EPC head | TI | 25 | **T3**, F3, F9 |
| E-OS-1b | ARGUS EPC head | CIC | 25 | T3 |
| E-OS-2 | Openness sweep, `C_h ∈ {1,2,3,4}` | TI | 25 | **F9** |
| E-OS-BL1 | Softmax + threshold | TI | 25 | T3 |
| E-OS-BL2 | OpenMax | TI | 25 | T3 |
| E-OS-BL3 | Energy score | TI | 25 | T3 |
| E-OS-BL4 | ODIN | TI | 25 | T3 |
| E-OS-BL5 | MSPL (non-graph) | TI | 25 | T3 |
| E-OS-BL6 | CLOSR (non-graph) | TI | 25 | T3 |
| **Subtotal** | | | **225** | |

> Largest block in the matrix and the core of C1. If compute is tight, reduce
> seeds to 3 (→135 runs) **before** reducing the number of holdout sets. Holdout
> variance is the methodological contribution; seed variance is routine.

## 4. Few-shot registration — P-FS ⭐

| ID | Configuration | n values | Dataset | Runs | Feeds |
|---|---|---|---|---:|---|
| E-FS-1 | ARGUS registration | 1,5,10,20,50 | TI | 25 | **T4** |
| E-FS-2 | ARGUS registration | 1,5,10,20,50 | CIC | 25 | T4 |
| E-FS-BL1 | Softmax fine-tuning | 1,5,10,20,50 | TI | 25 | **T4** (forgetting contrast) |
| E-FS-BL2 | Full retraining (upper bound) | 20 | TI | 5 | T4 |
| E-FS-3 | UNSW `Worms` (158 samples) as registration target | 1,5,10,20 | UNSW | 20 | T4 — turns the rarest class into a C1 demo |
| **Subtotal** | | | | **100** | |

## 5. Cross-dataset transfer — P-TR

| ID | Direction | Mode | Runs | Feeds |
|---|---|---|---:|---|
| E-TR-1 | TI → CIC | zero-shot | 5 | T7 |
| E-TR-2 | TI → CIC | after 20-shot registration | 5 | **T7** |
| E-TR-3 | CIC → TI | zero-shot | 5 | T7 |
| E-TR-4 | CIC → TI | after 20-shot registration | 5 | T7 |
| **Subtotal** | | | **20** | |

## 6. Temporal ladder — P-TEMP ⭐

Encoder-only (Stage-1 nearest-prototype). Independent of the EPC head.

| ID | Rung | Adds | Dataset | Seeds | Runs |
|---|---|---|---|---:|---:|
| E-TEMP-L0 | L0 | v2 features only | TI | 5 | 5 |
| E-TEMP-L1 | L1 | + raw temporal columns | TI | 5 | 5 |
| E-TEMP-L2 | L2 | + TE1 conditioning | TI | 5 | 5 |
| E-TEMP-L3 | L3 | + TE2 derived | TI | 5 | 5 |
| E-TEMP-L4 | L4 | + TE3 Time2Vec | TI | 5 | 5 |
| E-TEMP-L5 | L5 | + TE4 decay attention | TI | 5 | 5 |
| E-TEMP-L6 | L6 | + TE5 multi-scale + TE6 memory | TI | 5 | 5 |
| E-TEMP-L7 | L7 | + TE7 spectral | TI | 5 | 5 |
| E-TEMP-CIC | L0, L3, L7 only | replication | CIC | 3 | 9 |
| **Subtotal** | | | | | **49** |

Feeds **T6**, **F10**, **F5**. The L0→L2 delta is a named finding
(`09_TEMPORAL_STUDY.md` §3.1).

## 7. Adversarial — P-ADV ⭐

| ID | Attack | Sweep | Configs | Runs | Feeds |
|---|---|---|---|---:|---|
| E-ADV-A1 | Feature PGD | ε × 6 | ARGUS, E-GraphSAGE, EGATv2 | 18 | T5 |
| E-ADV-A2 | **Structural injection, `all_strata`** | m × 8 | mean, trimmed(.1/.25/.4), soft_medoid, E-GraphSAGE | 48 | **T5, F6** |
| E-ADV-A2b | Structural injection, `single_stratum` | m × 8 | same 6 | 48 | **F6** (sampling-defence gap) |
| E-ADV-A2c | Sampling strategy | m × 8 | recent, uniform, recency_stratified | 24 | F6 |
| E-ADV-A3 | Prototype poisoning | p × 5 | gate on / gate off | 10 | T5 |
| E-ADV-A4 | Adaptive white-box | — | ARGUS, ARGUS-mean | 2 | T5 |
| E-ADV-A5 | Temporal jitter | σ × 5 | L3, L7 (per-component) | 10 | T5, temporal robustness |
| **Subtotal** | | | | **160** | |

## 8. Ablations

| ID | Ablation | Values | Dataset | Seeds | Runs | Feeds |
|---|---|---|---|---:|---:|---|
| E-ABL-AGG | Aggregation | 5 | TI | 3 | 15 | T10, F6 |
| E-ABL-K | Neighbour cap | 4 | TI | 3 | 12 | T10 |
| E-ABL-SAMP | Sampling | 3 | TI | 3 | 9 | T10 |
| E-ABL-CH | `λ_ch` | 4 | TI | 3 | 12 | T10 |
| E-ABL-EPC | Head type | 5 | TI | 3 | 15 | T10 |
| E-ABL-UNK | Synthetic unknowns | 4 | TI | 3 | 12 | T10 |
| E-ABL-W | Window sizes | 9 | TI | 3 | 27 | T10 |
| E-ABL-DIM | `d_h`/`d_z` | 3 | TI | 3 | 9 | T10 |
| E-ABL-DEPTH | Layers | 3 | TI | 3 | 9 | T10 |
| **Subtotal** | | | | | **120** | |

## 9. Calibration, XAI, deployment, timeout

| ID | Content | Runs | Feeds |
|---|---|---:|---|
| E-CAL-1 | ECE/MCE/Brier/NLL, ARGUS vs softmax vs OpenMax | 15 | **T8**, F4 |
| E-SEL-1 | Risk–coverage sweep of `θ_defer` | 5 | F8 |
| E-XAI-1 | Explanation quality, 5 explainers × 1000 decisions | 5 | T11 |
| E-XAI-2 | **Injected-edge attribution vs budget** | 12 | **F7** |
| E-XAI-3 | UNKNOWN triage clustering | 5 | T12 |
| E-DEP-1 | Throughput/latency/memory, ARGUS + E-GraphSAGE + rungs | 10 | **T9** |
| E-TO-1 | Flow-timeout stability, 4 timeout settings | 12 | P-TO |
| **Subtotal** | | **64** | |

---

## 10. Totals and budget

| Block | Runs | Priority |
|---|---:|---|
| Baselines | 63 | **must** |
| ARGUS closed-set | 18 | **must** |
| Open-set | 225 | **must** |
| Few-shot | 100 | **must** |
| Transfer | 20 | should |
| Temporal ladder | 49 | **must** |
| Adversarial | 160 | **must** (A2 core) |
| Ablations | 120 | should |
| Cal/XAI/Dep/TO | 64 | mixed |
| **Total** | **819** | |

Most runs are short. Open-set, few-shot, adversarial, and XAI runs are
**inference-only against an already-trained checkpoint** — only the following
require training:

| Requires training | Count |
|---|---:|
| Baselines | 63 |
| ARGUS closed-set | 18 |
| Temporal ladder | 49 |
| Ablations | 120 |
| Open-set (retrain per holdout) | 225 |
| **Training runs** | **475** |

Open-set dominates because each holdout requires retraining without those
classes. **Mitigation:** train one checkpoint per (holdout set × seed) = 25
checkpoints, and reuse each across all 9 open-set configurations — the baselines
(softmax/OpenMax/energy/ODIN) are alternative *heads* on the same encoder, not
separate encoders. That reduces open-set training from 225 to **25**, and total
training runs to **275**.

Implement that reuse in `scripts/07_eval_open_set.py`; it is the single largest
compute saving available.

---

## 11. Execution order

Ordered so that each stage de-risks the next and produces a usable partial paper.

```
1. E-BL-*                    baselines             → validates the data pipeline
2. E-CS-FULL                 ARGUS closed-set      → validates the encoder
3. E-TEMP-L0..L7             temporal ladder       → C3 secured, independent of P4
4. E-OS-1, E-OS-BL1..BL4     open-set core         → C1 secured
5. E-ADV-A2, A2b, A2c        structural injection  → C2 secured
6. E-FS-1..3                 few-shot              → C1 completed
7. E-XAI-2                   attribution vs budget → C4 secured
8. E-CAL-1, E-SEL-1          calibration/selective → C1 supporting
9. E-ABL-*                   ablations             → reviewer defence
10. E-TR-*, E-DEP-1, E-TO-1  transfer/deployment   → deployability
11. E-ADV-A1/A3/A4/A5        remaining attacks     → completeness
12. E-XAI-1/E-XAI-3          explanation quality   → C4 completed
```

After step 7 all four claims have supporting evidence and a submittable draft
exists. Steps 8–12 strengthen it. **If a deadline bites, stop after step 9.**

---

## 12. Registry schema

One JSON object per line in `results/runs/registry.jsonl`:

```json
{
  "run_id": "E-TEMP-L3__ton_iot__default__s2",
  "experiment_id": "E-TEMP-L3",
  "dataset": "ton_iot",
  "variant": "default",
  "seed": 2,
  "phase": "P7",
  "config_hash": "sha256:...",
  "split_hash": "sha256:...",
  "feature_manifest_hash": "sha256:...",
  "git_sha": "abc1234",
  "status": "complete",
  "started_utc": "2026-07-26T09:14:00Z",
  "wall_clock_s": 2431,
  "gpu": "Tesla P100-PCIE-16GB",
  "metrics": {"macro_f1": 0.8412, "macro_f1_std": 0.0071, "ece": 0.031},
  "artifacts": ["results/runs/.../metrics.json", "results/runs/.../ckpt_epoch12.pt"]
}
```

`scripts/15_make_tables.py` joins on `experiment_id` to build each table, and
emits a provenance column listing contributing `run_id`s. Any table cell without
backing runs renders as `—` with a footnote, so partial results are never
mistaken for complete ones.
