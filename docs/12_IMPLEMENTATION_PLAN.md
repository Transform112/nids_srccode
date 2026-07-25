# 12 — Implementation Plan

Directory structure, build order, and acceptance criteria. **Read `02_DATASETS.md`
first** — two data traps there invalidate results if missed.

---

## 1. Directory structure

Every file listed with its responsibility. Files marked ⭐ are on the critical
path; build those first.

```
new_ids/
├── README.md                          # index + non-negotiable rules
├── AGENT_GUIDE.md                     # ⭐ master brief for an implementing agent
├── requirements.txt
├── environment.yml
├── pyproject.toml                     # package metadata, ruff/pytest config
├── .gitignore                         # excludes data/, results/, *.pt, *.parquet
│
├── dataset/                           # ⭐ RAW DATA, READ-ONLY, already present
│   ├── NF-CICIDS2018-v3.csv           # 4.03 GB, 20,115,529 rows — PRIMARY
│   ├── NF-ToN-IoT-v3.csv              # 5.06 GB, 27,520,260 rows — SECONDARY
│   ├── NF-UNSW-NB15-v3.csv            # 0.55 GB,  2,365,424 rows — tertiary
│   ├── NF-BoT-IoT-v3.csv              # 3.64 GB, 16,933,808 rows — degenerate
│   └── NetFlow_v3_Features.csv        # feature dictionary
│
├── docs/                              # 00-14 + TODO (this document set)
│
├── config/
│   ├── default.yaml                   # ⭐ base config; full key list in §4.2
│   ├── dataset/
│   │   ├── cicids2018.yaml            # ⭐ PRIMARY (ip nodes, 14 classes, families)
│   │   ├── ton_iot.yaml               # SECONDARY (ip nodes, 9 classes)
│   │   ├── unsw_nb15.yaml             # TERTIARY  (ip_port nodes! only 40 IPs)
│   │   └── bot_iot.yaml               # DEGENERATE case only
│   ├── model/
│   │   ├── argus.yaml                 # ⭐ full model
│   │   ├── argus_softmax.yaml         # ablation: EPC removed        (isolates C1)
│   │   ├── argus_mean_agg.yaml        # ablation: robust agg removed (isolates C2)
│   │   ├── egraphsage.yaml
│   │   ├── egatv2.yaml
│   │   ├── anomal_e.yaml
│   │   └── tabular.yaml               # Extra Trees / RF / MLP / identity-only
│   └── experiment/
│       ├── closed_set.yaml            # P-CS
│       ├── open_set.yaml              # P-OS  (Protocol B)
│       ├── open_set_family.yaml       # P-OS2 (Protocol B2, CICIDS2018 only)
│       ├── few_shot.yaml              # P-FS
│       ├── transfer.yaml              # P-TR  (Protocol C)
│       ├── host_disjoint.yaml         # P-ID  (Protocol D)
│       ├── adversarial.yaml           # P-ADV
│       ├── temporal_ladder.yaml       # P-TEMP (8 rungs x 5 seeds)
│       ├── xai.yaml                   # P-XAI
│       ├── deployment.yaml            # P-DEP
│       └── timeout_stability.yaml     # P-TO
│
├── data/                              # DERIVED data, all generated
│   ├── interim/                       # cleaned, deduped, canonicalised parquet
│   ├── processed/
│   │   └── <dataset>/
│   │       ├── train/  val/  test/
│   │       ├── subsample_report.json
│   │       └── cache/                 # precomputed graph batches (§4.5)
│   └── artifacts/
│       └── <run_id>/
│           ├── feature_pipeline.joblib
│           ├── feature_manifest.json  # ⭐ ordered names + channel assignment
│           └── pipeline_provenance.json
│
├── src/argus/
│   ├── __init__.py
│   ├── cli.py                         # unified entry: argus <command> --config
│   ├── config.py                      # ⭐ load/merge/validate/resolve configs
│   ├── constants.py                   # ⭐ NF-v3 column lists, class vocabularies
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # streaming CSV -> parquet, chunked
│   │   ├── clean.py                   # ⭐ nulls, negatives, dedup
│   │   ├── canonical.py               # ⭐ label canonicalisation (02 §5.2)
│   │   ├── subsample.py               # ⭐ stratified temporal subsampling
│   │   ├── splits.py                  # ⭐ Protocol A / B / B2 / C / D
│   │   └── audit.py                   # ⭐ leakage + identity-leakage audits
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pipeline.py                # ⭐ FeaturePipeline fit/transform/save
│   │   ├── conditioning.py            # ⭐ TE1 signed-log + quantile
│   │   ├── derived.py                 # ⭐ TE2 13 rhythm descriptors
│   │   ├── encoders.py                # categorical, bitfield, port encoding
│   │   ├── spectral.py                # TE7 per-host FFT descriptor
│   │   └── partition.py               # ⭐ controllable vs observer channels
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── windows.py                 # ⭐ multi-scale ring buffers
│   │   ├── builder.py                 # ⭐ window -> graph; TRAP-1 guard
│   │   ├── sampler.py                 # ⭐ degree-capped recency-stratified
│   │   ├── node_features.py           # 12 local stats + TE7
│   │   └── batching.py                # anchor-bin batching, BPTT chunks
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── time_encoding.py           # TE3 Time2Vec (log-grid init!)
│   │   ├── aggregation.py              # ⭐ mean / trimmed / soft_medoid + multi-agg
│   │   ├── norm.py                     # ⭐ GraphNorm / LayerNorm / RMSNorm (no BatchNorm)
│   │   ├── attention.py                # TE4 time-decayed multi-head
│   │   ├── memory.py                   # TE6 per-node GRU + eviction + mem dropout
│   │   ├── srteg.py                    # ⭐ encoder
│   │   ├── multiscale.py               # TE5 gated fusion
│   │   ├── prototypes.py               # ⭐ multi-prototype bank, register(), EMA
│   │   ├── epc.py                      # ⭐ log-space evidence, 3-way decision
│   │   └── baselines/
│   │       ├── __init__.py
│   │       ├── egraphsage.py
│   │       ├── egatv2.py
│   │       ├── anomal_e.py
│   │       ├── tabular.py              # Extra Trees, RF, MLP
│   │       ├── identity_only.py        # ⭐ leakage floor (02 §7.1)
│   │       └── posthoc_osr.py         # OpenMax, energy, ODIN
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── am_softmax.py              # L_am + L_compact
│   │   ├── evidential.py              # L_evid + L_kl
│   │   ├── channel_penalty.py         # L_channel (C2)
│   │   └── unknown_aug.py             # ⭐ mixup + structural pseudo-unknowns
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── loop.py                    # ⭐ shared train/eval loop, AMP, BPTT
│   │   ├── stage1_encoder.py          # ⭐ geometry stage
│   │   ├── stage2_head.py             # ⭐ evidential stage
│   │   ├── thresholds.py              # theta_unknown/theta_defer selection
│   │   ├── checkpoint.py              # ⭐ save/resume incl. RNG + node memory
│   │   └── gates.py                   # G1-G6 sanity gates
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # macro-F1, per-class, PR-AUC
│   │   ├── openset.py                 # OpenAUC, openness, unknown TPR/FPR
│   │   ├── calibration.py             # ECE, MCE, Brier, reliability
│   │   ├── selective.py               # risk-coverage, AURC, deferral precision
│   │   ├── continual.py               # forgetting, BWT, few-shot
│   │   ├── deployment.py              # throughput, latency, memory
│   │   ├── bootstrap.py               # paired bootstrap CIs
│   │   └── report.py                  # assemble run -> json
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── constraints.py             # ⭐ domain feasibility projection
│   │   ├── a1_feature_pgd.py
│   │   ├── a2_structural_injection.py # ⭐ headline attack
│   │   ├── a3_prototype_poison.py
│   │   ├── a4_adaptive.py
│   │   └── a5_temporal_jitter.py
│   │
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── evidence_attrib.py         # ⭐ native decomposition + IG
│   │   ├── explainers.py              # GNNExplainer/PGExplainer/SHAP/counterfactual
│   │   ├── metrics.py                 # fidelity, sparsity, necessity, stability
│   │   └── triage.py                  # UNKNOWN clustering + report rendering
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   └── detector.py                # ⭐ StreamingDetector (push / register_class)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                    # seed all RNGs from one run seed
│       ├── logging.py                 # structured run logging
│       ├── io.py                      # parquet/json/tensor helpers
│       ├── registry.py                # results/runs/registry.jsonl
│       └── timing.py                  # latency instrumentation
│
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_prepare_data.py             # clean + subsample -> interim
│   ├── 02_build_splits.py             # ⭐ Protocol A/B/C + audit
│   ├── 03_cache_graphs.py             # ⭐ precompute graph batches
│   ├── 04_train_encoder.py            # Stage 1
│   ├── 05_train_head.py               # Stage 2 + threshold selection
│   ├── 06_eval_closed_set.py
│   ├── 07_eval_open_set.py
│   ├── 08_eval_few_shot.py
│   ├── 09_eval_transfer.py
│   ├── 10_run_temporal_ladder.py
│   ├── 11_run_adversarial.py
│   ├── 12_run_xai.py
│   ├── 13_measure_deployment.py
│   ├── 14_run_baselines.py
│   └── 15_make_tables.py              # ⭐ all paper tables/figures from runs
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb           # class counts, unique IPs, day structure
│   ├── 02_temporal_eda.ipynb          # IAT distributions, spectrograms
│   └── 03_results_figures.ipynb       # final figure polish
│
├── tests/
│   ├── conftest.py                    # tiny synthetic fixture dataset
│   ├── test_constants.py              # 53 columns present, partition complete
│   ├── test_clean.py
│   ├── test_splits.py                 # ⭐ Protocol A/B correctness, no leakage
│   ├── test_features.py               # ⭐ TE1/TE2 formulas, fit-on-train-only
│   ├── test_spectral.py               # TE7 on synthetic periodic signal
│   ├── test_graph.py                  # ⭐ no lookahead, K cap, strata bound
│   ├── test_aggregation.py            # ⭐ trimmed-mean breakdown point
│   ├── test_time_encoding.py          # periodicity recoverable
│   ├── test_epc.py                    # ⭐ registration = zero param change
│   ├── test_losses.py
│   ├── test_checkpoint.py             # ⭐ resume reproduces trajectory
│   ├── test_attacks.py                # constraint projection validity
│   ├── test_metrics.py                # OpenAUC/ECE against known values
│   └── test_streaming.py              # StreamingDetector contract
│
└── results/
    ├── runs/
    │   ├── registry.jsonl             # ⭐ one line per completed run
    │   └── <run_id>/
    │       ├── config_resolved.yaml
    │       ├── env.json               # git SHA, pip freeze, GPU
    │       ├── split_audit.json
    │       ├── ckpt_epoch<k>.pt
    │       ├── metrics.json
    │       └── logs/
    ├── tables/                        # T1-T12 as csv + latex
    └── figures/                       # F1-F11 as pdf + png
```

---

## 2. Environment

```
python >= 3.11
torch >= 2.4
torch-geometric >= 2.6          # or write message passing manually (see §2.1)
numpy, pandas, pyarrow
scikit-learn >= 1.5             # QuantileTransformer, ExtraTrees, metrics
scipy                           # rfft for TE7, stats
shap                            # KernelSHAP baseline
matplotlib, seaborn
pyyaml, omegaconf               # config
joblib                          # pipeline persistence
tqdm, rich                      # progress + logging
pytest, pytest-cov              # tests
ruff                            # lint
```

### 2.1 On torch-geometric

PyG is convenient but its scatter kernels are non-deterministic and its `MessagePassing`
abstraction makes the coordinate-wise trimmed mean awkward (it assumes a
scatter-reducible aggregation). **Recommendation: use PyG for data containers and
baselines, but implement `srteg.py` aggregation manually** with explicit
`[n_nodes, K, d_h]` dense neighbour tensors. `K = 32` is small enough that dense
is fast, and it makes trimming, soft-medoid, and attribution trivial. This also
removes a hard dependency risk on Kaggle.

---

## 3. Build order and acceptance criteria

Phases must complete in order except where marked parallel. **Do not begin a
phase until the previous phase's acceptance criteria pass.**

### P1 — Data substrate ⭐ BLOCKING

Files: `constants.py`, `data/*`, `features/*`, `scripts/00`–`02`.

Accept when:
- [ ] All four NF-v3 datasets download and load; column count is exactly 53 + 2 labels.
- [ ] `test_constants.py` passes: every column in `partition.py` is in exactly one channel.
- [ ] Cleaning report matches the counts in `02_DATASETS.md` §2.1 within 1%.
- [ ] Protocol A produces splits where every class appears in train, val, and test.
- [ ] Protocol B produces 5 distinct holdout sets, recorded and reproducible.
- [ ] `split_audit.json` passes all 5 checks in `02_DATASETS.md` §4.2.
- [ ] `feature_manifest.json` written; `F_e` reported and consistent across splits.
- [ ] TRAP-1 guard hard-fails when `node_granularity: ip` is set for `unsw_nb15`.

### P2 — Baselines *(parallel with P3)*

Files: `models/baselines/*`, `scripts/14`.

Accept when:
- [ ] Extra Trees, RF, MLP, E-GraphSAGE, EGATv2 all train on the Protocol-A split.
- [ ] E-GraphSAGE macro-F1 is in a plausible range for the dataset (sanity, not parity).
- [ ] All baselines consume the identical graph batches ARGUS uses.
- [ ] Results land in `results/runs/` with the same schema as ARGUS runs.

> Baselines early is deliberate: if E-GraphSAGE cannot be reproduced on our split,
> that is a data-pipeline bug, and it is far cheaper to find now than after P4.

### P3 — SR-TEG encoder ⭐

Files: `graph/*`, `models/{time_encoding,aggregation,attention,memory,srteg,multiscale}.py`.

Accept when:
- [ ] `test_graph.py` passes all 7 assertions.
- [ ] `test_aggregation.py` confirms trimmed mean is unmoved by `< βn` corrupted inputs.
- [ ] `test_time_encoding.py` confirms a 30 s periodic signal is separable from aperiodic.
- [ ] Closed-set macro-F1 ≥ E-GraphSAGE on the primary dataset (parity is enough).
- [ ] Sanity gates G1, G2 pass.
- [ ] Graph-batch cache is built and a second epoch is materially faster than the first.

### P4 — EPC head ⭐ HIGHEST RISK

Files: `models/{prototypes,epc}.py`, `losses/*`, `train/*`.

Accept when:
- [ ] `test_epc.py` confirms `register_class` changes **zero** model parameters and
      leaves logits on old classes **bit-identical**.
- [ ] Sanity gates G3, G4 pass.
- [ ] Open-set OpenAUC exceeds the softmax-threshold baseline on ≥4 of 5 holdouts.
- [ ] Few-shot new-class F1 at `n = 20` is materially above chance, with old-class
      macro-F1 delta exactly 0.000.
- [ ] ECE below the softmax baseline.

> **Decision point.** If Stage 2 will not converge after two serious attempts,
> switch to `head.type: distance_threshold` per `06_TRAINING.md` §1 and proceed.
> This preserves all of the few-shot claim and most of the open-set claim.
> Do not let this phase block the paper.

### P5 — Adversarial *(depends on P4)*

Files: `attacks/*`, `scripts/11`.

Accept when:
- [ ] `test_attacks.py` confirms every perturbed vector satisfies all domain constraints.
- [ ] TE2 features are **recomputed** after perturbation, never perturbed directly.
- [ ] A2 runs at all budgets for all aggregation configs; figure F6 renders.
- [ ] Empirical breakdown point `m*` is computed and compared to the analytic sketch.

### P6 — XAI *(depends on P4, parallel with P5)*

Files: `xai/*`, `scripts/12`.

Accept when:
- [ ] Native attribution sums to the log-evidence within numerical tolerance
      (it is an exact decomposition — verify it).
- [ ] Figure F7 renders; the injected-mass knee position is reported.
- [ ] UNKNOWN triage purity/NMI/ARI computed on the P-OS split.

### P7 — Temporal ladder *(depends on P3; may start before P4 completes)*

Files: `scripts/10`, `config/experiment/temporal_ladder.yaml`.

Accept when:
- [ ] All 8 rungs × 5 seeds complete, or partial results are clearly marked.
- [ ] Per-class ΔF1 heatmap (F10) renders.
- [ ] Multi-scale gate-by-class figure (F5) renders.
- [ ] The §3.3 hypotheses in `09_TEMPORAL_STUDY.md` are recorded as confirmed or refuted.

> The ladder runs on the **encoder only** (Stage-1 nearest-prototype
> classification), so it does not depend on the evidential head. This is why it
> can proceed in parallel with P4 — and it de-risks C3 against P4's failure.

### P8 — Deployment + writing

Files: `streaming/detector.py`, `scripts/13`, `15`.

Accept when:
- [ ] `test_streaming.py` confirms no lookahead and bounded memory.
- [ ] Throughput/latency/memory measured for ARGUS and E-GraphSAGE on the same host.
- [ ] `scripts/15_make_tables.py` regenerates every paper table from the registry.
- [ ] Every number in the draft traces to a `run_id`.

---

## 4. Configuration system

Everything tunable lives in config. **No magic numbers in code.** This section is
the contract: an implementer builds `config/default.yaml` to match it exactly,
and `src/argus/config.py` validates against it.

### 4.1 Layering

Configs merge in this order, later overriding earlier:

```
config/default.yaml                 base — every key, documented, with defaults
  → config/dataset/<name>.yaml      dataset-specific (see 02_DATASETS.md §8)
    → config/model/<name>.yaml      model/ablation variant
      → config/experiment/<name>.yaml   protocol, seeds, sweeps
        → CLI overrides             --set model.d_h=128 --set train.lr=1e-4
```

The fully merged, resolved config is written verbatim to
`results/runs/<run_id>/config_resolved.yaml`. Never reconstruct a run from a
fragment.

### 4.2 `config/default.yaml` — complete key list

```yaml
run:
  seed: 0
  device: cuda                    # cuda | cpu
  precision: bf16                 # bf16 | fp16 | fp32
  deterministic: true
  out_dir: results/runs
  registry: results/runs/registry.jsonl

paths:
  dataset_dir: dataset            # raw CSVs live here, read-only
  interim_dir: data/interim
  processed_dir: data/processed
  artifact_dir: data/artifacts
  cache_dir: data/processed/cache

data:
  dataset: cicids2018             # cicids2018 | ton_iot | unsw_nb15 | bot_iot
  protocol: A                     # A | B | B2 | C | D
  subsample_target: 4000000
  minority_threshold: 200000
  benign_floor_fraction: 0.50
  dedup: true
  drop_null_labels: true
  time_bins_for_subsample: 100
  holdout_size: 3                 # protocol B
  holdout_repeats: 5
  holdout_seed: 1234

features:
  te1_enabled: true               # heavy-tail conditioning
  te2_enabled: true               # derived rhythm descriptors
  te7_enabled: true               # spectral beaconing descriptor
  quantile_n: 1000
  quantile_subsample: 1000000
  clip_post_transform: 5.0
  eps: 1.0e-6
  protocol_topk: 8
  l7_proto_topk: 16
  dst_port_topk: 32
  spectral_nbins: 64
  spectral_min_flows: 8

graph:
  node_granularity: ip            # overridden per dataset; TRAP-1 guarded
  anchor_bin_seconds: 1
  window_short_seconds: 1
  window_mid_seconds: 30
  window_long_seconds: 300
  neighbour_cap: 32               # K
  sampling: recency_stratified    # recent | uniform | recency_stratified
  strata: 4                       # Q
  memory_evict_seconds: 600

model:
  name: argus
  d_h: 128
  d_A: 48                         # controllable channel width
  d_B: 80                         # observer channel width  (d_A + d_B = d_h)
  d_t: 16
  d_z: 64
  layers: 2
  heads: 4
  aggregation: trimmed            # mean | trimmed | soft_medoid
  trim_beta: 0.20
  soft_medoid_temp: 1.0
  multi_aggregator: true          # robust + spread + degree scalar
  time_encoding: time2vec         # time2vec | bochner
  time2vec_period_min: 0.1
  time2vec_period_max: 600.0
  norm_node: graphnorm            # graphnorm | layernorm | rmsnorm  (NEVER batchnorm)
  norm_mlp: layernorm
  prenorm: true
  te3_enabled: true               # Time2Vec dt encoding
  te4_enabled: true               # time-decayed attention
  te5_enabled: true               # multi-scale fusion
  te6_enabled: true               # per-node GRU memory

head:
  type: epc                       # epc | softmax | openmax | energy | distance_threshold
  sub_prototypes_benign: 4
  sub_prototypes_attack_large: 2  # classes with >= 10k samples
  sub_prototypes_attack_small: 1
  margin_m: 0.35
  tau_start: 1.0
  tau_final: 0.10
  tau_anneal_epochs: 6
  tau_min: 0.02
  log_evidence_clamp: 15.0
  am_softmax_margin: 0.35
  am_softmax_scale_start: 10.0
  am_softmax_scale_final: 30.0
  am_softmax_warmup_epochs: 5
  fp32_head: true                 # keep head out of autocast
  theta_unknown: null             # selected on validation
  theta_defer: null
  target_false_unknown_rate: 0.05
  target_defer_rate: 0.02

regularisation:
  dropout: 0.10
  droppath: 0.05
  dropedge: 0.10
  edge_feature_dropout_a: 0.05    # Channel A only
  edge_feature_dropout_b: 0.0
  memory_dropout: 0.10            # anti identity-memorisation
  weight_decay: 1.0e-4
  label_smoothing: 0.05

loss:
  lambda_compact: 0.10
  lambda_channel: 0.05
  channel_ratio_tolerance: 0.5    # rho
  channel_penalty_stride: 4
  lambda_div: 0.05
  lambda_unknown: 1.00
  lambda_kl_max: 0.10
  kl_anneal_epochs: 10
  synth_unknown_ratio: 0.20       # gamma
  mixup_mu_low: 0.35
  mixup_mu_high: 0.65
  cos_reject: 0.90
  structural_candidates: 64
  effective_number_nu: 0.999

train:
  stage1_epochs: 30
  stage1_lr: 3.0e-4
  stage1_patience: 5
  stage2_epochs: 10
  stage2_lr: 5.0e-5
  stage2_patience: 3
  stage3_joint_finetune: false
  batch_anchor_bins: 64
  n_per_class: 32
  min_classes_per_batch: 8
  bptt_chunk: 8
  warmup_epochs: 3
  grad_clip: 1.0
  min_count_for_prototype: 100
  checkpoint_every_epoch: true

gates:
  g0_capacity_check: true
  g0_subset_size: 1000
  g0_required_train_acc: 0.99
  g1_min_val_f1_epoch5: 0.5
  g2_max_proto_cosine: 0.8
  g3_max_known_vacuity: 0.7
  g4_min_unknown_vacuity: 0.5
  g5_max_channel_ratio: 0.8
  g7_max_train_val_gap: 0.10
  monitor_every_steps: 100

eval:
  metrics_seeds: [0, 1, 2, 3, 4]
  bootstrap_resamples: 10000
  ece_bins: 15
  ece_binning: equal_mass         # equal_mass | equal_width
  few_shot_n: [1, 5, 10, 20, 50]
  openness_holdout_sizes: [1, 2, 3, 4]

attack:
  a1_steps: 40
  a1_epsilons: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
  a2_budgets: [0, 1, 2, 4, 8, 16, 32, 64]
  a2_spread: all_strata           # single_stratum | all_strata
  a3_poison_rates: [0.001, 0.005, 0.01, 0.05, 0.1]
  a4_steps: 100
  a5_jitter_sigmas: [0.05, 0.1, 0.25, 0.5, 1.0]

xai:
  topk_edges: 10
  topk_features: 15
  ig_steps: 50
  gnnexplainer_epochs: 200
  pgexplainer_epochs: 30
  shap_nsamples: 200
  stability_n_perturb: 20
```

### 4.3 Validation rules — `src/argus/config.py`

The loader must enforce these and **raise, not warn**:

| Rule | Check |
|---|---|
| Unknown keys | Any key not in the schema → error. Catches typo'd overrides that would silently no-op. |
| Channel widths | `model.d_A + model.d_B == model.d_h` |
| Head divisibility | `model.d_h % model.heads == 0` |
| Window ordering | `window_short < window_mid < window_long` |
| Anchor bin | `anchor_bin_seconds <= window_short_seconds` |
| Trim validity | `0 <= trim_beta < 0.5` |
| Temperature | `tau_final >= tau_min` and `tau_start >= tau_final` |
| **BatchNorm ban** | `norm_node` and `norm_mlp` must not be `batchnorm` — see `05_ARCHITECTURE.md` §9.1 |
| **TRAP-1 guard** | if `node_granularity == ip` then `measured_unique_src_ip >= min_unique_src_ip` |
| Protocol B2 | only permitted when the dataset config defines `families` |
| Sub-prototypes | `sub_prototypes_* >= 1` |
| Precision | if `precision == fp16` then `head.fp32_head` must be `true` |

### 4.4 Quick-change recipes

Common experiment variations, as CLI overrides.

```bash
# Switch dataset
--config config/dataset/ton_iot.yaml

# Temporal ladder rung L3 (features only, no architectural temporal parts)
--set model.te3_enabled=false --set model.te4_enabled=false \
--set model.te5_enabled=false --set model.te6_enabled=false \
--set features.te7_enabled=false

# Ablate robust aggregation (C2 control)
--set model.aggregation=mean

# Ablate the open-world head (C1 control)
--set head.type=softmax

# Sweep the breakdown point
--set model.trim_beta=0.10   # then 0.20, 0.25, 0.40

# Fast smoke run on a laptop
--set data.subsample_target=200000 --set train.stage1_epochs=2 \
--set run.device=cpu --set graph.neighbour_cap=8

# Fallback if the evidential head will not converge (06_TRAINING.md §1)
--set head.type=distance_threshold
```

### 4.5 Cross-cutting implementation rules

**Determinism.** Seed python/numpy/torch/cuda from one run seed. Where a
non-deterministic kernel is unavoidable, record it in `env.json` rather than
silently accepting it.

**Graph batch cache.** Graph construction is CPU-bound and must not repeat per
epoch or per ablation rung. Cache to `data/processed/<dataset>/cache/<key>/`,
where `key` hashes: split hash, scale set, `K`, sampling strategy, seed, node
granularity. Temporal ladder rungs L4–L7 share one cache (only the model
changes); L0–L3 need separate caches because `F_e` differs.

**Run registry.** Every completed run appends one line to
`results/runs/registry.jsonl`. Experiment runners **skip runs already in the
registry**, making every script resumable across Kaggle sessions.

**Fail loudly.** Prefer a hard exception to a warning for: TRAP-1 violation,
split-audit failure, pipeline/split provenance mismatch, incomplete feature
partition, unrecognised attack label, NaN loss, BatchNorm requested. Silent
degradation is the expensive failure mode in research code.

---

## 5. Estimated effort by phase

Relative sizing only, for sequencing decisions.

| Phase | Relative size | Risk | Parallelisable |
|---|---|---|---|
| P1 Data | Large | Medium (trap handling) | No — blocking |
| P2 Baselines | Medium | Low | Yes, with P3 |
| P3 Encoder | Large | Medium | — |
| P4 EPC head | Medium | **High** | — |
| P5 Adversarial | Medium | Medium | Yes, with P6 |
| P6 XAI | Medium | Low | Yes, with P5 |
| P7 Temporal ladder | Small code, large compute | Low | Yes, from P3 |
| P8 Deployment + writing | Medium | Low | — |

Compute is dominated by P7 (40 runs) and P5 (attack sweeps × configs). Both are
embarrassingly parallel across Kaggle sessions given the run registry.

---

## 6. Descoping order

If time runs short, drop in this order. Never drop anything above the line.

```
DROP FIRST  →  NF-BoT-IoT degenerate case
               P-TO flow-timeout ablation
               A3 prototype poisoning
               A4 adaptive attacker
               GraphIDS baseline
               Cross-dataset reverse direction
               NF-UNSW-NB15 entirely
─────────────────────────────────────────────
NEVER DROP  →  Open-set holdout variance (5 holdouts, mean ± std)
               Calibration metrics (ECE + reliability)
               Temporal ladder rungs L0-L3
               A2 structural injection
               Few-shot zero-forgetting demonstration
               Re-run baselines on our splits
```
