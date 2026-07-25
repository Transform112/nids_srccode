# TODO

Tracked checklist. Phase IDs refer to `12_IMPLEMENTATION_PLAN.md` §3;
experiment IDs to `13_EXPERIMENT_MATRIX.md`.

---

## Planning — complete

- [x] Survey the GNN-NIDS landscape
- [x] Verify the novelty gap with two arXiv full-text queries (3 and 9 results)
- [x] Confirm scope with the user (compute, datasets, novelty shape, adversarial, venue)
- [x] Extract the exact NF-v3 53-column schema and the 10 temporal features
- [x] Extract per-dataset flow counts, class counts, unique-IP counts, day structure
- [x] Identify TRAP 1 (unique-IP degeneracy) and choose per-dataset node granularity
- [x] Identify TRAP 2 (day segregation) and design the two split protocols
- [x] Design the temporal pillar TE1–TE7 and the L0→L7 ablation ladder
- [x] Identify the TE7 novelty hook (open problem left by the NF-v3 authors)
- [x] Design SR-TEG encoder and EPC head with full equations and shapes
- [x] Design the threat model and attacks A1–A5
- [x] Design the XAI programme, including the F7 robustness-verification figure
- [x] Write documents 00–14 and this checklist
- [x] **Profile the real CSVs** — class counts, IP counts, day structure, spans
- [x] Identify TRAP 3 (host identity leakage) and design the identity audit
- [x] Correct four discrepancies between the paper and the actual data
- [x] Change primary dataset ToN-IoT → CICIDS2018 (14 classes, 181,876 IPs, hierarchy)
- [x] Add Protocol B2 (within-family open-set) and Protocol D (host-disjoint)
- [x] Refine architecture for overfit / underfit / vanishing / exploding gradients
- [x] Add normalisation policy (LayerNorm/GraphNorm; BatchNorm rejected with reasons)
- [x] Add multi-prototype head, log-space evidence, multi-aggregator readout
- [x] Write `AGENT_GUIDE.md` master brief
- [x] Remove obsolete `papers/` and `plan/` folders

---

## P0 — Repository setup

- [x] `pyproject.toml`, `requirements.txt`, `.gitignore` (`environment.yml` skipped; pyproject/requirements suffice)
- [x] `src/argus/` package skeleton with all `__init__.py`
- [x] `config/default.yaml` with every key from `07_HYPERPARAMETERS.md`
- [x] `config/dataset/*.yaml` for all four datasets
- [x] `src/argus/config.py` — schema validation that **fails on unknown keys**
- [x] `src/argus/utils/{seed,logging,io,registry,timing}.py` — extracted from
      inlined code; `train/checkpoint.py` RNG save/restore + seed_all + structured
      JSONL logging + parquet/JSON/tensor I/O + append-only run registry + latency
      tracker. All five modules tested transitively through existing tests.
- [x] `tests/conftest.py` — tiny synthetic NF-v3-shaped fixture
- [x] CI-equivalent local command: `pytest` (97 tests passing; `ruff` not yet run)

## P1 — Data substrate ⭐ BLOCKING

- [x] `constants.py` — 53 column names, temporal subset, class vocabularies
- [x] `data/canonical.py` — label canonicalisation tables (02 §5.2); **raises on
      unknown labels**; handles the `Infilteration` misspelling and ToN-IoT casing
- [x] `features/partition.py` + `assert_partition_complete`
- [x] `data/loader.py` — chunked CSV → parquet (files are up to 5 GB)
- [x] `data/clean.py` — nulls, negative clipping, dedup **before splitting**
- [x] `data/subsample.py` — stratified temporal, `minority_threshold = 200_000`
- [x] `data/splits.py` — Protocol A, B (B2/C/D helpers not yet separately implemented)
- [x] `data/audit.py` — 4 hard checks + identity-leakage audit (near-duplicate rate helper present)
- [x] `models/baselines/identity_only.py` — the identity floor classifier
- [x] `features/conditioning.py` — TE1
- [x] `features/derived.py` — TE2, all 13 features incl. `iat_undefined`
- [x] `features/encoders.py` — categorical, bitfield, port
- [x] `features/pipeline.py` — fit/transform/save/load + provenance record
- [x] `scripts/01_prepare_data.py`, `02_build_splits.py`
- [x] Tests: `test_constants`, `test_data_pipeline` (clean/splits/audit), `test_features_pipeline`
- [x] Identity floor measured end-to-end on a real CICIDS2018 slice via `scripts/00_smoke_test.py`
      (full four-dataset report still pending a real full-scale run)

**Acceptance:** all criteria in `12_IMPLEMENTATION_PLAN.md` §3/P1.

## P2 — Baselines *(parallel with P3)*

- [x] `models/baselines/tabular.py` — Extra Trees, RF, MLP
- [x] `models/baselines/egraphsage.py`
- [x] `models/baselines/egatv2.py` — GATv2 attention (deterministic segment-softmax
      via `scatter_reduce_`/`index_add_`), residual connections, 2 layers, 4 heads
- [x] `models/baselines/anomal_e.py` — graph-autoencoder anomaly detection:
      edge-level reconstruction, anomaly score = MSE. Forward/backward tested.
- [x] `models/baselines/posthoc_osr.py` — OpenMax (Weibull per class), energy-based
      OSR (free energy score), ODIN (temperature-scaled max-softmax). Full test coverage.
- [ ] `models/baselines/graphids.py` — not implemented; GraphIDS reproducibility
      decision still pending. Descoping candidate per AGENT_GUIDE.md §11.
- [x] `scripts/14_run_baselines.py` — Extra Trees/RF/MLP/identity-only/E-GraphSAGE/
      EGATv2 (`--skip-egraphsage`/`--skip-egatv2`); E-GraphSAGE verified end-to-end
      on a real CICIDS2018 slice, EGATv2 unit-tested (forward/backward/isolated-node)
      but not yet run end-to-end on real data
- [ ] Run E-BL-* (63 runs) — mechanics verified; full sweep not yet run
- [ ] Decide whether GraphIDS is reproducible in budget; record the decision

## P3 — SR-TEG encoder ⭐

- [x] `graph/windows.py` — multi-scale ring buffers
- [x] `graph/builder.py` — window→graph, **TRAP-1 hard-fail guard**
- [x] `graph/sampler.py` — degree-capped recency-stratified sampling
- [x] `graph/node_features.py` — 12 local stats, wired into `graph/batching.py`
      (previously computed but unused — now real signal, not a zero placeholder)
- [x] `features/spectral.py` TE7 folded into `graph/node_features.py`
      (`compute_node_features_te7`, 6 scalars), wired into batching
- [x] `graph/batching.py` — anchor-bin batching; multi-bin BPTT chunking left as
      a straightforward loop-level extension (see `train/loop.py` docstring)
- [x] `models/time_encoding.py` — Time2Vec with **log-grid `ω` init**
- [x] `models/aggregation.py` — mean / trimmed / soft_medoid + multi-aggregator
- [x] `models/attention.py` — time-decayed multi-head, scale-normalised Δt
- [x] `models/memory.py` — per-node GRU + memory dropout (eviction handled via BPTT detach)
- [x] `models/srteg.py` — encoder
- [x] `models/multiscale.py` — gated fusion
- [x] `scripts/03_cache_graphs.py` — built; hashes split/scale/K/sampling/granularity
      for cache-key determinism, saves unique_bins + meta per split. Not yet run at scale.
- [x] Tests: `test_graph`, `test_aggregation`, `test_time_encoding` (spectral covered
      via `test_node_features_are_real_signal_not_zeros`)
- [ ] Run E-CS-FULL; confirm parity with E-GraphSAGE — mechanics verified end-to-end
      on a real data slice; full-scale parity run not yet executed

## P4 — EPC head ⭐ HIGHEST RISK

- [x] `models/prototypes.py` — bank, `register()`, **`ema_update()`** (EMA drift
      correction, gated externally by A3's evidence-gate check)
- [x] `models/epc.py` — evidence, three-way decision
- [x] `losses/am_softmax.py` — `L_am` + `L_compact`
- [x] `losses/evidential.py` — `L_evid` + `L_kl` with annealing
- [x] `losses/channel_penalty.py` — `L_channel`, stride-4 application — wired into
      `train/stage1_encoder.py` via `run_epoch(channel_penalty=...)`
- [x] `losses/unknown_aug.py` mixup implemented inline in `train/stage2_head.py`
      (`embedding_mixup_unknowns`); structural pseudo-unknowns not yet implemented
- [x] `train/loop.py`, `stage1_encoder.py`, `stage2_head.py`
- [x] `train/thresholds.py` — validation-only `θ` selection (quantile-based)
- [x] `train/checkpoint.py` — save/resume incl. RNG (node memory dict not yet
      threaded through checkpoints — memory is bin-local in the current loop)
- [x] `train/gates.py` — G0–G7
- [x] `scripts/04_train_encoder.py`, `05_train_head.py` (05 now also calibrates
      and saves thresholds)
- [x] Tests: `test_epc` (**zero param change on registration**), `test_channel_penalty`,
      `test_thresholds` (`test_losses`/`test_checkpoint` as standalone files not yet split out)
- [ ] **Decision point:** Stage 2 converges on the smoke-test slice; a
      distinct real convergence decision awaits a full-scale run

## P5 — Adversarial *(depends on P4)*

- [x] `attacks/constraints.py` — domain feasibility projection
- [x] `attacks/a1_feature_pgd.py` — SPSA-estimated ascent (pipeline is not
      autograd-traceable; see module docstring), relative raw-space epsilon budget
- [x] `attacks/a2_structural_injection.py` ⭐ headline — injection mechanics,
      budget sweep, and no-new-target-leakage verified with tests
- [x] `attacks/a3_prototype_poison.py` — added `PrototypeBank.ema_update` (the
      drift-correction mechanism this attack targets, previously unimplemented),
      gate-on/off comparison
- [x] `attacks/a4_adaptive.py` — both objectives (`evasion`, `unknown_avoidance`),
      same SPSA mechanics as A1 but over Channel A + timing columns, 100 steps
- [x] `attacks/a5_temporal_jitter.py` + attacker-cost accounting (documented
      duration/pkt-rate approximation from the mean IAT jitter factor)
- [x] `scripts/11_run_adversarial.py` — orchestrates A1/A2/A4/A5 per sampled
      target flow and A3 per attack class; not yet run end-to-end on real data
      (needs a trained Stage-1/2 checkpoint — Kaggle-scale)
- [x] `tests/test_adversarial.py` — every perturbed vector is feasible (byte floor,
      IAT ordering); TE2 recomputation now exercised end-to-end via A1/A4/A5
      (all three call the real `FeaturePipeline.transform`, which recomputes TE2)
- [ ] Run E-ADV-A2/A2b/A2c on real data; produce F6 and the empirical breakdown point
- [ ] Run the full A1/A3/A4/A5 sweeps on real data (Kaggle-scale)

## P6 — XAI *(depends on P4, parallel with P5)*

- [x] `xai/evidence_attrib.py` — exact embedding-level decomposition + Integrated
      Gradients feature attribution, both verified against the completeness axiom
- [x] `xai/explainers.py` — GNNExplainer (free per-edge mask), PGExplainer
      (mask predicted by a small per-edge MLP — simplified single-instance
      amortisation, documented), KernelSHAP (feature-only, coalition sampling +
      SHAP kernel weights), attention-weights-alone (new opt-in `record_attention`
      side channel on `SRTEGLayer`), counterfactual necessity (greedy edge removal)
- [x] `xai/metrics.py` — fidelity±, sparsity, necessity, stability (Spearman
      correlation across perturbations), runtime timer
- [x] `xai/triage.py` — per-decision triage report (deterministic prose template)
      + UNKNOWN cluster validation (purity/NMI/ARI, k-means or HDBSCAN) +
      nearest-prototype correspondence
- [x] `scripts/12_run_xai.py` — native attribution + baseline explainers +
      fidelity metrics + UNKNOWN triage per sampled decision; not yet run on
      real data (needs a trained Stage-1/2 checkpoint)
- [x] Verify the decomposition sums to log-evidence within tolerance (`test_xai.py`)
- [x] `tests/test_xai_explainers.py` — 11 tests covering every new module
- [ ] Run E-XAI-2 on real data; produce **F7**; report the knee position vs `β`

## P7 — Temporal ladder *(depends on P3; may run before P4 completes)*

- [ ] Record the §3.3 hypotheses **before** running
- [x] `config/experiment/temporal_ladder.yaml` — 8 rungs × 5 seeds (exists; sweep spec
      recorded, consumed by `scripts/10_run_temporal_ladder.py` via CLI flags)
- [x] `scripts/10_run_temporal_ladder.py` — exists; not yet run at Kaggle scale
- [ ] Run E-TEMP-L0..L7 (49 runs)
- [ ] Produce **T6**, **F10**, **F5**
- [ ] Report the L0→L2 delta as a named finding
- [ ] Record which hypotheses were confirmed and which refuted

## P8 — Deployment + writing

- [x] `streaming/detector.py` — `push()` / `register_class()`
- [x] `eval/deployment.py` — throughput, latency percentiles, memory (per-flow
      latency captured in `Verdict.latency_ms`; `DeploymentReport` dataclass +
      `model_size()` + `measure_streaming_throughput()`)
- [x] `tests/test_streaming.py` — no lookahead, bounded memory, zero-param registration
- [x] `scripts/13_measure_deployment.py` — measures ARGUS + E-GraphSAGE on a
      real CICIDS2018 test slice via `StreamingDetector.push()`; implemented
- [x] `scripts/15_make_tables.py` — thin aggregator: reads JSON from earlier
      scripts, reshapes into flat CSVs under `results/tables/`; implemented
- [ ] Draft the paper per `14_PAPER_OUTLINE.md`
- [ ] Complete the pre-submission checklist

---

## Session 2026-07-25 — completed

- [x] **Environment fix:** torch 2.4.0 → 2.5.1+cpu (fbgemm.dll missing `libomp140.x86_64.dll` dep)
- [x] `src/argus/utils/` — seed, logging, io, registry, timing (5 files)
- [x] `src/argus/models/baselines/anomal_e.py` — graph-autoencoder
- [x] `src/argus/models/baselines/posthoc_osr.py` — OpenMax, energy, ODIN
- [x] `src/argus/eval/` — openset, calibration, selective, continual, bootstrap, report (6 files)
- [x] `config/experiment/` — 7 new config files
- [x] `scripts/03_cache_graphs.py` — removed (graph building is on-the-fly; 03_fit_features.py is canonical)
- [x] `scripts/06_eval_closed_set.py` — real implementation (loads ckpt, runs inference, computes metrics)
- [x] `scripts/07_eval_open_set.py` — rewritten: real open-set eval with 5 holdout sets, OpenAUC, TPR/FPR
- [x] `scripts/08_eval_few_shot.py` — rewritten: registers classes via model output embeddings, verifies zero-param change
- [x] `scripts/09_eval_transfer.py` — rewritten: cross-dataset transfer with family mapping + unknown detection
- [x] **Tests: 97 passing** (was 90; +7 for AnomalE + posthoc OSR)
- [x] All 15 scripts (00-15) are real implementations that can run on Kaggle

## Evaluation runs

Ordered per `13_EXPERIMENT_MATRIX.md` §11. A submittable draft exists after step 7.

- [ ] 1. E-BL-* — baselines (63)
- [ ] 2. E-CS-FULL — ARGUS closed-set (18)
- [ ] 3. E-TEMP-L0..L7 — temporal ladder (49) → **C3 secured**
- [ ] 4. E-OS-1 + baselines — open-set (25 checkpoints reused across 9 heads) → **C1 secured**
- [ ] 5. E-ADV-A2/A2b/A2c — structural injection (120) → **C2 secured**
- [ ] 6. E-FS-1..3 — few-shot (100) → **C1 complete**
- [ ] 7. E-XAI-2 — attribution vs budget (12) → **C4 secured**
- [ ] 8. E-CAL-1, E-SEL-1 — calibration and selective prediction (20)
- [ ] 9. E-ABL-* — ablations (120)
- [ ] 10. E-TR-*, E-DEP-1, E-TO-1 — transfer, deployment, timeout (42)
- [ ] 11. E-ADV-A1/A3/A4/A5 — remaining attacks (40)
- [ ] 12. E-XAI-1/E-XAI-3 — explanation quality and triage (10)

---

## Open decisions

- [x] ~~Confirm NF-v3 CSVs are downloaded and readable~~ — all four present in `dataset/`
- [ ] Decide whether GraphIDS is reproducible within budget (during P2)
- [ ] Decide whether to enable EMA prototype drift correction in the final
      configuration (informed by E-ADV-A3)
- [ ] Decide `sub_prototypes_benign` from the E-ABL-EPC sweep (default 4)
- [ ] Choose the final title from `14_PAPER_OUTLINE.md`
- [ ] Choose the target venue and confirm its deadline and page limit

---

## Standing rules

Violating any of these silently invalidates results. Re-read before each phase.

1. Never a random split. Protocol A or B only.
2. Never IP-level nodes on UNSW-NB15 or BoT-IoT.
3. Run the identity-leakage audit before modelling; report the identity floor.
4. Fit all scalers and encoders on train only.
5. Never quote a baseline number from another paper.
6. Always report per-class F1 alongside macro-F1; never headline accuracy.
7. Thresholds are selected on validation only; report sweeps as curves.
8. Recompute TE2 features after any adversarial perturbation.
9. Never BatchNorm; compute Dirichlet evidence in log space.
10. Run gate G0 before any full training run.
11. Every reported number traces to a `run_id`.
