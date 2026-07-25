# 07 — Hyperparameters

Single reference table. Defaults are what `config/default.yaml` ships. "Tuned by"
names the experiment in `13_EXPERIMENT_MATRIX.md` that selects the value.

**Tuning rule: all selection happens on the validation split. The test split is
touched once, at the end, per configuration.** Threshold sweeps reported as
curves are exempt (a curve is not a selection).

---

## 1. Data and splits

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `split.train/val/test` | 0.70 / 0.15 / 0.15 | fixed | — | Per-class stratified temporal, `02_DATASETS.md` §4.1 |
| `split.protocol_b.holdout_size` | 3 | {1, 2, 3, 4} | E-OS-2 | Number of classes held out as unknown |
| `split.protocol_b.repeats` | 5 | ≥5 | fixed | Distinct random holdout sets; report mean ± std |
| `subsample_target` | 4,000,000 | {1M, 2M, 4M} | E-SCALE | Per dataset; see `02_DATASETS.md` §5.2 |
| `minority_threshold` | 50,000 | fixed | — | Classes below this are never subsampled |
| `min_unique_src_ip` | 1,000 | fixed | — | TRAP-1 hard-fail guard |

## 2. Feature engineering

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `quantile.n_quantiles` | 1000 | fixed | — | TE1 |
| `quantile.subsample` | 1,000,000 | fixed | — | Fit cost control |
| `clip_post_transform` | ±5.0 | {3, 5, 10} | — | Low sensitivity |
| `eps` | 1e-6 | fixed | — | All TE2 denominators |
| `protocol_topk` | 8 | fixed | — | + OTHER → dim 9 |
| `l7_proto_topk` | 16 | fixed | — | + OTHER → dim 17 |
| `dst_port_topk` | 32 | {16, 32, 64} | — | + OTHER → dim 33 |
| `channel_ratio_tolerance ρ` | 0.5 | {0.3, 0.5, 0.7} | E-ABL-CH | Provenance penalty knee |

## 3. Graph construction

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `anchor_bin_seconds` | 1 | {1, 5} | — | Prediction granularity |
| `window.short_seconds` | 1 | {0.5, 1, 2} | E-ABL-W | Burst scale |
| `window.mid_seconds` | 30 | {10, 30, 60} | E-ABL-W | Session scale |
| `window.long_seconds` | 300 | {120, 300, 600} | E-ABL-W | Low-and-slow scale |
| `neighbour_cap K` | 32 | **{8, 16, 32, 64}** | **E-ABL-K** | Robustness/accuracy knob — key ablation |
| `sampling` | `recency_stratified` | {recent, uniform, recency_stratified} | E-ABL-SAMP | See `04_GRAPH_CONSTRUCTION.md` §4 |
| `strata Q` | 4 | {2, 4, 8} | E-ABL-SAMP | Bounds per-interval share at 1/Q |
| `node_memory_evict_seconds` | 600 | fixed | — | 2 × long window |
| `spectral.nbins` | 64 | {32, 64, 128} | E-ABL-T7 | TE7 FFT resolution |
| `spectral.min_flows` | 8 | fixed | — | Below this, emit zero vector |

## 4. Model

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `d_h` | **128** | {64, 128, 256} | E-ABL-DIM | Raised from 64: `F_e = 147` into `2×32` was an underfit bottleneck |
| `d_A` | 48 | {32, 48, 64} | E-ABL-CH | Controllable channel width; deliberately `< d_B` |
| `d_B` | 80 | {64, 80, 96} | E-ABL-CH | Observer channel width; `d_A + d_B == d_h` |
| `d_t` | 16 | {8, 16, 32} | — | Time2Vec dim |
| `d_z` | 64 | {32, 64, 128} | E-ABL-DIM | Prototype space dim |
| `layers L` | 2 | {1, 2, 3} | E-ABL-DEPTH | 3 rarely helps; 2-hop is the useful radius |
| `heads H` | 4 | {2, 4, 8} | — | Must divide `d_h` |
| `aggregation` | `trimmed` | **{mean, trimmed, soft_medoid}** | **E-ABL-AGG** | Core C2 ablation |
| `trim_beta β` | **0.20** | **{0.0, 0.1, 0.2, 0.25, 0.4}** | **E-ABL-AGG** | Lowered from 0.25: that discarded half of all messages |
| `multi_aggregator` | true | {true, false} | E-ABL-AGGX | Robust + trimmed-spread + degree scalar; recovers expressiveness |
| `soft_medoid_temp T` | 1.0 | {0.1, 1, 10} | E-ABL-AGG | Floor at 0.05 for numerical stability |
| `norm_node` | `graphnorm` | {graphnorm, layernorm, rmsnorm} | — | **`batchnorm` is rejected by the validator** |
| `norm_mlp` | `layernorm` | {layernorm, rmsnorm} | — | Pre-norm placement |
| `prenorm` | true | {true, false} | — | Post-norm attenuates the identity path → vanishing gradients |
| `time_encoding` | `time2vec` | {time2vec, bochner} | — | |
| `time2vec_period_min` | 0.1 s | fixed | — | Log-grid init lower bound |
| `time2vec_period_max` | 600 s | fixed | — | Log-grid init upper bound |
| `decay.half_lives` | `[D_s/16, D_s/8, D_s/4, D_s/2]` | fixed | — | Per-head, per-scale init |

## 5. EPC head

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `sub_prototypes_benign` | **4** | {1, 2, 4, 8} | E-ABL-EPC | Benign is multi-modal; 1 is a known failure mode |
| `sub_prototypes_attack_large` | 2 | {1, 2, 4} | E-ABL-EPC | Classes ≥ 10k samples |
| `sub_prototypes_attack_small` | 1 | fixed | — | Classes < 10k samples |
| `margin_m` (evidence) | 0.35 | {0.2, 0.35, 0.5} | E-ABL-EPC | Distance offset in `e_c = exp(−(d_c−m)/τ)` |
| `tau_start` | **1.0** | {0.5, 1.0, 2.0} | — | **Annealing start — do not begin at `tau_final`** |
| `tau_final` | 0.10 | **{0.05, 0.1, 0.2, 0.5}** | **E-ABL-EPC** | Strongest single lever on calibration |
| `tau_anneal_epochs` | 6 | {3, 6, 10} | — | Geometric schedule |
| `tau_min` | 0.02 | fixed | — | `τ = softplus(τ̂) + tau_min`; prevents infinite gradient multiplier |
| `log_evidence_clamp` | 15.0 | fixed | — | **Mandatory overflow guard** |
| `fp32_head` | true | fixed | — | Head stays outside autocast |
| `am_softmax_margin` | 0.35 | {0.2, 0.35, 0.5} | E-ABL-EPC | Stage-1 angular margin |
| `am_softmax_scale_start` | 10.0 | fixed | — | Warmup start |
| `am_softmax_scale_final` | 30.0 | {10, 30, 64} | — | Standard for hyperspherical losses |
| `am_softmax_warmup_epochs` | 5 | fixed | — | `s = 30` from random init is a gradient spike |
| `theta_unknown` | *selected* | swept | E-OS-1 | Set for 5% false-UNKNOWN on val; also swept for OpenAUC |
| `theta_defer` | *selected* | swept | E-OS-1 | Set for 2% deferral on val; also swept for risk–coverage |
| `target_false_unknown_rate` | 0.05 | {0.01, 0.05, 0.10} | — | Operating point only |
| `target_defer_rate` | 0.02 | {0.01, 0.02, 0.05} | — | Operating point only |
| `cos_reject` (mixup) | 0.90 | {0.85, 0.9, 0.95} | — | Reject synthetic unknowns landing inside a real cone |
| `prototype_ema_momentum` | 0.99 | {0.9, 0.99, off} | E-DRIFT | Optional streaming drift correction |

## 5b. Regularisation

| Parameter | Default | Range | Purpose |
|---|---|---|---|
| `dropout` | 0.10 | {0.0, 0.1, 0.2} | Standard, MLP hidden layers |
| `droppath` | 0.05 | {0.0, 0.05, 0.1} | Stochastic depth on GNN residual branches |
| `dropedge` | 0.10 | {0.0, 0.1, 0.2} | Graph regularisation **and** implicit A2 adversarial training |
| `edge_feature_dropout_a` | 0.05 | {0.0, 0.05, 0.1} | **Channel A only** — regulariser and C2 mechanism |
| `edge_feature_dropout_b` | 0.0 | fixed | Observer channel is not dropped |
| `memory_dropout` | 0.10 | **{0.0, 0.1, 0.3}** | **Anti host-identity memorisation — see `02_DATASETS.md` §7** |
| `weight_decay` | 1e-4 | {0, 1e-4, 1e-3} | Excludes biases and norm parameters |
| `label_smoothing` | 0.05 | {0.0, 0.05, 0.1} | Prevents over-confident prototypes |
| `lambda_div` | 0.05 | {0.0, 0.05, 0.2} | Sub-prototype diversity; prevents collapse to single-prototype |

## 6. Losses

| Parameter | Default | Range | Tuned by | Notes |
|---|---|---|---|---|
| `λ_cmp` (compactness) | 0.10 | {0.0, 0.1, 0.5} | E-ABL-EPC | Stage 1 |
| `λ_ch` (channel penalty) | 0.05 | **{0.0, 0.02, 0.05, 0.2}** | **E-ABL-CH** | Core C2 ablation; 0.0 disables the partition |
| `λ_unk` (synthetic unknown) | 1.00 | {0.5, 1.0, 2.0} | E-ABL-EPC | Stage 2 |
| `λ_kl` (Dirichlet KL) | 0.10 max | {0.05, 0.1, 0.5} | E-ABL-EPC | Annealed over 10 epochs |
| `kl_anneal_epochs` | 10 | fixed | — | Without annealing, evidence collapses |
| `γ` (synthetic ratio) | 0.20 | {0.1, 0.2, 0.4} | E-ABL-EPC | Fraction of batch that is synthetic |
| `mixup_mu_range` | (0.35, 0.65) | fixed | — | Mid-band only |
| `structural_candidates` | 64 | fixed | — | Host search cap |
| `channel_penalty_stride` | 4 | fixed | — | Apply every 4th batch, scale `λ_ch` by 4 |

## 7. Optimisation

| Parameter | Stage 1 | Stage 2 | Range | Notes |
|---|---|---|---|---|
| Optimiser | AdamW | AdamW | fixed | |
| LR | 3e-4 | 5e-5 | {1e-4, 3e-4, 1e-3} | |
| Weight decay | 1e-4 | 1e-4 | {0, 1e-4, 1e-3} | |
| Warmup epochs | 3 | 0 | fixed | Linear |
| Scheduler | cosine | constant | fixed | |
| Grad clip | 1.0 | 1.0 | fixed | Global norm |
| Batch (anchor bins) | 64 | 64 | {16, 32, 64} | GPU-bound |
| `n_per_class` (loss targets) | 32 | 32 | {16, 32, 64} | Class-balanced target subsampling |
| `T_bptt` | 8 | 8 | {4, 8, 16} | Truncated BPTT chunk |
| Epochs | 30 | 10 | — | Early stop |
| Patience | 5 | 3 | — | |
| `ν` (effective-number) | 0.999 | 0.999 | {0.99, 0.999, 0.9999} | Class weighting |
| Precision | bf16 AMP | bf16 AMP | — | fp16 fallback |
| Seeds | 0,1,2,3,4 | inherit | fixed | All headline numbers are 5-seed mean ± std |

## 8. Adversarial

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `A1.steps` | 40 | {20, 40, 100} | PGD iterations |
| `A1.epsilon` | swept | {0.01 … 0.5} | In normalised feature space, **Channel A only** |
| `A1.step_size` | `2.5·ε/steps` | fixed | Standard PGD ratio |
| `A2.injection_budget m` | swept | {0, 1, 2, 4, 8, 16, 32, 64} | Flows injected per victim window |
| `A2.injection_spread` | `all_strata` | {single_stratum, all_strata} | `single_stratum` is the cheap attack; `all_strata` the strong one |
| `A3.poison_rate` | swept | {0.001 … 0.1} | Fraction of EMA update stream poisoned |
| `A4.steps` | 100 | fixed | Adaptive attack, more steps than A1 |
| `A5.jitter_sigma` | swept | {0.05 … 1.0} | Multiplicative log-normal on IAT columns |

## 9. XAI

| Parameter | Default | Notes |
|---|---|---|
| `explain.topk_edges` | 10 | Edges retained in an explanation |
| `explain.topk_features` | 15 | Features retained |
| `gnnexplainer.epochs` | 200 | Baseline explainer budget |
| `pgexplainer.epochs` | 30 | |
| `shap.nsamples` | 200 | KernelSHAP budget |
| `stability.n_perturb` | 20 | Perturbations per sample for stability metric |
| `triage.n_clusters` | `n_holdout_classes` | For UNKNOWN purity evaluation |

## 10. Sensitivity expectations

Recorded so time is not wasted tuning insensitive knobs.

**High sensitivity — tune carefully**
`tau_final`, `tau_anneal_epochs`, `trim_beta`, `K`, `λ_ch`, `sampling`,
`window.long_seconds`, `sub_prototypes_benign`, `memory_dropout`.

**Medium sensitivity**
`d_h`, `d_z`, `d_A`/`d_B` ratio, `margin_m`, `λ_unk`, `γ`, `layers`, `dropedge`,
`lambda_div`.

**Low sensitivity — take the default and move on**
`d_t`, `heads`, `dropout`, `am_softmax_scale_final`, `clip_post_transform`,
`protocol_topk`, `l7_proto_topk`, `n_quantiles`, `droppath`.

**Do not tune on test.** If a knob appears to need test-set tuning, that is
evidence the validation split is too small or mis-constructed — fix the split.

## 11. Values that are guards, not knobs

These exist to prevent a specific failure and must not be "tuned away" when they
appear to cost performance.

| Parameter | Guards against | Doc |
|---|---|---|
| `log_evidence_clamp = 15` | Overflow / exploding gradient in the head | `05` §6.3 |
| `tau_min = 0.02` | Infinite gradient multiplier as `τ → 0` | `05` §6.3 |
| `fp32_head = true` | fp16 overflow in `exp` | `05` §6.3 |
| `tau_start = 1.0` | Gradient bomb at Stage-2 epoch 0 | `06` §2.3 |
| `min_unique_src_ip = 1000` | TRAP 1 — degenerate graph | `02` §3 |
| `memory_dropout = 0.1` | TRAP 3 — host identity memorisation | `02` §7 |
| `min_count_for_prototype = 100` | Memorising a 440-sample class | `02` §6.3 |
| `norm_* ≠ batchnorm` | Five failure modes incl. an attack surface | `05` §9.1 |
| `g0_required_train_acc = 0.99` | Shipping a silently broken pipeline | `06` §8.1 |
