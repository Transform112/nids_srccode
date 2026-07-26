# 06 — Training

Module map: `src/argus/train/` — `stage1_encoder.py`, `stage2_head.py`,
`loop.py`, `checkpoint.py`. Losses in `src/argus/losses/`.

---

## 1. Two-stage schedule (risk mitigation)

Joint training of a prototype geometry *and* a Dirichlet evidence head from
random initialisation is unstable: early in training all embeddings are near each
other, all distances are similar, all evidence is similar, and the evidential
loss produces a near-uniform gradient that collapses prototypes together.

**Therefore train in two stages.** This is a deliberate design decision, not an
optimisation detail.

### Stage 1 — encoder + prototype geometry

Objective: learn an embedding space where classes occupy tight, well-separated
cones on the unit hypersphere.

- Trainable: SR-TEG encoder, embedding projection `W_z`, prototype bank `P`.
- Loss: `L_stage1 = L_am + λ_ch · L_channel + λ_cmp · L_compact`
- No evidential loss, no synthetic unknowns, no thresholds.
- Duration: 30 epochs (early-stop on validation macro-F1 computed with a
  nearest-prototype classifier, patience 5).

### Stage 2 — evidential calibration

Objective: turn distances into calibrated evidence and carve out low-evidence
regions for unknowns.

- **Encoder frozen.** Trainable: `W_z` (low LR), `P` (low LR), and the evidential
  parameters `m`, `τ`.
- Loss: `L_stage2 = L_evid + λ_unk · L_unknown + λ_kl · L_kl`
- Synthetic unknowns active (§3).
- Duration: 10 epochs, early-stop on validation OpenAUC, patience 3.

### Stage 3 (optional) — joint fine-tune

Unfreeze everything at LR/10 for 5 epochs. Run only if Stage 2 converged
cleanly. Skip if time-constrained; report which was used.

**Fallback path.** If Stage 2 will not converge (evidence collapses to uniform,
or OpenAUC does not exceed the softmax baseline), fall back to
`head.type: distance_threshold` — plain nearest-prototype classification with a
distance threshold for UNKNOWN. This preserves **all** of C1's few-shot
registration claim and most of the open-set claim, losing only the calibration
story. Decide by end of phase P4; do not let this block the paper.

---

## 2. Loss functions

### 2.1 `L_am` — additive-margin cosine loss (Stage 1)

Drives compactness and separation on the hypersphere.

```
logit_k = s · ( cos_k − m · 1[k = y] )        s = 30.0 , m = 0.35
L_am    = CrossEntropy( logit , y )
```

`s` (scale) and `m` (margin) as in `05_ARCHITECTURE.md` §6.3. Margin is applied
only to the true class, which forces the true-class cosine to exceed the others
by at least `m`.

### 2.2 `L_compact` — explicit compactness (Stage 1)

```
L_compact = mean_e ( 1 − ⟨ z_e , p_{y_e} ⟩ )
```

Small weight (`λ_cmp = 0.1`). `L_am` alone leaves classes as wide cones; tight
cones are what make the *distance-to-evidence* mapping meaningful in Stage 2.

### 2.3 `L_evid` — evidential loss (Stage 2)

Type-II maximum likelihood (Bayes risk of the cross-entropy under the Dirichlet):

```
L_evid = Σ_k y_k ( ψ(S) − ψ(α_k) )
```

where `ψ` is the digamma function, `α_k = e_k + 1`, `S = Σ α_k`. Numerically
stable and standard (Sensoy et al., NeurIPS 2018). Prefer this over the
sum-of-squares form.

**Temperature annealing is mandatory.** `τ` controls the sharpness of the
distance→evidence map and multiplies every head gradient by `1/τ`. Starting at
the target `τ = 0.1` is a gradient bomb: early in Stage 2 all distances are
similar, so all evidence is similar, and the `1/τ = 10` amplification turns that
uninformative uniform signal into a large gradient that destroys the Stage-1
geometry.

```
τ(epoch) = τ_start · (τ_final / τ_start) ^ (epoch / anneal_epochs)
τ_start = 1.0 ,  τ_final = 0.10 ,  anneal_epochs = 6      # geometric schedule
τ is stored as τ = softplus(τ̂) + 0.02 so it can never reach 0
```

Similarly, **warm up the AM-Softmax scale** `s` in Stage 1 from 10 → 30 linearly
over the first 5 epochs. `s = 30` from a random initialisation produces logits of
magnitude 30 and a correspondingly large initial gradient.

The evidential head runs in **fp32 even under AMP**
(`autocast(enabled=False)`), for the range and precision reasons in
`05_ARCHITECTURE.md` §6.3.

### 2.4 `L_channel` — provenance channel penalty (Stage 1)

This is the loss term that operationalises C2's provenance partition.

```
g_A = || ∂L_am / ∂x_e^A ||_2        (per-sample, controllable channel)
g_B = || ∂L_am / ∂x_e^B ||_2        (per-sample, observer channel)
r   = g_A / (g_A + g_B + eps)       ∈ [0,1]
L_channel = mean_e ( ReLU( r − ρ )² )
```

`ρ = 0.5` is the tolerated reliance ratio: no penalty while the controllable
channel accounts for at most half the input gradient; quadratic penalty beyond.
Weight `λ_ch = 0.05`.

Implementation: compute with `torch.autograd.grad(..., create_graph=True)` on the
detached input leaves for each channel. This roughly doubles backward cost — to
control it, **apply the penalty on every 4th batch only** and scale `λ_ch` by 4.
Verify equivalence in a small-scale test.

Do not set `ρ` to 0. The controllable features carry real signal (a SYN flood
genuinely has distinctive flags); the goal is to prevent *sole* reliance, not to
forbid use.

### 2.5 `L_unknown` — synthetic-unknown loss (Stage 2)

For synthetic-unknown samples (§3), the target is **zero evidence for every
class**:

```
L_unknown = mean_{e ∈ synth} ( Σ_k e_k )          # drive total evidence → 0
```

Weight `λ_unk = 1.0`. Optionally add a hinge to avoid over-suppression:
`ReLU(Σ_k e_k − ε_floor)` with `ε_floor = 0.01`.

### 2.6 `L_kl` — Dirichlet KL regulariser (Stage 2)

Penalises evidence assigned to *wrong* classes by pulling the misleading part of
the Dirichlet toward uniform:

```
α̃ = y + (1 − y) ⊙ α                              # keep true-class evidence
L_kl = KL( Dir(α̃) || Dir(1) )
```

Annealed: `λ_kl = min(1.0, epoch / 10) · 0.1`. Without annealing this term
dominates early and suppresses all evidence.

### 2.7 Total

```
Stage 1: L = L_am + 0.10 · L_compact + 0.05 · L_channel + 0.05 · L_div
Stage 2: L = L_evid + 1.00 · L_unknown + λ_kl(epoch) · L_kl
```

where `L_div` is the sub-prototype diversity penalty from
`05_ARCHITECTURE.md` §6.2:

```
L_div = Σ_c Σ_{i<j} ReLU( ⟨p_{c,i}, p_{c,j}⟩ − 0.8 )²
```

Without it, a class's sub-prototypes collapse onto each other and the
multi-prototype design silently degenerates to single-prototype — which would
re-open the benign-multimodality problem it exists to solve. Monitor mean
intra-class sub-prototype cosine as a training scalar; it should settle below
0.8.

---

## 3. Synthetic unknown generation

`src/argus/losses/unknown_aug.py`. Two generators, mixed 50/50, producing
synthetic samples at a rate of `γ = 0.2` of the real batch size.

### 3.1 Embedding-space mixup (low-density interpolation)

```
pick two distinct classes a ≠ b, both attack classes when possible
pick z_i from class a, z_j from class b
sample μ ~ Uniform(0.35, 0.65)                 # mid-region only
z_synth = normalise( μ z_i + (1−μ) z_j )
label = UNKNOWN
```

Restricting `μ` to the middle band matters: near `μ = 0` or `1` the sample is
essentially a real class member and labelling it UNKNOWN teaches the model to
distrust genuine data.

**Reject** any `z_synth` whose cosine to its nearest prototype exceeds
`cos_reject = 0.9` — that means the interpolation landed inside a real class cone
and would create a label conflict.

### 3.2 Structural pseudo-unknowns (the graph-native generator)

This is the generator that has no analogue in prior non-graph open-set NIDS work,
and is worth naming in the paper.

```
pick a real attack flow e with class a
pick a host v whose neighbourhood profile is atypical for class a
    (largest Euclidean distance in node-feature space among sampled candidates)
rebuild the graph context of e as if it originated at v
recompute h_e through the frozen encoder
label = UNKNOWN
```

Rationale: a novel attack is usually a *known technique in an unfamiliar context*
— the flow-level features look plausible but the relational and temporal context
does not match. Mixup cannot produce that, because it never touches the graph.
This generator tightens the decision boundary along the structural axis, which is
precisely the axis the open-set test exercises.

Cap the candidate search at 64 sampled hosts to bound cost.

### 3.3 Anti-leakage requirement

Synthetic unknowns are generated **only from training-split flows and
training-split graph contexts**. Generating them from validation or test data
leaks the test distribution into threshold selection. Assert this in
`tests/test_epc.py`.

---

## 4. Class imbalance

Prevalence ranges from 94.6% benign (UNSW-NB15) to 0.3% benign (BoT-IoT), and the
rarest attack class has 158 samples. Three mechanisms, applied together:

1. **Class-balanced sampling of target flows.** Within each anchor bin, all flows
   are embedded (the graph needs them), but the **loss** is computed on a
   class-balanced subsample of targets: draw up to `n_per_class = 32` targets per
   class per batch. This decouples graph density from loss balance.
2. **Effective-number class weights** in `L_am`:
   `w_c = (1 − ν) / (1 − ν^{n_c})` with `ν = 0.999`.
3. **Minimum-count guard.** Classes with fewer than 100 training samples are
   excluded from Stage 1 prototype training and instead **evaluated as few-shot
   registration targets**. On UNSW-NB15 this applies to `Worms` (158) — and
   turning the rarest class into a demonstration of C1 rather than a
   closed-set embarrassment is the right move. State it explicitly.

Do **not** oversample rare classes by duplication — with `n_c = 158` this
memorises rather than generalises.

### 4.4 Implementation notes (measured on CICIDS2018)

The three mechanisms interact, and on this dataset the interaction is large
enough to change what the weights should be computed from.

- **Capping is per anchor bin**, so its effect depends on how a class is
  distributed in *time*, not just how many rows it has. At
  `anchor_bin_seconds = 10` and `n_per_class = 32`, `ddos_hoic` (a burst: half
  a million flows in a few dense bins) falls from 27.35% to 3.61% of loss
  targets, while `infiltration` (low and slow, spread thin) *rises* from 7.56%
  to 30.82%. Capping alone takes the imbalance from 1546:1 to 175:1.
- **Weights are therefore computed from the post-cap counts**
  (`losses.class_balance.effective_target_counts`), not the raw ones. Raw
  counts would give `ddos_hoic` and `infiltration` near-identical weights when
  the loss actually sees one 8.5× more often than the other.
- **`ν` is dataset-scaled.** The effective-number correction saturates at
  `1/(1−ν)` samples, so `ν = 0.999` — Cui et al.'s CIFAR-LT value, where `n_c`
  runs 5,000 down to 50 — stops discriminating above ~1,000 while this split's
  post-cap classes run from 53,687 down to 308. Measured, it produces a 3.8×
  total spread against a 175:1 residual imbalance. CICIDS2018 therefore
  overrides `effective_number_nu = 0.9999` (32.8× spread), still well short of
  the 175× that inverse-frequency weighting would apply to 308 rows.

Net effect of all three, as share of the gradient: from **1546:1** between the
largest and smallest trained class to **5.3:1**, with every trained class
receiving at least 3.8%.

---

## 5. Optimisation

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Optimiser | AdamW | AdamW |
| LR | 3e-4 | 5e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | Cosine, 3-epoch linear warmup | Constant |
| Grad clip | 1.0 (global norm) | 1.0 |
| Batch (anchor bins) | 64 | 64 |
| BPTT chunk | 8 bins | 8 bins |
| Epochs | 30 (early stop, patience 5) | 10 (patience 3) |
| Early-stop metric | val macro-F1 (nearest prototype) | val OpenAUC (implemented: val macro-F1) |
| Precision | AMP bf16 (fp16 if unsupported) | AMP bf16 |
| Seeds | 5 seeds: 0, 1, 2, 3, 4 | inherit Stage-1 seed |

Prototype bank is re-normalised to unit norm after **every** optimiser step —
implement as a post-step hook, not as a loss penalty.

### 5.1 Regularisation summary

Every regulariser active in the default configuration, with its purpose. See
`05_ARCHITECTURE.md` §10.2 for the risks these address.

| Regulariser | Value | Applies to | Purpose |
|---|---|---|---|
| Dropout | 0.10 | All MLP hidden layers | Standard |
| DropPath (stochastic depth) | 0.05 | GNN layer residual branches | Forces layer 1 to be independently useful |
| **DropEdge** | 0.10 | Sampled neighbour edges, train only | Graph regularisation **and** implicit A2 adversarial training |
| **Edge-feature dropout** | 0.05 | **Channel A only** | Regularises and biases away from forgeable features simultaneously |
| **Memory dropout** | 0.10 | Per-node GRU state | **Prevents host identity memorisation** — see `02_DATASETS.md` §7 |
| Weight decay | 1e-4 | All weights, not biases/norms | Standard |
| Label smoothing | 0.05 | `L_am` targets | Prevents over-confident prototypes |
| Prototype diversity | `λ_div = 0.05` | Sub-prototypes within a class | Prevents sub-prototype collapse |
| Channel penalty | `λ_ch = 0.05` | Channel-A gradient share | C2 mechanism |
| Grad clip | 1.0 global norm | All | Exploding-gradient guard |
| Early stopping | patience 5 / 3 | — | Standard |
| Normalisation | LayerNorm / GraphNorm | See `05_ARCHITECTURE.md` §9 | **Never BatchNorm** |

Asymmetric edge-feature dropout is worth noting: dropping Channel A features
more aggressively than Channel B is a regulariser *and* a C2 mechanism, since it
teaches the model to survive without the features an attacker can forge. It is
the cheapest robustness intervention in the design.

---

## 6. Kaggle session strategy

12 h sessions, ~16 GB GPU. Plan for interruption.

1. **Checkpoint every epoch** to `results/runs/<run_id>/ckpt_epoch<k>.pt`
   containing model state, optimiser state, scheduler state, RNG states (python,
   numpy, torch, cuda), current epoch, current chunk index, and the node-memory
   dict.
2. **Resume is mandatory and tested.** `scripts/04_train_encoder.py --resume
   <run_id>` must reproduce the same trajectory. Add a test that trains 2 epochs,
   kills, resumes, and compares against an uninterrupted 4-epoch run
   (bit-identical is unrealistic with AMP; assert loss within 1e-3).
3. **Precompute and cache graph batches** to disk as compressed tensors during
   P1. Graph construction is CPU-bound and must not be repeated per epoch or per
   ablation rung. Cache key includes the split hash, scale set, `K`, sampling
   strategy, and seed.
4. **Stage the ablation grid.** The temporal ladder is 8 runs × 5 seeds = 40
   Stage-1 runs. At ~40 min per run that is ~27 h — split across sessions and
   keep a run registry (`results/runs/registry.jsonl`) so completed runs are
   never repeated.
5. If a session is lost, `scripts/15_make_tables.py` must still produce partial
   tables from whatever runs completed, with missing cells marked.

---

## 7. Reproducibility

- Seed python, numpy, torch, and cuda from a single run seed.
- `torch.use_deterministic_algorithms(True)` where feasible; where a
  non-deterministic kernel is required (scatter operations), record it in the run
  manifest rather than silently accepting it.
- Log to `results/runs/<run_id>/`: resolved config (fully expanded, not the
  fragment), git commit SHA, `pip freeze`, GPU model, feature manifest hash,
  split hash, wall-clock per epoch.
- Every number reported in the paper must be traceable to a `run_id`.
  `scripts/13_make_tables.py` emits a provenance column in every generated table.

---

## 8. Training-time sanity gates

Fail fast if any of these trip.

### 8.1 G0 — capacity check (run BEFORE any full training)

Take 1,000 flows spanning all classes. Disable **all** regularisation (dropout,
DropEdge, DropPath, weight decay, label smoothing, memory dropout). Train to
convergence on that subset.

**The model must reach ≳99% training accuracy.** If it cannot overfit 1,000
samples, there is a bug or a capacity bottleneck, and no amount of tuning on the
full dataset will fix it. Likely causes, in order: a broken feature pipeline
emitting constant columns; the channel partition dropping features silently; a
detached tensor in the graph path; `d_h` too small.

This is the cheapest bug-detector available. Run it every time the architecture
changes. It takes minutes.

### 8.2 G1–G7 — during training

| Gate | Condition | Meaning | Action |
|---|---|---|---|
| G1 | Stage-1 val macro-F1 < 0.5 after 5 epochs | Encoder not learning | Halt; check feature pipeline |
| G2 | Mean inter-class prototype cosine > 0.8 after Stage 1 | Prototype geometry collapsed | Raise `m` or `λ_cmp` |
| G2b | Mean intra-class sub-prototype cosine > 0.8 | Sub-prototypes collapsed | Raise `λ_div` |
| G3 | Mean vacuity `u` on known val data > 0.7 after Stage 2 | Evidence collapsed | Lower `λ_kl` or `λ_unk`; check `τ` anneal |
| G4 | Mean `u` on synthetic unknowns < 0.5 after Stage 2 | Unknown carving failed | Raise `λ_unk` |
| G5 | Channel ratio `r` > 0.8 sustained | Relying on forgeable features | Raise `λ_ch` |
| G6 | Any NaN/Inf in loss | Numerical failure | Halt, dump batch. **Check the evidence expression first** (`05_ARCHITECTURE.md` §6.3) |
| G7 | Train − val macro-F1 gap > 0.10 for 3 epochs | Overfitting | Raise dropout/DropEdge; check identity leakage |
| G8 | Any trained class at val F1 = 0 on the selected epoch | Tail collapse | Halt; check target ordering and class balancing (`BUGS.md` #49/#52) |


**Why G8 is not redundant with G0 or G1.** G0 trains on a *class-balanced*
subset, so a model that has stopped predicting the tail entirely still
memorises that subset and passes at 0.99. G1 and G7 are aggregates, and on
CICIDS2018 ignoring the six rarest classes costs 0.2 points of accuracy and
little macro-F1 movement early on. Only a per-class floor makes the failure
visible — three consecutive 12-hour runs shipped a near-constant single-class
predictor with every other gate green. Minimum-count classes (§4.3) are excluded
from G8, since they are deliberately not trained.

### 8.3 Gradient-health monitors (every 100 steps)

Logged as scalars; these diagnose vanishing and exploding gradients before they
become silent failures.

| Monitor | Healthy range | Interpretation outside it |
|---|---|---|
| Global grad norm (pre-clip) | 0.01 – 50 | > 50 exploding; < 1e-4 sustained means a dead path |
| Clip activation rate | < 30% of steps | Higher means LR too high or `τ` annealed too fast |
| Per-module grad norm ratio `max/min` | < 1000 | Higher means one module is starved — identify it from the per-module log |
| `log e` clamp saturation rate | < 1% after annealing | Higher means `τ` is too small for the achieved geometry |
| Mean `‖z_e‖` before normalisation | 0.1 – 100 | Collapse toward 0 makes the L2 normalisation numerically unstable |
| Multi-scale gate entropy | > 0.5 nats | Near 0 means fusion degenerated to one scale |

Per-module norms are tracked for
`{edge_proj_A, edge_proj_B, msg_mlp, attention, aggregation, gru, edge_readout,
fusion, embedding, prototypes}`. The A/B split is deliberate: comparing
`edge_proj_A` against `edge_proj_B` gradient norms is a direct read on whether
the channel penalty is doing its job.

All monitors are written to `results/runs/<run_id>/gradient_health.jsonl` and
plotted by `scripts/15_make_tables.py`. G2–G5 and the gate scalars also make a
useful appendix figure.
