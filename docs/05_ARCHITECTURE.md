# 05 — Architecture

Two components: **SR-TEG** (Structure-Robust Temporal Edge GNN) encoder and
**EPC** (Evidential Prototype Classifier) head.

Module map: `src/argus/models/` — `time_encoding.py`, `aggregation.py`,
`attention.py`, `memory.py`, `srteg.py`, `multiscale.py`, `epc.py`,
`prototypes.py`.

---

## 0. Symbols and default dimensions

| Symbol | Meaning | Default |
|---|---|---:|
| `F_e` | Edge feature dim (from feature manifest) | 147 |
| `F_v` | Node feature dim | 18 |
| `d_h` | Hidden dim | **128** |
| `d_A` | Channel-A (controllable) projection dim | 48 |
| `d_B` | Channel-B (observer) projection dim | 80 |
| `d_t` | Time-encoding dim | 16 |
| `d_z` | Embedding dim (prototype space) | 64 |
| `L` | GNN layers | 2 |
| `H` | Attention heads | 4 |
| `K` | Neighbour cap | 32 |
| `β` | Trim fraction | 0.20 |
| `S` | Scales (short/mid/long) | 3 |
| `C` | Known classes (benign + attacks) | dataset-dependent |
| `M_c` | Sub-prototypes per class | 4 benign / 2 attack |

> `d_h` was raised from 64 to 128 because `F_e = 147` compressed into `2 × 32`
> was an underfitting bottleneck — see §10.3. `d_A < d_B` deliberately caps the
> capacity available to attacker-controllable features (§3.2).

---

## 1. Forward pass overview

```
edge features x_e ─┬─ Channel A proj (controllable)  ─┐
                   └─ Channel B proj (observer)      ─┤
node features x_v ─────────────────────────────────── ├─→ SR-TEG per scale s
Δt ─→ Time2Vec φ(Δt) ─────────────────────────────────┘         │
                                                                 ↓
                              h_e^S, h_e^M, h_e^L  ──→ gated fusion (TE5)
                                                                 ↓
                                                          h_e ∈ R^{d_h}
                                                                 ↓
                                          projection + L2 norm → z_e ∈ S^{d_z−1}
                                                                 ↓
                    prototype bank P ∈ R^{C×d_z} → distances → Dirichlet evidence
                                                                 ↓
                                       {CLASSIFY k | DEFER | UNKNOWN}
```

---

## 2. Time encoding (TE3)

`src/argus/models/time_encoding.py`

For a neighbour edge `e` sampled into the context of target time `t`, define
`Δt = (t − t_e) / 1000` in seconds, `Δt ≥ 0`.

**Time2Vec** with `d_t = 16`:

```
φ(Δt)[0]   = ω_0 · Δt + b_0                       # linear term, preserves magnitude
φ(Δt)[i]   = sin(ω_i · Δt + b_i)   for i = 1..d_t−1
```

`ω ∈ R^{d_t}`, `b ∈ R^{d_t}` are learnable.

**Initialisation matters.** Initialise `ω_i` for `i ≥ 1` on a log-uniform grid
spanning the periods we care about:

```
periods = logspace(log10(0.1), log10(600), d_t - 1)     # 0.1 s … 600 s
ω_i = 2π / periods[i-1]
b_i ~ Uniform(0, 2π)
ω_0 = 1 / 300 ,  b_0 = 0
```

Random initialisation of `ω` makes the sinusoidal basis very slow to discover
periodicity; the log-grid gives it beaconing intervals from 100 ms to 10 minutes
on the first step. This is a small detail with a large effect — do not skip it.

An alternative Bochner/TGAT encoding is available behind
`model.time_encoding: time2vec | bochner`; Time2Vec is the default because it is
simpler and equally expressive here.

**Ablation hook.** When `te3_enabled = false`, replace `φ(Δt)` with the scalar
`[log1p(Δt)]` broadcast to `d_t` zeros plus that one value (i.e. recency only,
no periodicity).

---

## 3. SR-TEG layer

`src/argus/models/srteg.py`. One layer is applied `L = 2` times, per scale.
Weights are **shared across scales** (TE5) — this is what keeps 3 scales
affordable.

### 3.1 Inputs

- Node states `h_v^{(l)} ∈ R^{d_h}`, with `h_v^{(0)} = W_v x_v + m_v`, where
  `m_v` is the GRU memory (§4) and `W_v ∈ R^{d_h × F_v}`.
- Edge features `x_e`, split into `x_e^A` (controllable) and `x_e^B` (observer).

### 3.2 Provenance-partitioned edge projection

```
u_e^A = GELU(LayerNorm(W_A x_e^A + b_A))     W_A ∈ R^{d_A × |A|},  d_A = 48
u_e^B = GELU(LayerNorm(W_B x_e^B + b_B))     W_B ∈ R^{d_B × |B|},  d_B = 80
u_e   = concat(u_e^A, u_e^B) ∈ R^{d_h}       d_A + d_B = d_h = 128
```

The two channels are projected **separately and never mixed before this
concatenation**, so the gradient attributable to each channel is cleanly
separable — which is what makes the channel penalty in `06_TRAINING.md` §2.4
well-defined.

**Asymmetric widths are deliberate.** Channel A holds ~100 of the 147 features
but receives only 48 dimensions; Channel B holds ~47 features and receives 80.
This is an architectural prior favouring the hard-to-forge channel, and it does
part of the C2 job structurally rather than relying solely on the gradient
penalty. Report `d_A / d_B` as an ablation axis alongside `λ_ch`.

LayerNorm before the non-linearity (not BatchNorm — see §9) keeps the two
channels on comparable scales despite their very different input widths, which
matters because the channel-ratio penalty compares their gradient norms.

### 3.3 Message construction

For a directed edge `e = (u → v)` sampled into `v`'s neighbourhood:

```
m_{e→v} = MLP_msg( concat[ h_u^{(l)} , u_e , φ(Δt_e) ] )
```
`MLP_msg : R^{d_h + d_h + d_t} → R^{d_h}`, one hidden layer of width `2·d_h`,
GELU, LayerNorm.

### 3.4 Time-decayed attention (TE4)

Multi-head, `H = 4` heads, per-head dim `d_h/H = 16`.

```
raw score:     s_{e,v}^{(h)} = ⟨ W_Q^{(h)} h_v , W_K^{(h)} m_{e→v} ⟩ / sqrt(d_h/H)
time decay:    δ_{e}^{(h)}   = − λ^{(h)} · Δt_e            ,  λ^{(h)} = softplus(λ̂^{(h)}) ≥ 0
attention:     α_{e,v}^{(h)} = softmax_e ( s_{e,v}^{(h)} + δ_{e}^{(h)} )
```

The decay is applied **inside the softmax as an additive log-space bias**, not as
a multiplicative post-hoc factor — this keeps the attention weights a proper
distribution and keeps gradients well-behaved.

Initialise `λ̂^{(h)}` so the half-life is `D_L / 4 = 75 s` for the long scale,
scaled per head so heads cover different horizons:

```
half_lives = [D_s/16, D_s/8, D_s/4, D_s/2]     for the current scale s
λ^{(h)} = ln(2) / half_lives[h]
λ̂^{(h)} = softplus_inverse(λ^{(h)})
```

Because weights are shared across scales but `Δt` ranges differ by 300×, feed
`Δt` **normalised by the scale duration** (`Δt / D_s`) into both `φ(·)` and the
decay. Otherwise the short-scale head sees `Δt ∈ [0,1]` and the long-scale head
sees `Δt ∈ [0,300]`, and shared `λ` cannot serve both.

**Ablation hook.** `te4_enabled = false` sets `δ ≡ 0` (plain attention).

### 3.5 Robust aggregation

`src/argus/models/aggregation.py`. This is the core of C2.

Three modes, selectable by `model.aggregation`:

**`mean`** — baseline, matches E-GraphSAGE. Unbounded influence: a single
extreme neighbour moves the aggregate arbitrarily far.

**`trimmed`** (DEFAULT) — coordinate-wise trimmed mean. For each of the `d_h`
output coordinates independently, sort the `n ≤ K` attention-weighted messages by
value, drop the lowest `⌊βn⌋` and highest `⌊βn⌋`, and average the rest.

```python
def trimmed_mean(msgs, weights, beta=0.25):
    # msgs: [n, d_h], weights: [n] summing to 1
    n = msgs.shape[0]
    k = int(beta * n)
    if n - 2 * k < 1:                      # degenerate; fall back
        return (weights.unsqueeze(-1) * msgs).sum(0)
    order = msgs.argsort(dim=0)            # per-coordinate ordering
    keep  = order[k : n - k]               # [n-2k, d_h]
    m     = torch.gather(msgs,    0, keep)
    w     = torch.gather(weights.unsqueeze(-1).expand_as(msgs), 0, keep)
    return (w * m).sum(0) / (w.sum(0) + 1e-9)
```

Breakdown point `β`: the aggregate is provably unmoved while the adversary
controls fewer than `βn` of the sampled neighbours, *per coordinate*.

> **Default lowered from β = 0.25 to β = 0.20.** At β = 0.25 the trimmed mean
> discards half of all messages (`βn` from each tail), which is a large
> expressiveness loss and a genuine underfitting risk. β = 0.20 retains 60% of
> messages while still requiring an adversary to control a fifth of the
> neighbourhood. The full range {0, 0.1, 0.2, 0.25, 0.4} remains an ablation
> axis and the robustness–accuracy frontier is reported over it.

**`soft_medoid`** — differentiable medoid (Geisler et al., NeurIPS 2020).
Weights each message by its total distance to all others:

```
c_i = softmax_i ( − (1/T) · Σ_j ||m_i − m_j||_2 )
agg = Σ_i (c_i · w_i / Σ_j c_j w_j) · m_i
```
`T` is a temperature (default `T = 1.0`, tune in {0.1, 1, 10}). Stronger
robustness guarantee, ~2× the cost of `trimmed` because of the pairwise distance
matrix (`O(K²d_h)`, tolerable at `K = 32`).

Run all three in the C2 ablation. Expect `trimmed` to be the best
robustness/accuracy/cost compromise and `soft_medoid` to be the most robust.

### 3.5b Multi-aggregator readout — recovering expressiveness

Any single robust aggregator loses expressive power relative to the sum
aggregator (this is the Principal Neighbourhood Aggregation argument: one
aggregator cannot distinguish all multisets). Trimming compounds it by
discarding 40% of messages. Left unaddressed this is an **underfitting risk**,
and it would show up as ARGUS failing to reach parity with E-GraphSAGE in P3.

Mitigation: concatenate a small set of complementary statistics, then project
back to `d_h`.

```
a_robust = trimmed_mean(msgs, weights, β)          # d_h — robust, primary
a_scale  = log1p(n) / log1p(K)                     # 1   — degree scalar
a_spread = trimmed_std(msgs, β)                    # d_h — robust dispersion
agg_v    = W_agg · concat[a_robust, a_spread, a_scale·a_robust]   → R^{d_h}
```

Design rules that keep this from re-opening the attack surface:

- **No `max` or `sum` aggregator.** Both have unbounded sensitivity to a single
  injected neighbour and would void the breakdown-point argument. `a_spread`
  uses the *trimmed* standard deviation, which inherits the same breakdown point
  as the trimmed mean.
- The degree scalar uses the **capped** count `n ≤ K` and is log-compressed, so
  a flooding attacker cannot drive it arbitrarily.
- `a_spread` is what lets the model distinguish "quiet host with one odd flow"
  from "host with uniformly odd flows" — a distinction the mean alone erases,
  and one that matters for scanning and DDoS.

Ablation `E-ABL-AGGX`: `{robust only, robust+spread, robust+spread+scale}`.
If the single-aggregator variant matches, prefer it for simplicity and say so.

### 3.6 Node update

```
h_v^{(l+1)} = h_v^{(l)} + DropPath_p( MLP_upd( GraphNorm( concat[h_v^{(l)}, agg_v] ) ) )
```

Pre-normalisation residual (norm inside the branch, not after the addition).
`MLP_upd : R^{2 d_h} → R^{d_h}`, one hidden layer of width `2 d_h`, GELU,
dropout `0.1`.

Three deliberate choices, each defended in §9:

- **Pre-norm, not post-norm.** Post-norm (`LayerNorm(h + f(h))`) rescales the
  residual stream at every layer and attenuates the identity path, which is the
  classic route to vanishing gradients in deep residual stacks. Pre-norm leaves
  a clean identity path from the loss to layer 0.
- **GraphNorm, not BatchNorm.** See §9.1 — BatchNorm is actively harmful here.
- **DropPath (stochastic depth), `p = 0.05`.** At `L = 2` this is mild
  regularisation and, more usefully, it forces layer 1 to be independently
  useful rather than relying on layer 2 to fix its output.

### 3.7 Edge representation readout

After `L` layers, the representation of a **target edge** `e = (u → v)`:

```
h_e^{(s)} = MLP_edge( concat[ h_u^{(L)} , h_v^{(L)} , u_e , φ(Δt=0) ] )
```
`MLP_edge : R^{3 d_h + d_t} → R^{d_h}`, one hidden layer `2 d_h`, GELU,
LayerNorm. Both endpoints are included because direction matters (a scan source
and a scan target play different roles).

---

## 4. Per-node temporal memory (TE6)

`src/argus/models/memory.py`

One GRU cell, shared across all nodes, state `m_v ∈ R^{d_h}`. Updated **once per
anchor bin**, using the long-scale aggregate:

```
m_v ← GRUCell( input = agg_v^{L, layer=L} , hidden = m_v )
```

- Initialised to zeros at split start and for any newly-seen host.
- Detached at truncated-BPTT chunk boundaries (`T_bptt = 8` bins).
- Evicted after `2 · D_L = 600 s` of node inactivity — this bounds memory in
  deployment and must be reflected in the streaming measurement.
- Stored in a dict `{node_global_id: tensor}`, not a dense `[N_hosts, d_h]`
  matrix; host counts reach 181,876 on NF-CICIDS2018 and most are inactive.
- **Memory dropout `p_mem = 0.1`** — during training, zero a node's memory with
  probability `p_mem` before the forward pass.
- GRU input and hidden are **LayerNormed** before the cell (see §9.2); the
  candidate activation is `tanh`, whose saturation is the main vanishing-gradient
  source in a recurrent stack.

> **Memory dropout is not routine regularisation — it defends against host
> identity memorisation.** A GRU state that persists per host can encode "this
> host is the attacker" rather than "this behaviour is an attack". On
> NF-UNSW-NB15, with only 40 unique source IPs, a model could reach very high
> accuracy purely by memorising which of 40 hosts is malicious, and every
> reported number would be meaningless. Memory dropout forces the per-flow
> evidence to stand on its own. This must be paired with the identity-leakage
> audit in `02_DATASETS.md` §7 — the audit measures the hazard, the dropout
> mitigates it, and neither substitutes for the other.

This is deliberately **not** a continuous-time TGN memory. One update per bin,
bounded state, no message-store, no time-encoded memory reads. Cheaper, simpler,
reproducible.

**Ablation hook.** `te6_enabled = false` sets `m_v ≡ 0`.

---

## 5. Multi-scale fusion (TE5)

`src/argus/models/multiscale.py`

Three scale-specific edge representations `h_e^S, h_e^M, h_e^L ∈ R^{d_h}` from
the **weight-shared** encoder. Fuse with a learned gate:

```
g = softmax( W_g · concat[h_e^S, h_e^M, h_e^L] + b_g )     g ∈ Δ², W_g ∈ R^{3 × 3d_h}
h_e = Σ_{s∈{S,M,L}} g_s · h_e^{(s)}
```

Then `h_e ← LayerNorm(h_e + MLP_fuse(concat[h_e^S, h_e^M, h_e^L]))`.

**The gate `g` is an interpretability output.** Log its mean value per attack
class — this directly visualises C3, showing e.g. DDoS relying on `g_S` and
Infiltration on `g_L`. Include as a figure (`08_EVALUATION.md` figure F5).

Scale-specific bias: give each scale a small learned embedding
`ε_s ∈ R^{d_h}` added to `u_e` before encoding, so the shared weights can still
distinguish which scale they are operating on.

**Ablation hook.** `te5_enabled = false` uses only the mid scale (`h_e = h_e^M`).

---

## 6. EPC head

`src/argus/models/epc.py` and `prototypes.py`. This is C1.

### 6.1 Embedding

```
z_e = W_z h_e + b_z ,  z_e ← z_e / (||z_e||_2 + eps)      z_e ∈ S^{d_z − 1}
```
Unit-norm embedding on the hypersphere, `d_z = 64`.

### 6.2 Prototype bank — multi-prototype per class

`P ∈ R^{(Σ_c M_c) × d_z}`, each row unit-norm, with a row→class map. Each class
`c` owns `M_c` sub-prototypes.

| Class type | `M_c` | Rationale |
|---|---:|---|
| Benign | **4** | Benign traffic is strongly multi-modal (web, DNS, SSH, backup, telemetry…) |
| Attack, ≥10k samples | 2 | Most attack families have variants |
| Attack, <10k samples | 1 | Not enough data to fit more |
| Registered (few-shot) | 1 | Only `n` samples available |

**Why multi-prototype is not optional.** A single benign prototype must cover
every legitimate traffic pattern in the network. Forcing that into one cone has
exactly two outcomes, both fatal: either the cone grows wide enough to swallow
genuine attacks and unknown detection collapses, or it stays tight and most
benign traffic is flagged UNKNOWN. On NF-CICIDS2018 benign is 87% of traffic
across many services; a single cone cannot represent it. This is the single most
likely cause of a disappointing OpenAUC, and it is cheap to prevent.

Class score is a max over the class's sub-prototypes:

```
cos_c = max_{j ∈ 1..M_c} ⟨ z_e , p_{c,j} ⟩
```

Sub-prototypes are initialised by k-means on the Stage-1 embeddings of each
class (after 3 warm-up epochs), then trained. A **diversity penalty** keeps them
from collapsing onto each other:

```
L_div = Σ_c Σ_{i<j} ReLU( ⟨p_{c,i}, p_{c,j}⟩ − 0.8 )²        weight λ_div = 0.05
```

Two population modes:

- **Trained prototypes** (classes present at training) — `nn.Parameter`,
  re-normalised to unit norm after every optimiser step.
- **Registered prototypes** (classes added post-training) — normalised mean
  embedding of `n` labelled samples, appended with `requires_grad = False`.

```python
@torch.no_grad()
def register(self, name: str, embeddings: Tensor, n_sub: int = 1) -> int:
    if n_sub > 1 and embeddings.shape[0] >= 4 * n_sub:
        centres = kmeans(embeddings, k=n_sub)        # optional, n permitting
    else:
        centres = embeddings.mean(0, keepdim=True)
    centres = centres / (centres.norm(dim=-1, keepdim=True) + 1e-9)
    self.bank = torch.cat([self.bank, centres], dim=0)
    self.class_of.extend([len(self.names)] * centres.shape[0])
    self.names.append(name)
    return len(self.names) - 1
```

No gradient steps, no optimiser, no data replay. Forgetting on previously known
classes is **exactly zero** because no existing parameter changes — this is a
structural guarantee, and `tests/test_epc.py` must assert bit-identical logits on
a held-out batch before and after registration.

### 6.3 Distance and evidence — computed in log space

Cosine distance to each class (max over its sub-prototypes):

```
cos_c = max_j ⟨ z_e , p_{c,j} ⟩ ∈ [−1, 1]
d_c   = 1 − cos_c              ∈ [0, 2]
```

Convert to Dirichlet evidence with margin `m` and temperature `τ`:

```
log e_c = − (d_c − m) / τ
e_c     = exp(log e_c)
α_c     = e_c + 1                    Dirichlet concentration
S       = Σ_c α_c = Σ_c e_c + C      total strength
p̂_c     = α_c / S                    expected class probability
u       = C / S                      vacuity ∈ (0, 1]
```

> **⚠ Numerical hazard — this is the most dangerous expression in the model.**
> With `τ = 0.1`, `d_c ∈ [0,2]` and `m = 0.35`, the exponent
> `−(d_c − m)/τ` spans **[−16.5, +3.5]**, so `e_c` spans `6·10⁻⁸` to `33`. The
> `1/τ` factor multiplies every gradient flowing back through the head by 10.
> Naïve evaluation overflows in fp16 and produces exploding gradients in fp32.
>
> **Mandatory implementation rules:**
> 1. Never materialise `e_c` directly. Clamp `log e_c` to `[−15, +15]` before
>    exponentiating.
> 2. Compute `log S = logsumexp([log e_1 … log e_C, log C])`, and derive
>    `log p̂_c = log α_c − log S`, `log u = log C − log S`.
> 3. Keep the head in **fp32 even under AMP** (`autocast(enabled=False)` around
>    it). bf16 has adequate range but poor precision exactly where `p̂` is near 1.
> 4. Parameterise `τ = softplus(τ̂) + τ_min` with `τ_min = 0.02` so it can never
>    reach zero and produce an infinite gradient multiplier.
> 5. **Anneal `τ` from 1.0 down to 0.1** over Stage-2 epochs. Starting at
>    `τ = 0.1` from a randomly-placed embedding is a gradient bomb: every
>    distance is near-identical, so every evidence is near-identical, and the
>    `1/τ` amplification turns that uniform signal into a large, uninformative
>    gradient. See `06_TRAINING.md` §2.3.
>
> If gate G6 (NaN loss) fires in Stage 2, this expression is the first place to
> look.

Defaults `m = 0.35`, `τ_final = 0.10`. Both tuned on validation
(`07_HYPERPARAMETERS.md`).

Note `u → 1` when all evidence is near zero (input far from every prototype) and
`u → 0` when some prototype attracts large evidence. This single scalar drives
the UNKNOWN decision.

### 6.4 Three-way decision rule

```
E_total = Σ_k e_k                                  # total evidence
p̂_max   = max_k p̂_k
p̂_2nd   = second largest p̂_k
margin  = p̂_max − p̂_2nd

if E_total < θ_unknown:            → UNKNOWN
elif margin < θ_defer:             → DEFER
else:                              → CLASSIFY argmax_k p̂_k
```

Thresholds `θ_unknown`, `θ_defer` are **selected on the validation split only**,
never on test:

- `θ_unknown` at the value giving a target 5% false-UNKNOWN rate on known-class
  validation data. Also sweep it to produce the openness/OpenAUC curves.
- `θ_defer` at the value giving a target 2% deferral rate on validation. Also
  sweep it to produce the risk–coverage curve.

Record both chosen values in the run artifact. Report the full sweeps, not just
the operating point — a single threshold is cherry-pickable, a curve is not.

### 6.5 Why one head does three jobs

The prior scoping work (`plan/previous_work.txt`) required a separate one-class
Deep SVDD per position for deferral, plus a distinct mechanism for unknowns.
Here, "defer" and "unknown" are two different readings of the same Dirichlet
posterior: **spread evidence** (high `S`, low margin) means the model is
confident something is happening but cannot disambiguate known classes;
**absent evidence** (low `S`) means the input resembles nothing known. One head,
one forward pass, one calibration story. This unification is the core of C1 and
should be stated as such in the paper.

---

## 7. Parameter budget

At defaults (`d_h = 128`, `L = 2`, `H = 4`, `F_e = 147`, `F_v = 18`, `d_z = 64`):

| Component | Approx. params |
|---|---:|
| Edge channel projections `W_A`, `W_B` | ~9.6 k |
| Node projection `W_v` | ~2.4 k |
| Time2Vec | 32 |
| `MLP_msg` × L | ~2 × 98 k = 196 k |
| Attention `W_Q`, `W_K` × L | ~2 × 33 k = 66 k |
| `MLP_upd` × L | ~2 × 98 k = 196 k |
| Multi-aggregator `W_agg` × L | ~2 × 33 k = 66 k |
| GRUCell | ~99 k |
| `MLP_edge` | ~131 k |
| Fusion gate + `MLP_fuse` | ~101 k |
| Embedding `W_z` | ~8 k |
| Prototype bank (C = 10, ~16 sub-prototypes) | ~1 k |
| **Total** | **≈ 875 k** |

~3.5 MB in fp32. Still small: at 4 M training flows this is roughly 4.6 flows
per parameter, which is a comfortable regime and not an overfitting risk from
capacity alone (the real overfitting risks are per-class and identity-based —
see §10.2). Do not scale `d_h` past 128 without re-checking the latency budget.

---

## 8. Baseline model specs

`src/argus/models/baselines/`. All must consume the **same** graph batches so the
comparison is clean.

| Baseline | Spec |
|---|---|
| **E-GraphSAGE** | Constant node init (ones, dim 1 → projected to `d_h`), mean aggregation over *uncapped* neighbourhoods, 2 layers, edge readout = concat of endpoint states + edge features, softmax head. Single scale (mid). No time encoding. |
| **EGATv2 / E-ResGAT** | GATv2 attention, residual connections, 2 layers, 4 heads, softmax head. Single scale. No time decay. |
| **Anomal-E** | DGI-style self-supervised edge embeddings, then Isolation Forest / one-class SVM on embeddings. Binary output; map to multi-class by thresholding only for the binary table. |
| **Extra Trees / Random Forest** | Flow-independent, on the same `F_e` vector, no graph. 200 trees, `class_weight="balanced_subsample"`. **Mandatory** — reviewers will ask, and the flow-timeout paper shows tree ensembles are the stable non-DL bar. |
| **Identity-only classifier** | Decision tree on `(src IP, dst IP, src port, dst port)` **only**. Not a competitor — a **leakage floor**. See `02_DATASETS.md` §7. |
| **Softmax + threshold** | ARGUS encoder with a plain softmax head and max-softmax-probability thresholding for unknowns. **The key ablation isolating the EPC head.** |
| **OpenMax / Energy score** | Same encoder, post-hoc open-set scoring. Isolates "is the prototype geometry needed, or would post-hoc scoring do?" |
| **MSPL** | Non-graph multi-space prototypical network (arXiv:2501.00050). Closest non-graph mechanism. |
| **CLOSR** | Non-graph contrastive OSR (code at `github.com/jackwilkie/CLOSR`). Reports OpenAUC natively. |

---

## 9. Normalisation

### 9.1 BatchNorm is rejected — and the reasons are load-bearing

BatchNorm is the default choice in most deep architectures and is deliberately
**not used anywhere in ARGUS**. Five independent reasons, any one of which would
be sufficient:

1. **Batch composition is pathological under class imbalance.** Benign is
   61–99.7% of traffic. A batch's running mean and variance are effectively
   benign statistics, and attack flows are then normalised by a distribution
   they do not belong to. Minority-class representations get systematically
   distorted — precisely the classes that matter.
2. **Graph batches have variable node and edge counts.** Anchor bins differ
   wildly in density (a DDoS bin has orders of magnitude more flows than a quiet
   bin), so batch statistics are non-stationary in a way unrelated to the
   learning signal.
3. **Streaming inference breaks the train/test contract.** `StreamingDetector`
   processes one anchor bin at a time, sometimes containing a single flow.
   BatchNorm's train-time batch statistics versus eval-time running statistics is
   a classic source of silent deployment degradation, and our deployment claim
   depends on train and inference being the same code path.
4. **It interacts badly with the recurrent memory.** Normalising across a batch
   dimension inside a stateful, sequential-over-bins model mixes statistics
   across time steps that are not exchangeable.
5. **It is an attack surface.** BatchNorm makes a sample's normalisation depend
   on *other samples in the batch*. An adversary who controls part of a batch can
   shift the statistics used to normalise a victim's flow. For a paper whose C2
   claim is structural robustness, shipping a layer with a documented
   batch-level attack surface would be self-defeating. This point is worth one
   sentence in the paper.

### 9.2 What is used instead

| Location | Normaliser | Why |
|---|---|---|
| Edge channel projections | **LayerNorm** | Puts the two provenance channels on comparable scales despite very different input widths; required for the channel-ratio penalty to be meaningful |
| Node update (§3.6) | **GraphNorm** | Designed for GNNs; normalises per graph with a learnable mean-shift, empirically better than LayerNorm on graph tasks and immune to reasons 1–5 above |
| `MLP_msg`, `MLP_edge`, `MLP_fuse` | **LayerNorm** (pre-norm) | Standard, per-sample, batch-independent |
| GRU input and hidden | **LayerNorm** | Prevents `tanh`/sigmoid saturation, the main vanishing-gradient source in the recurrent path |
| Embedding `z_e` | **L2 normalisation** | Required by construction — the prototype space is a hypersphere |
| Prototype bank | **L2 normalisation** after every optimiser step | Keeps cosine geometry valid |
| Input features | Quantile-to-normal (TE1) | See `03_FEATURE_ENGINEERING.md` §3 |

All are **per-sample or per-graph**, never per-batch. This makes the model
batch-composition-invariant: a flow's verdict does not depend on which other
flows happen to share its batch. That property is worth stating explicitly in
the paper, because it is unusual and it is what makes single-flow streaming
inference legitimate.

`RMSNorm` may be substituted for `LayerNorm` as a throughput optimisation if the
deployment budget is tight; ablate rather than assume.

---

## 10. Failure-mode prevention

Three named failure modes, each with a mechanism and a detection gate. Detection
matters as much as prevention: a model that silently underfits looks like a
model that has converged.

### 10.1 Vanishing and exploding gradients

| Source | Risk | Mitigation |
|---|---|---|
| Evidence `exp(−(d−m)/τ)`, `τ = 0.1` | **Exploding — the worst in the model.** `1/τ` multiplies every head gradient by 10; exponent range is `[−16.5, 3.5]` | Log-space computation, clamp `log e` to ±15, fp32 head, `τ = softplus(τ̂) + 0.02`, anneal `τ` 1.0 → 0.1 (§6.3) |
| AM-Softmax scale `s = 30` | Exploding — logits scaled 30× | Warm up `s` from 10 → 30 over 5 epochs; keep global grad clip at 1.0 |
| Channel penalty (double backward) | Exploding — second-order gradients | Clamp the ratio before squaring; apply every 4th batch; detach the denominator |
| Time2Vec, shortest period 0.1 s | Exploding — `ω ≈ 62.8`, so `∂sin(ωΔt)/∂Δt` is large | Δt normalised by scale duration; `ω` gradient scaled by `1/ω_init`; clamp `ω` to `[0.1·ω_init, 10·ω_init]` |
| Soft-medoid at `T = 0.1` | Exploding | Compute the softmax over `−dist/T` with `logsumexp`; floor `T` at 0.05 |
| GRU over `T_bptt = 8` bins | Vanishing — `tanh` saturation | LayerNorm on input and hidden; truncated BPTT at 8; orthogonal init for recurrent weights |
| Post-norm residuals | Vanishing — identity path attenuated each layer | **Pre-norm residuals** (§3.6) |
| Deep MLP stacks | Vanishing | GELU (no dead-unit region), residual connections, depth held at `L = 2` |

**Detection gates** (logged every 100 steps, escalate per `06_TRAINING.md` §8):

- Global grad norm — alert if `> 50` (pre-clip) or `< 1e-4` sustained.
- **Per-module grad norm** for `{edge_proj, msg_mlp, attention, aggregation,
  gru, edge_readout, fusion, embedding, prototypes}`. A ratio of
  `max_module / min_module > 1000` indicates a vanishing path; find it early.
- Fraction of steps where clipping activates — if `> 30%`, the LR is too high or
  `τ` has been annealed too fast.
- `log e` saturation rate — fraction of entries hitting the ±15 clamp. Should be
  `< 1%` after annealing; higher means `τ` is too small for the achieved
  geometry.

### 10.2 Overfitting

Capacity is not the risk (875 k parameters on 4 M flows). Three specific risks:

**Risk A — host identity memorisation.** The most dangerous, and specific to
NIDS. See §4: on UNSW-NB15 (40 IPs) a model can memorise attacker identity and
report near-perfect accuracy that means nothing. Mitigations: node identity never
enters the feature vector; memory dropout `p_mem = 0.1`; and the mandatory
**identity-leakage audit** (`02_DATASETS.md` §7) that quantifies the hazard as a
reported baseline.

**Risk B — tiny classes.** `Worms` (158), `Theft` (1,615), `Web Attacks` (2,538),
`Ransomware` (3,971). A prototype fitted to 158 samples memorises them.
Mitigation: the `min_count = 100` guard routes such classes to few-shot
registration instead of prototype training (`06_TRAINING.md` §4) — which converts
the problem into a demonstration of C1.

**Risk C — temporal near-duplicates.** Flows from the same session, seconds
apart, are near-identical. If duplicates straddle the split boundary the test set
is contaminated. Mitigation: exact deduplication before splitting, plus a
reported near-duplicate rate (`02_DATASETS.md` §5.1).

Standard regularisers, all active: dropout `0.1`; DropPath `0.05`; **DropEdge
`0.1`** (randomly drop sampled neighbour edges — doubles as A2 adversarial
training); **edge-feature dropout `0.05`** on Channel A only, which regularises
and biases the model away from forgeable features simultaneously; weight decay
`1e-4`; label smoothing `0.05`; prototype diversity penalty; early stopping.

**Detection gate:** track the train−val macro-F1 gap every epoch. A gap `> 0.10`
sustained for 3 epochs triggers a warning and is recorded in the run manifest.

### 10.3 Underfitting

Under-appreciated here because three design choices *deliberately remove
capacity* in exchange for robustness:

| Choice | Capacity cost | Compensation |
|---|---|---|
| Trimmed aggregation `β = 0.20` | Discards 40% of messages | `β` lowered from 0.25; multi-aggregator readout (§3.5b) |
| Degree cap `K = 32` | Bounded receptive field | `K` is an ablation axis; 64 available |
| Channel penalty `ρ = 0.5` | Suppresses reliance on ~100 features | `ρ` tolerates 50% reliance; ablate at 0 |
| Weight sharing across 3 scales | One encoder does three jobs | Per-scale embedding `ε_s`; `d_h` raised to 128 |

**Mandatory capacity sanity check, before any full training run.** Take 1,000
flows spanning all classes and train to convergence with all regularisation
disabled. The model **must** reach ≳99% training accuracy. If it cannot overfit
1,000 samples, there is a bug or a capacity bottleneck, and no amount of
hyperparameter tuning on the full dataset will fix it. This is gate G0 and it is
the cheapest bug-detector available — run it first, every time the architecture
changes.

**Detection gates:** training macro-F1 plateauing below 0.75 by epoch 10 (G1);
mean inter-prototype cosine above 0.8, meaning the geometry collapsed (G2);
multi-scale gate `g` degenerate (one scale above 0.9 for every class), meaning
fusion is not learning.

---

## 11. Configuration surface

`config/model/argus.yaml`:

```yaml
model:
  name: argus
  d_h: 64
  d_t: 16
  d_z: 64
  layers: 2
  heads: 4
  neighbour_cap: 32
  sampling: recency_stratified     # recent | uniform | recency_stratified
  strata: 4
  aggregation: trimmed             # mean | trimmed | soft_medoid
  trim_beta: 0.25
  soft_medoid_temp: 1.0
  time_encoding: time2vec          # time2vec | bochner
  dropout: 0.1
  # temporal ablation switches (see 09_TEMPORAL_STUDY.md)
  te1_enabled: true
  te2_enabled: true
  te3_enabled: true
  te4_enabled: true
  te5_enabled: true
  te6_enabled: true
  te7_enabled: true
head:
  type: epc                        # epc | softmax | openmax | energy
  margin_m: 0.35
  tau: 0.10
  am_softmax_margin: 0.35
  am_softmax_scale: 30.0
  theta_unknown: null              # selected on validation
  theta_defer: null                # selected on validation
  target_false_unknown_rate: 0.05
  target_defer_rate: 0.02
```
