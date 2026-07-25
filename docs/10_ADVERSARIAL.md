# 10 — Adversarial Evaluation

The evidence base for **C2**. Module map: `src/argus/attacks/` —
`constraints.py`, `a1_feature_pgd.py`, `a2_structural_injection.py`,
`a3_prototype_poison.py`, `a4_adaptive.py`, `a5_temporal_jitter.py`.

---

## 1. Threat model

Stated explicitly because unconstrained gradient attacks on tabular network data
produce feature vectors that no real packet sequence could generate, and results
from such attacks are meaningless. Venturi et al. (arXiv:2403.11830) make exactly
this point for GNN-NIDS; we follow their problem-space framing.

### 1.1 Attacker goal

Cause malicious flows to be classified as benign (or, weaker, as any non-alerting
class). A secondary goal is to be classified as a *known benign* class rather
than UNKNOWN — an UNKNOWN verdict still raises an alert, so it counts as partial
defence success. We measure this separately as **unknown-evasion rate**.

### 1.2 Attacker capabilities

| Capability | Granted | Rationale |
|---|---|---|
| Modify their own flows' controllable features | **Yes** | They control their own traffic generator |
| Modify their own flows' timing | **Yes, at a cost** | Changing IAT changes attack efficacy — priced in A5 |
| Emit additional flows from hosts they control | **Yes** | The core structural capability |
| Modify *other parties'* flows | **No** | Would require path compromise, a different threat model |
| Modify TTL / retransmissions / path-derived features | **No** | Requires network-path control |
| Modify the victim's node features directly | **No** | Only indirectly, via emitted flows |
| Read model weights | **A4 only** | White-box adaptive attacker |
| Poison the streaming prototype EMA update | **A3 only** | Requires flows accepted as high-confidence |
| Modify training data | **No** | Training-set poisoning is out of scope; state this |

### 1.3 Domain constraints (`constraints.py`)

Every perturbed feature vector must satisfy:

```
IN_PKTS, OUT_PKTS                integer ≥ 0
IN_BYTES  ≥ IN_PKTS  · 40        (min IPv4+TCP header)
OUT_BYTES ≥ OUT_PKTS · 40
FLOW_DURATION_MILLISECONDS ≥ 0
DURATION_IN + DURATION_OUT ≤ FLOW_DURATION_MILLISECONDS
SHORTEST_FLOW_PKT ≤ LONGEST_FLOW_PKT ≤ 1514
MIN_IP_PKT_LEN    ≤ MAX_IP_PKT_LEN
all IAT_MIN ≤ IAT_AVG ≤ IAT_MAX
IAT_STDDEV ≥ 0
Σ NUM_PKTS_* bins == IN_PKTS + OUT_PKTS
TCP flag bits ∈ {0,1}; SYN must be set if the flow is a TCP connection attempt
ports ∈ [0, 65535] integer
protocol ∈ observed vocabulary
```

Enforced by projection after every attack step:

```python
def project(x_raw: Tensor) -> Tensor:
    """Project a perturbed raw-space feature vector onto the feasible set.
    Applied after EVERY optimisation step, not just at the end."""
```

**Attacks operate in raw feature space, then pass through the frozen feature
pipeline.** Perturbing normalised features directly and inverting is wrong: the
quantile transform is not invertible outside its fitted range, and the derived
TE2 features must be *recomputed* from perturbed raw columns, not perturbed
independently. Recomputation is mandatory — otherwise the attacker gets to set
`iat_cv` and `SRC_TO_DST_IAT_STDDEV` inconsistently, which is physically
impossible and inflates attack success.

---

## 2. A1 — Constrained feature-space evasion

**Capability:** modify Channel A (attacker-controllable) features of own flows.
**Knowledge:** white-box (gradients available).

```
Input: malicious flow e, budget ε, steps N=40
x ← raw features of e
for i in 1..N:
    x_norm ← pipeline.transform(x)                # frozen, no grad through fit
    loss   ← −CE(model(graph, x_norm), benign_class)
    g      ← ∂loss/∂x   restricted to CONTROLLABLE columns
    x      ← x + (2.5ε/N) · sign(g)
    x      ← clip to ε-ball around original (in normalised space)
    x      ← recompute_derived(x)                 # TE2 recomputed, not perturbed
    x      ← project(x)                           # domain constraints
return x
```

Sweep `ε ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}` in normalised units.

**Expected result:** GNN-NIDS are already relatively resistant to this class of
attack (Venturi et al. found exactly that). ARGUS should be at least comparable.
This attack is included as a control, not as the headline.

---

## 3. A2 — Structural node/edge injection ⭐

**The headline attack.** This is the one existing GNN-NIDS fail.

**Capability:** emit additional benign-looking flows from a controlled host into
the victim's window, to dilute the victim's neighbourhood and shift the
aggregate.
**Knowledge:** grey-box (knows the architecture and window sizes; not weights).

```
Input: target malicious flow e = (u → v), budget m, spread ∈ {single_stratum, all_strata}
1. Choose an injection source host w (attacker-controlled).
2. Synthesise m flows w → v with features drawn from the benign flow
   distribution (fit a KDE on benign training flows; sample from it).
3. Place their timestamps:
     single_stratum: all within the most recent stratum before t_e  (cheap attack)
     all_strata:     spread evenly across the Q strata of each window (strong attack)
4. Rebuild the graph context for e including the injected flows.
5. Re-run the neighbour sampler and the model.
6. Record whether e is still detected.
```

Sweep `m ∈ {0, 1, 2, 4, 8, 16, 32, 64}`.

**Why `all_strata` is the strong variant.** Recency-stratified sampling
(`04_GRAPH_CONSTRUCTION.md` §4) caps any single time-interval's share at `1/Q`.
A burst confined to one stratum therefore cannot exceed 25% of the sample no
matter how large `m` is. To beat the sampler the attacker must sustain injection
across the full window — which increases both their cost and their exposure.
**Report both variants**; the gap between them is direct evidence that the
sampling strategy is a defence, not just an efficiency device.

### 3.1 The money plot (figure F6)

ASR vs injection budget `m`, one line per configuration:

- E-GraphSAGE (mean aggregation, uncapped degree) — expected to fail early
- ARGUS with `aggregation=mean`
- ARGUS with `aggregation=trimmed`, β ∈ {0.1, 0.25, 0.4}
- ARGUS with `aggregation=soft_medoid`
- ARGUS with `sampling=recent` vs `recency_stratified`

### 3.2 Empirical breakdown point

Report the budget `m*` at which robust macro-F1 falls below 90% of clean
macro-F1, per configuration, and compare against the analytic sketch in
`04_GRAPH_CONSTRUCTION.md` §4.1. **Where they disagree, trust the empirical
number and say so** — the analytic form assumes an independence that stratified
sampling does not exactly satisfy, and presenting a clean theoretical bound that
the experiment contradicts is worse than presenting no bound.

---

## 4. A3 — Prototype poisoning

Applicable only when streaming EMA prototype drift correction is enabled
(`prototype_ema_momentum` not `off`).

**Capability:** emit flows designed to be accepted as high-evidence members of a
class, thereby dragging that class's prototype.
**Knowledge:** grey-box.

```
Input: target class c, poison rate p, momentum μ
for each streaming batch:
    craft flows that (a) pass the evidence gate for class c, and
                     (b) sit at the far edge of c's cone, away from its centre
    inject at rate p of the accepted stream
measure prototype drift ||p_c(t) − p_c(0)|| and downstream F1 on class c
```

Sweep `p ∈ {0.001, 0.005, 0.01, 0.05, 0.1}`.

**Defence to demonstrate:** the evidence gate. Only flows exceeding
`θ_unknown` and the margin threshold contribute to the EMA, so poison flows must
already look strongly like class `c` — which bounds how far they can drag the
prototype. Report the drift curve with the gate on and off.

This attack salvages the memory-poisoning idea from the earlier TGN iteration of
this project (`plan/previous_work.txt`), which is a genuinely novel attack
formulation worth keeping.

If EMA drift correction is disabled in the final configuration, report A3 as a
justification for that decision rather than dropping it silently.

---

## 5. A4 — Adaptive white-box attacker

**Capability:** full knowledge of weights, prototype bank, thresholds, and the
three-way decision rule. Optimises the embedding directly.
**Knowledge:** white-box.

```
Objective: minimise  d(z_e, p_benign)   subject to domain constraints
           i.e. push the embedding into the benign cone
Steps: 100.  Restricted to CONTROLLABLE features + timing (with A5 cost accounting).
Also report a variant that targets UNKNOWN avoidance:
           maximise E_total while minimising d(z_e, p_benign)
```

This is the attack that tests whether the prototype geometry is itself a
weakness. **Expect it to be the most successful attack.** Report it honestly —
a paper claiming robustness against an adaptive white-box attacker with no
degradation is not credible.

The useful result is the *cost*: how large a constrained perturbation is required,
and what that perturbation does to the attack's own function.

---

## 6. A5 — Temporal jitter

**Capability:** perturb own inter-packet timing.
**Cost:** non-zero and quantified.

```
for each attacker-side IAT column:
    x ← x · exp(N(0, σ²))          # multiplicative log-normal jitter
recompute TE2 derived features
project onto constraints (IAT_MIN ≤ IAT_AVG ≤ IAT_MAX)
```

Sweep `σ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}`.

**Mandatory extra column: attacker cost.** At jitter `σ`, report the induced
change in effective packet rate and flow duration. A flood that must jitter to
`σ = 0.5` has materially reduced its own throughput; a scan that must jitter has
extended its own dwell time and exposure. Evasion is not free, and the paper
must show the price.

**Mandatory extra analysis: which temporal components survive.** TE7 (host-level
spectral) and TE5's long-scale gate operate on *flow arrival times at the host*,
not on intra-flow IAT. Report per-component degradation. If some components prove
fragile, report that — identifying the limits of one's own mechanism is a
strength at security venues.

---

## 7. Reporting

| Metric | Definition |
|---|---|
| **ASR** | Fraction of malicious flows evading detection after attack |
| **Robust macro-F1 @ budget** | Macro-F1 under attack at each budget level |
| **Unknown-evasion rate** | Fraction of successful evasions landing on a *known benign* class rather than UNKNOWN. Lower is better — routing evasions to UNKNOWN is partial defence success. |
| **Empirical breakdown point `m*`** | Budget at which robust macro-F1 < 90% of clean |
| **Attacker cost** | A5 only: induced degradation of the attack's own function |
| **Clean-accuracy price** | Clean macro-F1 of each defensive configuration, so the robustness–accuracy frontier is visible |

The last row matters. Robust aggregation may cost clean accuracy; the correct
presentation is a **robustness–accuracy frontier**, not a claim that robustness
is free. Plot clean macro-F1 against `m*` for every configuration — a frontier
plot is honest and is the kind of figure reviewers trust.

---

## 8. What is out of scope, and say so

| Excluded | Why |
|---|---|
| Training-data poisoning | Assumes attacker access to the labelled training corpus; different threat model |
| Model extraction / stealing | Not a detection-evasion attack |
| Physical-layer or encryption-level attacks | Below the flow abstraction |
| Attacks requiring network-path control | Would permit forging TTL/retransmission features, collapsing the provenance partition — state this as an explicit assumption boundary |
| Adversarial attacks on the explainer | Interesting, but a separate paper |

The path-control exclusion is important and must be stated plainly in the paper:
**the provenance partition's value depends on the attacker not controlling the
network path.** An attacker who does control the path can forge Channel B, and
ARGUS's C2 guarantee degrades toward the mean-aggregation baseline. Naming this
boundary is better than having a reviewer find it.
