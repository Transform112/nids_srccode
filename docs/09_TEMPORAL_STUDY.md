# 09 — Temporal Study

The evidence base for **C3**. This document exists because "we added temporal
features" is not a contribution; "we measured exactly which temporal mechanism
rescues which attack class, and priced it" is.

---

## 1. Why temporal modelling is under-exploited in GNN-NIDS

Four levels of temporal signal exist in NF-v3 data:

| Level | Horizon | Signal | Carried by |
|---|---|---|---|
| **T1 intra-flow** | sub-second | Packet rhythm inside one flow: burst vs paced, jitter, duty cycle | The 8 IAT columns + durations |
| **T2 inter-flow** | seconds–minutes | Ordering and spacing of flows; burst structure, fan-out rate, periodicity | Window membership + Δt |
| **T3 host evolution** | minutes–hours | How a host's behaviour profile drifts | Per-node memory |
| **T4 dataset drift** | days | Campaign-level change; the NF-v3 day structure | Split protocol |

**Existing GNN-NIDS use only T2, and only implicitly** — a flow is either in the
window or not. T1 is fed as unconditioned scalars that vanish under scaling, T3
is absent or handled by an expensive continuous-time memory, T4 is destroyed by
random splits.

The closest competitor (Dai et al., arXiv:2606.17109) uses real timestamps for
*ordering* at a single scale. Ordering is not rhythm: it cannot represent "this
host emits a flow every 30 seconds", which is what beaconing looks like.

---

## 2. The seven techniques

Each is independently switchable via `model.teN_enabled` so the ladder in §3 can
be run.

### TE1 — Heavy-tail conditioning
*Where:* `src/argus/features/conditioning.py`. Spec in `03_FEATURE_ENGINEERING.md` §3.

IAT and volume columns span 6+ orders of magnitude. Signed-log then
quantile-to-normal, fitted on train only. **Most likely explanation for temporal
features previously appearing useless**: without this, nearly all mass collapses
to ~0 and the columns contribute no usable gradient. Near-zero cost.

*Expected effect:* large jump at rung L2, broadly across classes.

### TE2 — Derived rhythm descriptors
*Where:* `src/argus/features/derived.py`. Thirteen features, spec in
`03_FEATURE_ENGINEERING.md` §4.

The central one is `iat_cv = IAT_STDDEV / (IAT_AVG + ε)`. Coefficient of
variation separates *machine-regular* traffic (CV → 0: DoS, DDoS, scanning) from
human/application traffic (CV high). `duty_cycle` separates low-and-slow
(→ 0: Infiltration, Backdoor) from active sessions.

*Bonus, and worth stating in the paper:* every TE2 feature is a timing function,
and timing is **costly for an attacker to forge** — slowing a scan to look benign
makes the scan slower; jittering a flood reduces its throughput. This is why they
sit in the observer-derived channel and why A5 is a *costly* attack.

*Expected effect:* large gain on DoS/DDoS/Scanning (via `iat_cv`) and on
Infiltration/Backdoor (via `duty_cycle`).

### TE3 — Functional Δt encoding
*Where:* `src/argus/models/time_encoding.py`. Spec in `05_ARCHITECTURE.md` §2.

Time2Vec: one linear term plus `d_t − 1` learnable sinusoids. Raw Δt conveys only
recency; a sinusoidal basis can represent **periodicity**. Initialise `ω` on a
log-uniform period grid from 0.1 s to 600 s — random init makes periodicity very
slow to discover.

*This is the key upgrade over feeding temporal features as plain scalars.*

*Expected effect:* gain on beaconing/periodic classes — Backdoor, MITM, BoT.

### TE4 — Time-decayed attention
*Where:* `src/argus/models/attention.py`. Spec in `05_ARCHITECTURE.md` §3.4.

Additive log-space decay bias `−λ^{(h)}Δt` inside the attention softmax, `λ`
learnable per head, heads initialised to different half-lives. Dual purpose:
time-aware receptive field, and down-weighting of stale injected edges — so it is
simultaneously a C3 and a C2 mechanism.

*Expected effect:* modest accuracy gain; measurable robustness gain under A2.
**Report its A2 effect as well as its F1 effect** — that dual reporting is part
of the argument that C2 and C3 reinforce each other.

### TE5 — Multi-scale windows
*Where:* `src/argus/models/multiscale.py`. Spec in `05_ARCHITECTURE.md` §5.

Three concurrent scales (1 s / 30 s / 300 s), weight-shared encoder, learned gate
fusion. Every existing GNN-NIDS uses a single fixed window, which forces a choice
between catching bursts and catching slow campaigns.

The gate `g` is an interpretability output: log its mean per attack class and
plot as figure F5. Expect DDoS → `g_S`, BruteForce → `g_M`, Infiltration → `g_L`.
**If that pattern appears, it is the single most convincing figure for C3.**

*Expected effect:* large gain on the extremes (fast and slow classes); little on
mid-rate classes. Cost ~3× encoder compute, mitigated by weight sharing.

### TE6 — Per-node temporal memory
*Where:* `src/argus/models/memory.py`. Spec in `05_ARCHITECTURE.md` §4.

One shared GRU cell, one update per anchor bin, evicted after 600 s of node
inactivity. Deliberately not a continuous-time TGN.

*Expected effect:* gain on multi-stage attacks where a host's history matters
(Infiltration, Backdoor, Ransomware). Modest elsewhere.

### TE7 — Per-host spectral beaconing descriptor
*Where:* `src/argus/features/spectral.py`. Spec in `04_GRAPH_CONSTRUCTION.md` §3.2.

64-bin FFT of a host's flow-arrival counts over the long window, reduced to six
scalars: dominant frequency, dominant power ratio, spectral entropy, spectral
flatness, peak-to-mean, low-frequency energy. Cost is one 64-point real FFT per
active host per long window — negligible.

**Novelty hook.** The NF-v3 dataset authors applied spectrograms to their own
data, observed distinct per-class time-frequency signatures, but reported their
"initial investigations have not yet yielded definitive results" and left
refinement to future work (`01_RELATED_WORK.md` §6). TE7 is a direct, citable
follow-up on an open problem left by the dataset authors.

Also highly interpretable: "this host emits a flow every 30 s" is a statement an
analyst can act on.

*Expected effect:* gain concentrated on C2/beaconing classes — Backdoor, MITM,
BoT, Ransomware.

---

## 3. The L0 → L7 ablation ladder

Cumulative rungs. Same splits, same 5 seeds, same everything else.

| Rung | Adds | `F_e` | Architectural change |
|---|---|---:|---|
| **L0** | 43 NF-v2 features only. No temporal columns, no TE2. | 111 | Single scale (mid), no Δt, no decay, no memory, no spectral |
| **L1** | + the 10 NF-v3 temporal columns, raw scalars | 119 | none |
| **L2** | + TE1 heavy-tail conditioning | 119 | none |
| **L3** | + TE2 derived rhythm descriptors | 147 | none |
| **L4** | + TE3 Time2Vec Δt encoding | 147 | time encoding into messages |
| **L5** | + TE4 time-decayed attention | 147 | decay bias in attention |
| **L6** | + TE5 multi-scale windows + TE6 node memory | 147 | 3 scales, gate fusion, GRU |
| **L7** | + TE7 spectral descriptor | 147 (`F_v` 12→18) | node features extended |

L7 = full ARGUS.

### 3.1 The two rungs that matter most

**L0 → L2.** This isolates *what NF-v3's temporal features are worth to a graph
model, once properly conditioned*. Nobody has published this. NF-v3 was released
in 2025 specifically to add temporal features, and its own authors did not
evaluate a model on them — their paper is an analysis, not a modelling paper.
**This is a publishable result on its own** and should be stated as a named
finding, not buried in an ablation table.

Note that L1 → L2 may show that raw temporal columns (L1) barely help while
conditioned ones (L2) help substantially. If so, say so plainly: it explains why
the field has under-used these features and is a useful negative-to-positive
result.

**L6.** The multi-scale rung, where the gate-by-class figure comes from.

### 3.2 Reporting rules

1. **Per-class F1 deltas, not just macro.** Build a heatmap: rows = rungs,
   columns = attack classes, cells = ΔF1 vs the previous rung (figure F10). The
   story is *which* classes each mechanism rescues.
2. **Attribution.** For each rung, name the classes it improved and state the
   mechanistic reason. If the observed pattern contradicts the expectation in §2,
   **report the contradiction** — an honest surprise is more valuable than a
   massaged confirmation.
3. **Cost.** Report latency, throughput, and parameter count per rung. A gain
   that costs 3× throughput must be priced.
4. **Feature-group importance.** Permutation importance and SHAP aggregated over
   feature *groups* (temporal vs volumetric vs categorical vs structural), not
   individual features. Report at L3 and L7.
5. **Seeds.** 5 seeds per rung; report mean ± std. Some rungs will be within
   noise of each other — say so.

### 3.3 Expected outcome table (hypotheses, to be confirmed or refuted)

| Rung | Expected primary beneficiaries | Mechanism |
|---|---|---|
| L2 | Broad, all classes | Temporal columns become numerically usable |
| L3 | DoS, DDoS, Scanning | `iat_cv` → 0 marks machine-generated regularity |
| L3 | Infiltration, Backdoor | `duty_cycle` → 0 marks low-and-slow |
| L4 | Backdoor, MITM, BoT | Periodicity becomes representable |
| L5 | Modest F1; notable A2 robustness | Stale/injected edges down-weighted |
| L6 | DDoS (short), Infiltration/Ransomware (long) | Scale-matched receptive field |
| L7 | Backdoor, MITM, BoT, Ransomware | Beaconing spectrum |

These are hypotheses. Record them **before** running, then report agreement and
disagreement. Pre-registering the expectation makes the result far more credible
than post-hoc rationalisation.

---

## 4. Temporal robustness — the honest counter-analysis

An obvious reviewer objection: *if the model leans on timing, can an attacker
just change their timing?*

**A5 — temporal jitter attack** (`10_ADVERSARIAL.md`). Apply multiplicative
log-normal jitter of scale `σ` to the attacker-side IAT columns, sweeping
`σ ∈ {0.05 … 1.0}`, and measure attack success rate.

Report both halves of the answer:

1. **The cost to the attacker.** Jitter changes the attack's own behaviour.
   Quantify it: at jitter `σ`, the effective packet rate falls by a computable
   factor. A DDoS that must jitter to `σ = 0.5` has materially reduced its own
   throughput. Include this as a column in the A5 table — *evasion is not free*.
2. **The residual signal.** TE7's spectral descriptor and TE5's long-scale gate
   operate on *flow arrival times at the host*, not on intra-flow IAT. Jittering
   intra-flow IAT does not remove host-level periodicity. Report which temporal
   components survive A5 and which do not.

If some temporal components prove fragile under A5, **report that**. A paper that
identifies the limits of its own mechanism is stronger than one that claims
uniform robustness, and reviewers of security venues specifically look for this.

---

## 5. Deliverables

| Artifact | Content |
|---|---|
| **T6** | Ladder table: macro-F1 mean ± std, Δ vs previous rung, latency, params |
| **F10** | Per-class ΔF1 heatmap across rungs |
| **F5** | Multi-scale gate `g` mean by attack class |
| Appendix | Full per-class F1 at every rung |
| Appendix | Feature-group permutation importance at L3 and L7 |
| Appendix | A5 results with attacker-cost column |
| `results/tables/temporal_ladder.csv` | Machine-readable ladder output |
| `results/runs/registry.jsonl` | Run provenance for every rung × seed |

Runner: `scripts/09_run_temporal_ladder.py`, driven by
`config/experiment/temporal_ladder.yaml`, which enumerates the 8 rungs × 5 seeds
= 40 runs and skips any already present in the registry.
