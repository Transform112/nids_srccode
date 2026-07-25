# 04 — Graph Construction

Module map: `src/argus/graph/` — `windows.py`, `builder.py`, `sampler.py`,
`batching.py`. Node spectral features come from `src/argus/features/spectral.py`.

---

## 1. Graph definition

A graph is built **per window per scale**. It is a directed multigraph.

- **Nodes** `V` — hosts. Identity is `IPV4_*_ADDR`, or `(IPV4_*_ADDR, L4_*_PORT)`
  when `node_granularity: ip_port`. See `02_DATASETS.md` §3.1 for the mandatory
  per-dataset policy.
- **Edges** `E` — flows. One flow record = one directed edge
  `src_host → dst_host`, carrying the `F_e`-dimensional feature vector from
  `03_FEATURE_ENGINEERING.md` §7, its arrival time `t_e = FLOW_START_MILLISECONDS`,
  and its label.

**The classification target is the edge, not the node.** ARGUS classifies flows.
Node embeddings exist only to provide context to the edges incident on them.

---

## 2. Multi-scale windowing (TE5)

Three concurrent scales, all anchored at the same instant:

| Scale | Symbol | Duration | Captures |
|---|---|---:|---|
| Short | `S` | 1 s | DDoS, DoS, port/host scanning — burst-rate phenomena |
| Mid | `M` | 30 s | BruteForce, Password, Injection, XSS — session-rate phenomena |
| Long | `L` | 300 s | Infiltration, Backdoor, Ransomware, MITM — low-and-slow, beaconing |

For a flow `e` with time `t_e`, its three context windows are
`[t_e − D_s, t_e]` for `D_s ∈ {1, 30, 300}` seconds. Windows are **backward-
looking only** — a flow is never contextualised by future flows. This is a
deployment-faithfulness requirement, not merely an anti-leakage one: a streaming
detector cannot see the future.

Note that `D_L = 300 s` exceeds nProbe's 120 s flow-record cap, so the long
window deliberately spans multiple records of the same underlying session. State
this in the paper.

### 2.1 Window realisation

For efficiency, do not materialise a distinct graph per flow. Instead:

```
partition the split's time range into non-overlapping ANCHOR BINS of 1 s
for each anchor bin b:
    for each scale s in {S, M, L}:
        G[b][s] ← graph over all flows with t_e in [end(b) − D_s, end(b)]
    all flows whose t_e falls in b are the PREDICTION TARGETS for bin b
```

Each target flow is thus embedded using three graphs, all of which end at its
anchor bin. Flows in the same 1 s anchor bin share graphs — an approximation
worth ~1 s of temporal resolution, and a large constant-factor saving.

Bins with zero target flows are skipped. Bins are processed in chronological
order so that the per-node GRU memory (TE6) evolves correctly.

### 2.2 Sliding-window state

Maintain three ring buffers, one per scale, each holding the flows currently
inside that scale's window. On advancing one anchor bin: append the new bin's
flows, evict flows older than `D_s`. Cost is O(flows entering + flows leaving),
not O(window size). Peak memory is bounded by the number of flows in a 300 s
window — record this in the deployment measurement.

---

## 3. Node features

`F_v = 18` dimensions per node, all computed **within the current window** so the
representation is inductive (new hosts get valid features immediately, with no
lookup table and no transductive node embeddings). Two blocks.

### 3.1 Block 1 — observer-derived local statistics (12)

Computed for node `v` over window scale `L` (300 s) unless stated:

| # | Feature | Definition |
|---|---|---|
| 1 | `out_degree_capped` | `min(#outgoing flows, K)` normalised by `K` |
| 2 | `in_degree_capped` | `min(#incoming flows, K)` normalised by `K` |
| 3 | `distinct_peers` | `log1p(#distinct peer hosts)` normalised |
| 4 | `distinct_dst_ports` | `log1p(#distinct destination ports contacted)` normalised |
| 5 | `fanout_ratio` | `distinct_peers / (out_degree + eps)` — scanning signature |
| 6 | `flow_rate` | `log1p(#flows / D_L)` |
| 7 | `mean_log_gap` | mean of `log1p(Δt)` between consecutive flows at `v` |
| 8 | `std_log_gap` | std. dev. of the same — regularity of arrival |
| 9 | `byte_volume` | `log1p(total bytes across incident flows)`, normalised |
| 10 | `short_scale_burst` | `#flows in scale S / (#flows in scale M / 30 + eps)` — burst factor |
| 11 | `reverse_ratio` | `in_degree / (in_degree + out_degree + eps)` — client vs server role |
| 12 | `is_new_host` | 1 if `v` was unseen in the previous window |

**Why not E-GraphSAGE's constant node vector.** E-GraphSAGE initialises every
node identically, so all host context must be reconstructed by message passing.
That wastes capacity and, more importantly for C2, means the *only* node signal
is the aggregate of neighbour edges — precisely the quantity an attacker
manipulates. Giving nodes their own observer-derived statistics creates a signal
path an edge-injecting attacker does not fully control.

Every Block-1 feature is degree-capped or log-compressed so a flooding attacker
cannot drive any of them arbitrarily.

### 3.2 Block 2 — TE7 spectral beaconing descriptor (6)

Computed for node `v` over the long window `D_L = 300 s`:

```
1. Collect arrival times of all flows incident on v within the window.
2. Bin into NBINS = 64 equal bins  →  counts c[0..63]
3. Detrend:  c ← c − mean(c)
4. P ← |rfft(c)|²                     # power spectrum, length 33
5. Discard the DC term P[0]           # removed by detrending anyway
6. Emit 6 scalars from P[1..32]:
```

| # | Feature | Formula |
|---|---|---|
| 1 | `dominant_freq` | `argmax(P) / 32` — normalised dominant frequency |
| 2 | `dominant_power_ratio` | `max(P) / (sum(P) + eps)` — how concentrated the spectrum is |
| 3 | `spectral_entropy` | `−Σ p_i log p_i / log(32)` where `p = P / sum(P)` — low = strongly periodic |
| 4 | `spectral_flatness` | `geometric_mean(P) / (arithmetic_mean(P) + eps)` — Wiener entropy; →0 = tonal/beaconing, →1 = noise |
| 5 | `peak_to_mean` | `max(P) / (mean(P) + eps)` |
| 6 | `low_freq_energy` | `sum(P[1:5]) / (sum(P) + eps)` — slow periodicity (long-interval beacons) |

Cost: one 64-point real FFT per active node per long window. Negligible.

Nodes with fewer than 8 flows in the window get a zero vector plus an implicit
signal via `is_new_host` — do not attempt to compute a spectrum from 3 points.

**Novelty note.** This is the direct follow-up on the open problem the NF-v3
dataset authors left (`01_RELATED_WORK.md` §6). They observed distinct
time-frequency signatures per attack class but did not operationalise them.
Expected wins: `Backdoor`, `MITM`, `BoT`, `Ransomware` — all beaconing/C2
behaviours invisible to intra-flow IAT statistics.

**Ablation hook.** `features.te7_enabled: bool`; when false, `F_v = 12`.

---

## 4. Degree-capped neighbour sampling

The cap `K = 32` is simultaneously the efficiency mechanism and the robustness
mechanism. It is what makes the breakdown-point argument in C2 well-defined.

```python
def sample_neighbours(v, window, K=32, strategy="recency_stratified", rng=None):
    """
    Returns at most K incident edges of v within the window.

    strategy:
      "recent"              take the K most recent incident edges
      "uniform"             uniform sample without replacement
      "recency_stratified"  DEFAULT. Split the window into 4 equal sub-intervals;
                            take K/4 from each, drawing uniformly within the
                            sub-interval; if a sub-interval is short, redistribute
                            its quota to the most recent sub-interval.
    """
```

**Why `recency_stratified` is the default.** Pure `recent` is trivially
attackable: an adversary emits `K` flows immediately before the victim's flow and
owns 100% of the neighbourhood regardless of the robust aggregator. Pure
`uniform` discards the recency signal the temporal design depends on. Stratifying
by recency bounds any single time-interval's share of the sample at 1/4, so a
burst-injection attacker must sustain the injection across the whole window —
raising cost and increasing their own detectability.

Sampling is **deterministic given the run seed** so evaluation is reproducible.
Store the RNG state per anchor bin.

### 4.1 Breakdown point

With cap `K`, trim fraction `β` (see `05_ARCHITECTURE.md` §3.3), and
recency-stratified sampling over `Q = 4` strata, an adversary who injects `m`
flows into a victim's window obtains an expected sampled share of

```
share(m) ≈ min(1, m / (m + n_benign)) , capped at 1/Q per stratum if the
injection is confined to one stratum
```

and the trimmed-mean aggregate is unmoved while `share < β`. With the defaults
`K = 32`, `β = 0.25`, `Q = 4`, the attacker must contribute more than 25% of the
sampled neighbourhood, spread across all four strata, i.e. sustain injection for
the full window duration. **Report the empirical version of this curve** (A2 in
`10_ADVERSARIAL.md`) rather than relying on the analytic sketch — the analytic
form assumes independence that stratified sampling does not exactly satisfy.

---

## 5. Batching

Anchor bins are the unit of batching. A batch is `B = 64` consecutive anchor bins
(default; tune to GPU memory).

Because the per-node GRU memory (TE6) is sequential across bins, batching is over
*bins within a contiguous chronological chunk*, and the memory is carried forward
across batches within a chunk and **detached** at chunk boundaries (truncated BPTT
with chunk length `T_bptt = 8` bins). Reset memory to zeros at the start of each
split.

For each batch emit a `torch_geometric.data.HeteroData`-like structure, or a plain
dict:

```python
{
  "scale_S": {"edge_index": LongTensor[2, E_s],
              "edge_attr":  FloatTensor[E_s, F_e],
              "edge_time":  FloatTensor[E_s],        # ms, absolute
              "node_feat":  FloatTensor[N, F_v],
              "node_id":    LongTensor[N]},          # global host index
  "scale_M": {...},
  "scale_L": {...},
  "target_edge_idx": LongTensor[T],                  # indices into scale_S edges
  "target_label":    LongTensor[T],
  "target_time":     FloatTensor[T],
  "anchor_bin_id":   LongTensor[B],
}
```

Target flows are always present in the short-scale graph (a flow is inside its own
1 s window by construction), so `target_edge_idx` indexes `scale_S`.

Node index spaces are unified across the three scales within a batch — the same
host has the same local index in `scale_S`, `scale_M`, and `scale_L`. Build the
union node set first, then index all three.

---

## 6. Streaming inference contract

Deployment measurement (`08_EVALUATION.md` §4.6) requires the same code path as
training, driven one anchor bin at a time.

```python
class StreamingDetector:
    def __init__(self, model, pipeline, config): ...

    def push(self, flows: list[FlowRecord]) -> list[Verdict]:
        """
        Ingest flows for one anchor bin. Returns one Verdict per flow:
          Verdict = (decision, class_or_none, evidence, uncertainty, latency_ms)
          decision ∈ {CLASSIFY, DEFER, UNKNOWN}
        Must be O(new flows + evicted flows), not O(window size).
        """

    def register_class(self, name: str, samples: list[FlowRecord]) -> None:
        """
        Few-shot registration. Computes the mean unit-norm embedding of samples
        and appends it to the prototype bank. NO GRADIENT STEPS. O(n).
        """
```

`register_class` is the operational embodiment of C1. It must be genuinely
gradient-free — assert `torch.is_grad_enabled() == False` inside it and verify in
`tests/test_epc.py` that no model parameter changes value across a registration.

Constraints to enforce and measure:
- Memory bounded by the 300 s window contents plus the prototype bank plus one
  GRU state per *active* node (evict node memory after 2·`D_L` of inactivity).
- No lookahead: `push` may only use flows already pushed.
- Latency measured per flow, reported as p50/p95/p99.

---

## 7. Validation tests

`tests/test_graph.py` must assert:

1. No edge in any window has `t_e` greater than the window end (no lookahead).
2. Every target flow appears in its own short-scale graph.
3. Node index spaces are consistent across the three scales in a batch.
4. `sample_neighbours` never returns more than `K` edges.
5. `recency_stratified` returns at most `ceil(K/Q)` edges from any single
   stratum when all strata are populated.
6. Windows never span a train/val/test boundary.
7. Given a fixed seed, two builds of the same bin are byte-identical.
