# 03 — Feature Engineering

Implements TE1 and TE2 from `09_TEMPORAL_STUDY.md`, plus the provenance
partition that C2 depends on.

Module map: `src/argus/features/` — `conditioning.py` (TE1), `derived.py` (TE2),
`spectral.py` (TE7, applied at node level in `04_GRAPH_CONSTRUCTION.md`),
`partition.py`, `encoders.py`.

---

## 1. The fit-on-train-only rule

Every stateful transform — quantile transformer, robust scaler, category
vocabulary, port frequency table — is **fitted on the training split only** and
then applied unchanged to val and test.

Implementation contract:

```python
class FeaturePipeline:
    def fit(self, train_df) -> None: ...
    def transform(self, df) -> np.ndarray: ...
    def save(self, path) -> None: ...      # writes artifact + provenance record
    @classmethod
    def load(cls, path) -> "FeaturePipeline": ...
```

The provenance record (`data/artifacts/<run>/pipeline_provenance.json`) stores
the split hash the pipeline was fitted on. `src/argus/data/audit.py` verifies at
evaluation time that the pipeline's recorded split hash equals the training split
hash. Mismatch is a hard error.

Unseen categories at transform time map to a reserved `OTHER` index; unseen
numeric values are clipped to the fitted quantile range.

---

## 2. Column roles

| Role | Columns | Handling |
|---|---|---|
| **Structural** | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` (+ `L4_SRC_PORT`, `L4_DST_PORT` when `node_granularity: ip_port`) | Build node identities. Never enter the feature vector as identities. |
| **Temporal index** | `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS` | Window assignment and Δt computation. Never a raw feature — absolute epoch time would let the model memorise the capture schedule. |
| **Categorical** | `PROTOCOL`, `L7_PROTO` | One-hot, top-k + OTHER. |
| **Bitfield** | `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS` | Expand to 8 bits each. |
| **Port** | `L4_SRC_PORT`, `L4_DST_PORT` | Binned + log, see §5. |
| **Heavy-tailed numeric** | All byte/packet/duration/throughput/IAT columns | TE1 conditioning, §3. |
| **Bounded numeric** | `MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_*`, `ICMP_*`, `DNS_QUERY_TYPE`, `FTP_COMMAND_RET_CODE` | Robust-scale only. |
| **Label** | `Attack`, `Label` | Target. |

> **Never feed absolute timestamps as features.** In NF-v3 attacks are segregated
> by day (see `02_DATASETS.md` §4); a model given absolute time will learn "day 9
> means Backdoor" and report near-perfect, entirely fake accuracy. Only *relative*
> Δt is permitted, and only through the TE3 encoder.

---

## 3. TE1 — heavy-tail conditioning

**Motivation.** IAT and byte/packet counts span six or more orders of magnitude
(sub-millisecond IAT up to multi-second, single bytes up to gigabytes). Fed raw
or min-max scaled, almost all mass collapses to ~0 and the feature contributes
nothing to gradient flow. This is the most likely reason temporal features have
previously appeared unhelpful.

**Transform**, applied in order to every heavy-tailed numeric column:

```
1. clip:      x ← max(x, 0)                    # nProbe emits rare negatives
2. log:       x ← sign(x) · log1p(|x|)         # signed log; safe at 0
3. quantile:  x ← QuantileTransformer(
                     output_distribution="normal",
                     n_quantiles=1000,
                     subsample=1_000_000
                  ).fit(train).transform(x)
4. clip:      x ← clip(x, -5, +5)              # bound outliers post-transform
```

Columns subject to TE1 (23):

```
IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS,
FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT,
SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES,
SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT,
RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS,
RETRANSMITTED_OUT_BYTES, RETRANSMITTED_OUT_PKTS,
SRC_TO_DST_IAT_MIN, SRC_TO_DST_IAT_MAX, SRC_TO_DST_IAT_AVG, SRC_TO_DST_IAT_STDDEV,
DST_TO_SRC_IAT_MIN, DST_TO_SRC_IAT_MAX, DST_TO_SRC_IAT_AVG, DST_TO_SRC_IAT_STDDEV
```

Packet-size histogram bins (`NUM_PKTS_*`) get log1p but **not** quantile
transform — they are counts on a common scale and their ratios are meaningful.
Normalise them instead to a simplex: divide each by the row sum plus ε, and keep
`log1p(row_sum)` as one extra feature.

Length-like columns (`LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`,
`MAX_IP_PKT_LEN`) are bounded by MTU; robust-scale only.

**Ablation hook.** TE1 must be switchable (`features.te1_enabled: bool`) so rung
L1 vs L2 of the temporal ladder can be run.

---

## 4. TE2 — derived rhythm descriptors

Twelve features, each a pure function of existing columns. Zero extra data cost.
Computed **before** TE1 conditioning, then themselves passed through TE1.

Let `eps = 1e-6`.

| # | Name | Formula | Interpretation |
|---|---|---|---|
| 1 | `iat_cv_fwd` | `SRC_TO_DST_IAT_STDDEV / (SRC_TO_DST_IAT_AVG + eps)` | Coefficient of variation, forward. **→0 = machine-regular** (DoS, DDoS, scanning); high = human/bursty. Expected top-ranked temporal feature. |
| 2 | `iat_cv_bwd` | `DST_TO_SRC_IAT_STDDEV / (DST_TO_SRC_IAT_AVG + eps)` | Same, reverse direction. |
| 3 | `iat_burst_fwd` | `(SRC_TO_DST_IAT_MAX - SRC_TO_DST_IAT_MIN) / (SRC_TO_DST_IAT_AVG + eps)` | Burstiness / spread of the forward rhythm. |
| 4 | `iat_burst_bwd` | `(DST_TO_SRC_IAT_MAX - DST_TO_SRC_IAT_MIN) / (DST_TO_SRC_IAT_AVG + eps)` | Reverse burstiness. |
| 5 | `duty_cycle` | `(DURATION_IN + DURATION_OUT) / (FLOW_DURATION_MILLISECONDS + eps)` | Fraction of flow lifetime actively transmitting. **→0 = low-and-slow** (Infiltration, Backdoor). |
| 6 | `pkt_rate` | `(IN_PKTS + OUT_PKTS) / (FLOW_DURATION_MILLISECONDS + eps)` | Packets per ms. |
| 7 | `byte_rate` | `(IN_BYTES + OUT_BYTES) / (FLOW_DURATION_MILLISECONDS + eps)` | Bytes per ms. |
| 8 | `dir_asymmetry` | `(SRC_TO_DST_IAT_AVG - DST_TO_SRC_IAT_AVG) / (SRC_TO_DST_IAT_AVG + DST_TO_SRC_IAT_AVG + eps)` | ∈[-1,1]. Near 0 = request/response; near ±1 = one-way flood. |
| 9 | `pkt_size_spread` | `LONGEST_FLOW_PKT - SHORTEST_FLOW_PKT` | Uniform packet size (→0) indicates generated traffic. |
| 10 | `bytes_per_pkt_in` | `IN_BYTES / (IN_PKTS + eps)` | Mean inbound packet size. |
| 11 | `bytes_per_pkt_out` | `OUT_BYTES / (OUT_PKTS + eps)` | Mean outbound packet size. |
| 12 | `retrans_ratio` | `(RETRANSMITTED_IN_PKTS + RETRANSMITTED_OUT_PKTS) / (IN_PKTS + OUT_PKTS + eps)` | Retransmission fraction — congestion or spoofing signal. |

**Guard rails.** Where a denominator column is exactly 0 for an entire flow
(single-packet flows have undefined IAT), nProbe emits 0. Emit an accompanying
binary mask feature `iat_undefined` set when `IN_PKTS + OUT_PKTS <= 2`, so the
model can distinguish "rhythm is zero" from "rhythm is undefined". This is
feature 13; do not skip it — without it, single-packet scan flows and perfectly
regular flood flows become indistinguishable.

**Why these matter for C2.** Every TE2 feature is a function of *timing*, and
timing is expensive for an attacker to forge: slowing a scan to look benign makes
the scan slower, and adding jitter to a flood reduces its throughput. This is why
they sit in the observer-derived channel (§6) and why A5 (temporal jitter attack)
is a *costly* attack rather than a free one.

---

## 5. Categorical, bitfield, and port encoding

**`PROTOCOL`** — one-hot over the top 8 values by training frequency, plus
`OTHER`. Dimension 9. (In practice 6/17/1 = TCP/UDP/ICMP dominate.)

**`L7_PROTO`** — one-hot over the top 16 by training frequency, plus `OTHER`.
Dimension 17. Store the vocabulary in the fitted artifact; it differs per dataset
and must be re-fitted (not shared) for cross-dataset transfer — for Protocol C,
apply the **source** dataset's vocabulary and let unseen protocols map to
`OTHER`. Record the `OTHER` rate; a high rate is itself a drift signal worth
reporting.

**TCP flag bitfields** — expand each of `TCP_FLAGS`, `CLIENT_TCP_FLAGS`,
`SERVER_TCP_FLAGS` to 8 binary features (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR).
Dimension 24. Keep the raw integer out of the vector.

**Ports** — do not one-hot 65,536 values. Per direction (src, dst), emit 4
features:
```
is_well_known   = 1 if port < 1024
is_registered   = 1 if 1024 <= port < 49152
is_ephemeral    = 1 if port >= 49152
log_port        = log1p(port) / log1p(65535)
```
Plus, for the destination port only, a one-hot over the **top 32 destination
ports by training frequency** plus `OTHER` (dimension 33). Source ports are
usually ephemeral and carry little signal; destination ports carry service
identity.

Port dimension total: 4 (src) + 4 (dst) + 33 (dst top-k) = 41.

---

## 6. Provenance partition — the C2 mechanism

Every edge feature is assigned to exactly one channel.

### 6.1 Channel A — attacker-controllable

Features an attacker can set freely, at negligible cost, without reducing attack
efficacy:

```
IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS
TCP_FLAGS bits, CLIENT_TCP_FLAGS bits, SERVER_TCP_FLAGS bits
L4_SRC_PORT features, L4_DST_PORT features (incl. top-k one-hot)
PROTOCOL one-hot, L7_PROTO one-hot
NUM_PKTS_* histogram bins (+ normalised simplex + log row sum)
LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN
TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT
DNS_QUERY_ID, DNS_QUERY_TYPE, ICMP_TYPE, ICMP_IPV4_TYPE, FTP_COMMAND_RET_CODE
pkt_size_spread, bytes_per_pkt_in, bytes_per_pkt_out
```

### 6.2 Channel B — observer-derived / costly to forge

Features whose manipulation either requires control of the network path, or
degrades the attacker's own objective:

```
All 8 IAT statistics
FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT
SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES
SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT
MIN_TTL, MAX_TTL                       # path-derived; forging requires path control
RETRANSMITTED_* (4)                    # emergent from network conditions
DNS_TTL_ANSWER                         # set by the resolver, not the client
iat_cv_fwd, iat_cv_bwd, iat_burst_fwd, iat_burst_bwd,
duty_cycle, pkt_rate, byte_rate, dir_asymmetry, retrans_ratio, iat_undefined
```

The partition is a static table in `src/argus/features/partition.py`:

```python
CONTROLLABLE: frozenset[str] = frozenset({...})
OBSERVER:     frozenset[str] = frozenset({...})

def assert_partition_complete(feature_names: list[str]) -> None:
    """Every emitted feature is in exactly one channel. Hard-fail otherwise."""
```

Call `assert_partition_complete` at pipeline construction. A feature silently
missing from both channels would be dropped from the model without warning.

### 6.3 How the partition is used

1. **Two message channels.** The encoder projects Channel A and Channel B
   separately before fusion (see `05_ARCHITECTURE.md` §3.2).
2. **Gradient penalty.** A loss term penalises the ratio of gradient norm on
   Channel A to total gradient norm, discouraging sole reliance on the
   easily-forged channel (see `06_TRAINING.md` §2.4).
3. **Attack surface definition.** A1 (feature-space PGD) may perturb **only**
   Channel A. A5 (temporal jitter) perturbs Channel B and is reported as a
   *costly* attack. See `10_ADVERSARIAL.md`.

---

## 7. Final edge feature vector

| Block | Dim |
|---|---:|
| TE1-conditioned heavy-tail numeric (23) | 23 |
| Bounded numeric, robust-scaled (`MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_IN/OUT`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE`, 4 length columns) | 14 |
| Packet-size histogram: 5 normalised + 1 log row-sum | 6 |
| TCP flag bits (3 × 8) | 24 |
| `PROTOCOL` one-hot | 9 |
| `L7_PROTO` one-hot | 17 |
| Port features | 41 |
| TE2 derived (12 + `iat_undefined`) | 13 |
| **Total `F_e`** | **147** |

The loader emits the realised dimension and writes it to
`data/artifacts/<run>/feature_manifest.json` together with the ordered feature
name list and the channel assignment of each. **All downstream code reads
`F_e` from that manifest — never hard-code 147.**

Ablation rungs change `F_e`:

| Rung | Included | Approx. `F_e` |
|---|---|---:|
| L0 | v2 features only, no temporal, no TE2 | 111 |
| L1 | + raw 10 temporal, no TE1, no TE2 | 119 |
| L2 | + TE1 conditioning | 119 |
| L3 | + TE2 derived | 147 |
| L4–L7 | unchanged `F_e`; changes are architectural | 147 |
