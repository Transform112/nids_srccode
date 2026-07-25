# 02 — Datasets

**Read this document completely before writing any data code.** Sections 3, 4 and
7 describe failure modes that silently invalidate results.

All statistics in this document were **measured directly from the CSVs in
`dataset/`** on 2026-07-25, not copied from the source paper. Where they differ
from the paper, the measured values are authoritative and the differences are
flagged — there are four, and two of them change design decisions.

Source of the data: Luay, Layeghy, Hosseininoorbin, Sarhan, Moustafa, Portmann,
*"Temporal Analysis of NetFlow Datasets for Network Intrusion Detection
Systems"*, arXiv:2503.04404v2, DOI 10.1109/ACCESS.2026.3688204.

## 0. Files on disk

| File | Size | Rows | Role |
|---|---:|---:|---|
| `dataset/NF-CICIDS2018-v3.csv` | 4.03 GB | 20,115,529 | **PRIMARY** |
| `dataset/NF-ToN-IoT-v3.csv` | 5.06 GB | 27,520,260 | **SECONDARY** |
| `dataset/NF-UNSW-NB15-v3.csv` | 0.55 GB | 2,365,424 | Tertiary (IP:port nodes) |
| `dataset/NF-BoT-IoT-v3.csv` | 3.64 GB | 16,933,808 | Degenerate case only |
| `dataset/NetFlow_v3_Features.csv` | 4 KB | 53 | Feature dictionary |

Column order on disk differs from the paper's table: the two timestamps come
**first**, and the layout is
`FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS, IPV4_SRC_ADDR, L4_SRC_PORT,
IPV4_DST_ADDR, L4_DST_PORT, …, Label, Attack` — 55 columns total. All four files
share this header exactly. Never index columns positionally; always by name.

---

## 1. Feature schema — 53 columns

NF-v3 = the 43 NF-v2 features **plus 10 temporal features**. All four datasets
share this schema exactly, which is what makes cross-dataset transfer possible
without any feature alignment.

### 1.1 The 10 temporal features (the NF-v3 addition)

Two groups: *flow timing* and *inter-packet arrival time (IAT)*.

| # | Column | Description | Unit |
|---|---|---|---|
| 1 | `FLOW_START_MILLISECONDS` | Flow start, Unix epoch | ms |
| 2 | `FLOW_END_MILLISECONDS` | Flow end, Unix epoch | ms |
| 3 | `SRC_TO_DST_IAT_MIN` | Minimum inter-packet arrival time, src→dst | ms |
| 4 | `SRC_TO_DST_IAT_MAX` | Maximum IAT, src→dst | ms |
| 5 | `SRC_TO_DST_IAT_AVG` | Mean IAT, src→dst | ms |
| 6 | `SRC_TO_DST_IAT_STDDEV` | Std. dev. of IAT, src→dst | ms |
| 7 | `DST_TO_SRC_IAT_MIN` | Minimum IAT, dst→src | ms |
| 8 | `DST_TO_SRC_IAT_MAX` | Maximum IAT, dst→src | ms |
| 9 | `DST_TO_SRC_IAT_AVG` | Mean IAT, dst→src | ms |
| 10 | `DST_TO_SRC_IAT_STDDEV` | Std. dev. of IAT, dst→src | ms |

Two facts that matter for implementation:

- Timestamps are the **original pcap timestamps**. The authors ran nProbe with
  `--dont-reforge-time`, so chronological ordering is faithful and safe to sort
  on. Do not attempt to reconstruct time from any other column.
- **nProbe exports flow records at intervals not exceeding 120 s by default.**
  A long-lived session therefore appears as multiple flow records. This is why
  the long window (300 s) spans multiple records of the same session — that is
  intentional, and should be stated in the paper.

### 1.2 The 43 NF-v2 features

| Column | Description |
|---|---|
| `IPV4_SRC_ADDR` | IPv4 source address |
| `IPV4_DST_ADDR` | IPv4 destination address |
| `L4_SRC_PORT` | Source port |
| `L4_DST_PORT` | Destination port |
| `PROTOCOL` | IP protocol identifier byte |
| `L7_PROTO` | Application protocol (numeric) |
| `IN_BYTES` | Incoming bytes |
| `OUT_BYTES` | Outgoing bytes |
| `IN_PKTS` | Incoming packets |
| `OUT_PKTS` | Outgoing packets |
| `FLOW_DURATION_MILLISECONDS` | Flow duration |
| `TCP_FLAGS` | Cumulative TCP flags |
| `CLIENT_TCP_FLAGS` | Cumulative client TCP flags |
| `SERVER_TCP_FLAGS` | Cumulative server TCP flags |
| `DURATION_IN` | Client→server stream duration (ms) |
| `DURATION_OUT` | Server→client stream duration (ms) |
| `MIN_TTL` | Minimum flow TTL |
| `MAX_TTL` | Maximum flow TTL |
| `LONGEST_FLOW_PKT` | Longest packet in flow (bytes) |
| `SHORTEST_FLOW_PKT` | Shortest packet in flow (bytes) |
| `MIN_IP_PKT_LEN` | Smallest IP packet length observed |
| `MAX_IP_PKT_LEN` | Largest IP packet length observed |
| `SRC_TO_DST_SECOND_BYTES` | src→dst bytes/sec |
| `DST_TO_SRC_SECOND_BYTES` | dst→src bytes/sec |
| `RETRANSMITTED_IN_BYTES` | Retransmitted TCP bytes (src→dst) |
| `RETRANSMITTED_IN_PKTS` | Retransmitted TCP packets (src→dst) |
| `RETRANSMITTED_OUT_BYTES` | Retransmitted TCP bytes (dst→src) |
| `RETRANSMITTED_OUT_PKTS` | Retransmitted TCP packets (dst→src) |
| `SRC_TO_DST_AVG_THROUGHPUT` | src→dst average throughput (bps) |
| `DST_TO_SRC_AVG_THROUGHPUT` | dst→src average throughput (bps) |
| `NUM_PKTS_UP_TO_128_BYTES` | Packet-size histogram bin |
| `NUM_PKTS_128_TO_256_BYTES` | Packet-size histogram bin |
| `NUM_PKTS_256_TO_512_BYTES` | Packet-size histogram bin |
| `NUM_PKTS_512_TO_1024_BYTES` | Packet-size histogram bin |
| `NUM_PKTS_1024_TO_1514_BYTES` | Packet-size histogram bin |
| `TCP_WIN_MAX_IN` | Max TCP window (src→dst) |
| `TCP_WIN_MAX_OUT` | Max TCP window (dst→src) |
| `ICMP_TYPE` | ICMP type × 256 + code |
| `ICMP_IPV4_TYPE` | ICMP type |
| `DNS_QUERY_ID` | DNS query transaction ID |
| `DNS_QUERY_TYPE` | DNS query type (1=A, 2=NS, …) |
| `DNS_TTL_ANSWER` | TTL of first A record |
| `FTP_COMMAND_RET_CODE` | FTP client command return code |

### 1.3 Label columns

Two additional columns beyond the 53:
- `Label` — binary, `0` = benign, `1` = malicious.
- `Attack` — string, the specific attack class name (or benign marker).

Always train and evaluate multi-class on `Attack`. Derive binary from it when a
binary number is needed; never train on `Label` alone.

---

## 2. Dataset statistics

### 2.1 Sizes and coverage

| Dataset | Rows | Benign | Malicious | src IPs | dst IPs | Days | Span |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NF-CICIDS2018** | 20,115,529 | 17,514,626 (87.07%) | 2,600,903 (12.93%) | **181,876** | 29,036 | 11 | 16.3 d |
| **NF-ToN-IoT** | 27,520,260 | 16,792,214 (61.02%) | 10,728,046 (38.98%) | 15,270 | 8,777 | 7 | 6.0 d |
| NF-UNSW-NB15 | 2,365,424 | 2,237,731 (94.60%) | 127,693 (5.40%) | **40** | **40** | 3 | 27.0 d |
| NF-BoT-IoT | 16,933,808 | 51,989 (0.31%) | 16,881,819 (99.69%) | **20** | **291** | 6 | 35.2 d |

Row counts and benign/malicious shares match the paper. IP counts, day counts and
spans are measured.

> **Discrepancy 1 — UNSW-NB15 minority class counts are wrong in the paper.**
> Its Table 3 gives Backdoor 1,226 / Analysis 2,381 / Shellcode 4,659. The data
> has **Backdoor 4,659 / Shellcode 2,381 / Analysis 1,226** — permuted. Use the
> measured values.
>
> **Discrepancy 2 — CICIDS2018 has 14 attack classes, not 6.** The paper
> aggregates into families; the CSV carries the **fine-grained original labels**.
> This changes the primary-dataset choice — see §2.3.
>
> **Discrepancy 3 — UNSW-NB15 spans 27 days, not 3.** Two capture sessions about
> 27 days apart (2015-01-22 and 2015-02-18) with a benign-only day between. Both
> attack days contain *all nine* attack classes.
>
> **Discrepancy 4 — ToN-IoT has 7 active days, not 10**, with a different per-day
> attack assignment than the paper's Table 4. Use the measured mapping in §4.

### 2.2 Class distributions (measured)

**NF-CICIDS2018 — 14 attack classes**

| Class | Count | Share | Ratio vs benign |
|---|---:|---:|---:|
| Benign | 17,514,626 | 87.07% | — |
| DDOS_attack-HOIC | 1,032,311 | 5.13% | 17:1 |
| FTP-BruteForce | 386,720 | 1.92% | 45:1 |
| DDoS_attacks-LOIC-HTTP | 288,589 | 1.43% | 61:1 |
| Bot | 207,703 | 1.03% | 84:1 |
| SSH-Bruteforce | 188,474 | 0.94% | 93:1 |
| Infilteration *(sic)* | 188,152 | 0.94% | 93:1 |
| DoS_attacks-SlowHTTPTest | 105,550 | 0.52% | 166:1 |
| DoS_attacks-Hulk | 100,076 | 0.50% | 175:1 |
| DoS_attacks-GoldenEye | 61,300 | 0.30% | 286:1 |
| DoS_attacks-Slowloris | 36,040 | 0.18% | 486:1 |
| DDOS_attack-LOIC-UDP | 3,450 | 0.017% | 5,076:1 |
| Brute_Force_-Web | 1,618 | 0.008% | 10,824:1 |
| Brute_Force_-XSS | 480 | 0.0024% | 36,489:1 |
| **SQL_Injection** | **440** | **0.0022%** | **39,806:1** |

`Infilteration` is **misspelled in the source data**. Map to canonical
`infiltration` — see §5.2.

**NF-ToN-IoT — 9 attack classes**

| Class | Count | Share |
|---|---:|---:|
| Benign | 16,792,214 | 61.02% |
| ddos | 4,141,256 | 15.05% |
| xss | 2,834,435 | 10.30% |
| password | 1,594,777 | 5.79% |
| scanning | 1,358,977 | 4.94% |
| injection | 381,777 | 1.39% |
| dos | 203,456 | 0.74% |
| Backdoor | 203,384 | 0.74% |
| mitm | 6,013 | 0.022% |
| ransomware | 3,971 | 0.014% |

**Label casing is inconsistent** — `Benign` and `Backdoor` are capitalised, every
other class is lowercase. Canonicalisation is mandatory, not cosmetic.

**NF-UNSW-NB15 — 9 attack classes** (corrected)

| Class | Count | Share |
|---|---:|---:|
| Benign | 2,237,731 | 94.60% |
| Exploits | 42,748 | 1.81% |
| Fuzzers | 33,816 | 1.43% |
| Generic | 19,651 | 0.83% |
| Reconnaissance | 17,074 | 0.72% |
| DoS | 5,980 | 0.25% |
| Backdoor | 4,659 | 0.20% |
| Shellcode | 2,381 | 0.10% |
| Analysis | 1,226 | 0.05% |
| **Worms** | **158** | **0.0067%** |

**NF-BoT-IoT — 4 attack classes**

| Class | Count | Share |
|---|---:|---:|
| DoS | 8,034,190 | 47.44% |
| DDoS | 7,150,882 | 42.23% |
| Reconnaissance | 1,695,132 | 10.01% |
| Benign | 51,989 | 0.31% |
| Theft | 1,615 | 0.0095% |

### 2.3 Primary dataset: NF-CICIDS2018

The measured data changes the choice. CICIDS2018 wins on every axis that matters:

| Criterion | CICIDS2018 | ToN-IoT |
|---|---|---|
| Attack classes | **14** | 9 |
| Unique source IPs | **181,876** | 15,270 |
| Active days | **11** | 7 |
| Openness sweep range | **wide** (hold out 1–6 of 14) | narrow |
| Class hierarchy | **yes** | no |

The hierarchy is decisive. CICIDS2018's fine-grained labels form families:

```
DoS    → DoS_attacks-{Hulk, GoldenEye, Slowloris, SlowHTTPTest}
DDoS   → DDOS_attack-{HOIC, LOIC-UDP}, DDoS_attacks-LOIC-HTTP
Brute  → {FTP-BruteForce, SSH-Bruteforce, Brute_Force_-Web, Brute_Force_-XSS}
Web    → {SQL_Injection, Brute_Force_-XSS, Brute_Force_-Web}
Other  → {Bot, Infiltration}
```

This enables a **more realistic and much harder open-set protocol** than holding
out a whole family: hold out `DoS_attacks-Slowloris` while training on
`DoS_attacks-Hulk`. That is the real operational question — *does the detector
flag a new variant of a family it already knows, or confidently misclassify it as
its sibling?* Holding out an entire unseen family is the easy case. See §4.1
Protocol B2.

ToN-IoT becomes the secondary dataset and the cross-dataset transfer partner.

---

## 3. TRAP 1 — unique-IP counts decide node granularity

| Dataset | Unique src IP | Unique dst IP | Unique src port | Unique dst port |
|---|---:|---:|---:|---:|
| NF-CICIDS2018 | 181,876 | 29,036 | 65,319 | 63,307 |
| NF-ToN-IoT | 15,270 | 8,777 | 65,536 | 65,536 |
| NF-UNSW-NB15 | **40** | **40** | 64,597 | 64,617 |
| NF-BoT-IoT | **20** | **291** | 65,536 | 65,536 |

**The failure mode.** NF-UNSW-NB15 contains only 40 unique source IPs and 40
unique destination IPs. An IP-level graph is therefore a near-complete graph on
≈40 nodes carrying millions of parallel edges. Consequences:

- Message passing aggregates over essentially the whole dataset — the GNN
  degenerates to a global pooling operation and adds nothing over a flow-wise MLP.
- The degree cap `K` becomes meaningless (every node is adjacent to every other).
- The **breakdown-point argument for C2 collapses**, because "the attacker's
  share of a neighbourhood" is undefined when all hosts share one neighbourhood.
- Any reported topology-derived gain would be an artefact.

NF-BoT-IoT is worse: 20 source IPs *and* 99.69% attack prevalence.

### 3.1 Mitigation — mandatory node-granularity policy

| Dataset | Node granularity | Role in the paper |
|---|---|---|
| **NF-CICIDS2018** | IP | **PRIMARY.** 14 attack classes, 181,876 source IPs, 11 days, class hierarchy. |
| **NF-ToN-IoT** | IP | **SECONDARY** and cross-dataset transfer partner. 9 classes, 15,270 source IPs. |
| **NF-UNSW-NB15** | **IP:port composite** | Tertiary. Yields ~64k node identities. Must be stated explicitly in the paper as a deliberate deviation, with the 40-IP count as justification. |
| **NF-BoT-IoT** | — | **Degenerate case.** Report as an explicit negative result: graph-based NIDS cannot help when topology is trivial and prevalence is inverted (0.31% benign). Do not use for headline numbers. |

Reporting the BoT-IoT degeneracy honestly is a strength, not a gap — it
demonstrates the method's applicability boundary, which reviewers reward.

### 3.2 Implementation

`src/argus/graph/builder.py` must read node granularity from the dataset config
(`config/dataset/*.yaml`, key `node_granularity: ip | ip_port`) and must raise a
hard error if `ip` is selected for a dataset whose unique-source-IP count is
below a threshold (set `min_unique_src_ip: 1000`). Fail loudly; do not warn.

---

## 4. TRAP 2 — attacks are segregated by day

Measured active-day → attack mapping (**not** the paper's Table 4, which differs):

**NF-CICIDS2018** — the most strongly segregated, and the most dangerous:

| Day | Attacks present |
|---:|---|
| 1 | FTP-BruteForce, SSH-Bruteforce |
| 2 | DoS_attacks-GoldenEye, DoS_attacks-Slowloris |
| 3 | DoS_attacks-Hulk, DoS_attacks-SlowHTTPTest |
| 4 | DDoS_attacks-LOIC-HTTP |
| 5 | DDOS_attack-HOIC, DDOS_attack-LOIC-UDP |
| 6 | Brute_Force_-Web, Brute_Force_-XSS, SQL_Injection |
| 7 | *(benign only)* |
| 8 | *(benign only)* |
| 9 | Infilteration |
| 10 | Infilteration |
| 11 | Bot |

**NF-ToN-IoT**

| Day | Attacks present |
|---:|---|
| 1 | scanning |
| 2 | dos, scanning |
| 3 | ddos, dos, injection |
| 4 | ddos, password |
| 5 | password, xss |
| 6 | Backdoor, ransomware |
| 7 | Backdoor, mitm |

**NF-UNSW-NB15** — days 1 and 3 each contain *all nine* attack classes; day 2 is
benign only. The two attack days are ~27 days apart.

**NF-BoT-IoT**

| Day | Attacks present |
|---:|---|
| 1–3 | Reconnaissance |
| 4 | DoS, DDoS |
| 5–6 | Theft |

**The failure mode.** In CICIDS2018, `FTP-BruteForce` occurs only on day 1 and
`Bot` only on day 11. A naive chronological 70/15/15 split by timestamp puts
FTP-BruteForce entirely in train and Bot entirely in test. The closed-set
comparison is destroyed: the model is scored on classes it never saw while
classes it learned never appear at test. Macro-F1 becomes uninterpretable and
comparison against baselines becomes meaningless.

NF-UNSW-NB15 is the exception — all nine attacks appear on both attack days,
which is the most realistic mixed-traffic scenario.

### 4.1 Mitigation — two split protocols, never conflated

**Protocol A — closed-set: per-class stratified temporal split.**

```
for each class c in {benign} ∪ attack_classes:
    rows_c ← all rows with Attack == c
    sort rows_c by FLOW_START_MILLISECONDS ascending
    n ← len(rows_c)
    train ← rows_c[0 : 0.70n]
    val   ← rows_c[0.70n : 0.85n]
    test  ← rows_c[0.85n : n]
concatenate per-class parts into global train/val/test
re-sort each split by FLOW_START_MILLISECONDS  (windowing requires time order)
```

This preserves chronology *within* each class (no future→past leakage of that
class's evolution) while guaranteeing every class appears in every split. It is
the honest compromise given the day segregation, and must be described in the
paper's methodology with the day table as justification.

**Protocol B — open-set: leave-attack-classes-out.**

```
choose holdout set H ⊂ attack_classes, |H| = C  (default C = 3)
train/val ← Protocol-A splits restricted to classes ∉ H
test      ← Protocol-A test split, PLUS all rows of classes ∈ H
            (rows of H classes are labelled UNKNOWN at evaluation time)
repeat for R = 5 different random choices of H; report mean ± std
```

The day segregation *helps* here: holding out a class largely holds out whole
days, which is exactly the realistic "a new attack campaign begins" scenario.

**Protocol B2 — open-set, within-family (CICIDS2018 only). The hard case.**

CICIDS2018's fine-grained labels form families (§2.3). Holding out an entire
family is the easy open-set problem: the novel class looks nothing like anything
seen. The operationally important question is harder — *when a new variant of a
known family appears, does the detector flag it or confidently absorb it into a
sibling variant?*

```
families = {
  "dos":   [DoS_attacks-Hulk, DoS_attacks-GoldenEye,
            DoS_attacks-Slowloris, DoS_attacks-SlowHTTPTest],
  "ddos":  [DDOS_attack-HOIC, DDOS_attack-LOIC-UDP, DDoS_attacks-LOIC-HTTP],
  "brute": [FTP-BruteForce, SSH-Bruteforce, Brute_Force_-Web, Brute_Force_-XSS],
  "web":   [SQL_Injection, Brute_Force_-XSS, Brute_Force_-Web],
  "other": [Bot, Infiltration],
}

for each family F with |F| >= 2:
    for each member v in F:
        hold out v as UNKNOWN; keep all other members of F in training
        evaluate: unknown TPR on v, and the confusion of v against its siblings
```

Report **sibling-absorption rate**: the fraction of held-out-variant flows
classified with high confidence as a sibling variant rather than flagged UNKNOWN.
A low sibling-absorption rate is the strongest possible evidence for C1, and no
prior open-set NIDS paper reports this because no prior work used fine-grained
labels. This is a genuinely novel evaluation axis — give it its own table (T3b).

Expect B2 numbers to be substantially worse than B. **Report both.** A method
that only works on the easy case should say so.

**Protocol C — cross-dataset transfer.** Train on NF-CICIDS2018 (Protocol A train
split), test on NF-ToN-IoT, and the reverse. Classes present in the target but
not the source count as UNKNOWN. Schemas are identical so no feature alignment is
needed — but the label vocabularies are disjoint, so evaluation is by family
mapping (see `08_EVALUATION.md` §4.4) plus unknown detection.

### 4.2 Leakage audit — must run and must pass

`src/argus/data/audit.py` asserts, for every produced split:

1. No exact duplicate rows across train/val/test (hash the full feature tuple).
2. Every class in `test` has ≥1 example in `train` **under Protocol A only**
   (this assertion is deliberately disabled for Protocols B and B2).
3. Scalers/encoders were fitted only on train — verified by checksumming the
   fitted-artifact provenance record.
4. No graph window crosses a split boundary. Windows are built independently per
   split; a flow in train must never appear in a test-split window.
5. Report the per-split class histogram and the per-split time range.
6. **Identity overlap**: report the fraction of test flows whose `(src IP, dst IP)`
   pair never appears in train. If this is near 0, the model can succeed by
   memorising hosts — see §7.
7. **Near-duplicate rate**: fraction of test flows within L2 distance `1e-6` of
   some train flow in normalised feature space, estimated on a 100k sample.
   Report it; do not silently drop.

Emit `results/runs/<run_id>/split_audit.json`. Refuse to train if any of checks
1–4 fails. Checks 5–7 are reported, not enforced — they are context the paper
must disclose.

---

## 5. Preprocessing and subsampling

### 5.1 Cleaning

1. Drop rows with null `Attack` or null `FLOW_START_MILLISECONDS`.
2. Clip negative durations and negative IAT values to 0 (nProbe artefacts).
3. Deduplicate on the full 53-tuple; record the count removed. **Do this before
   splitting** — duplicates straddling a split boundary are direct test
   contamination.
4. Cast timestamps to `int64` ms; verify the range is plausible
   (2014-01-01 ≤ t ≤ 2020-01-01 for these captures).
5. Canonicalise `Attack` strings — see §5.2. This is mandatory.

### 5.2 Label canonicalisation

The raw labels are inconsistent across and within datasets: mixed casing in
ToN-IoT, a misspelling in CICIDS2018, and multiple naming conventions. Store the
mapping in `src/argus/constants.py` as an explicit dict — never rely on
`str.lower()` alone, because that would not fix `Infilteration`.

**NF-CICIDS2018**

| Raw label | Canonical | Family |
|---|---|---|
| `Benign` | `benign` | — |
| `DDOS_attack-HOIC` | `ddos_hoic` | ddos |
| `DDOS_attack-LOIC-UDP` | `ddos_loic_udp` | ddos |
| `DDoS_attacks-LOIC-HTTP` | `ddos_loic_http` | ddos |
| `DoS_attacks-Hulk` | `dos_hulk` | dos |
| `DoS_attacks-GoldenEye` | `dos_goldeneye` | dos |
| `DoS_attacks-Slowloris` | `dos_slowloris` | dos |
| `DoS_attacks-SlowHTTPTest` | `dos_slowhttptest` | dos |
| `FTP-BruteForce` | `brute_ftp` | brute |
| `SSH-Bruteforce` | `brute_ssh` | brute |
| `Brute_Force_-Web` | `brute_web` | brute, web |
| `Brute_Force_-XSS` | `brute_xss` | brute, web |
| `SQL_Injection` | `sql_injection` | web |
| `Infilteration` **(sic)** | `infiltration` | other |
| `Bot` | `bot` | other |

**NF-ToN-IoT**

| Raw label | Canonical |
|---|---|
| `Benign` | `benign` |
| `Backdoor` | `backdoor` |
| `ddos` | `ddos` |
| `dos` | `dos` |
| `injection` | `injection` |
| `mitm` | `mitm` |
| `password` | `password` |
| `ransomware` | `ransomware` |
| `scanning` | `scanning` |
| `xss` | `xss` |

**NF-UNSW-NB15** — lowercase the nine names as-is.
**NF-BoT-IoT** — lowercase `DoS`, `DDoS`, `Reconnaissance`, `Theft`, `Benign`.

`constants.py` must expose `canonicalise(dataset, raw_label) -> str` and
`FAMILY_OF: dict[str, str]`, and must **raise on an unrecognised label** rather
than silently passing it through. A new label appearing in the data means the
vocabulary is stale, and that must be loud.

### 5.3 Subsampling for compute budget

Kaggle sessions are ~12 h on ~16 GB GPU. Targets:

| Dataset | Raw flows | Target | Note |
|---|---:|---:|---|
| NF-CICIDS2018 | 20,115,529 | **4,000,000** | primary |
| NF-ToN-IoT | 27,520,260 | **4,000,000** | secondary |
| NF-UNSW-NB15 | 2,365,424 | full | no subsampling |
| NF-BoT-IoT | 16,933,808 | 1,000,000 | degenerate case only |

**Stratified temporal subsampling** — never uniform random:

```
for each canonical class c:
    if count_c <= minority_threshold:  keep ALL rows of c
    else:
        partition rows_c into T = 100 equal-count consecutive time bins
        draw quota(c) / T rows from each bin, without replacement
```

`minority_threshold = 200_000`. Under this rule, on CICIDS2018 **every attack
class except `DDOS_attack-HOIC`, `FTP-BruteForce` and `DDoS_attacks-LOIC-HTTP` is
kept in full**, and the 4 M budget is spent mostly on subsampling benign. This is
the correct trade: benign has 17.5 M rows and the tail classes have 440.

Benign quota is whatever remains after all attack quotas are allocated, floored
at 50% of the total budget so the benign manifold stays well covered (it needs to
support 4 benign sub-prototypes — see `05_ARCHITECTURE.md` §6.2).

Record the realised class histogram before and after in
`data/processed/<dataset>/subsample_report.json`.

### 5.4 Storage format

Write partitioned Parquet, one directory per dataset per split, sorted by
`FLOW_START_MILLISECONDS`, row-group size 200,000. Raw CSVs stay in `dataset/`
untouched. Never re-derive splits at training time — they are produced once by
`scripts/02_build_splits.py` and referenced by path.

---

## 6. Class imbalance — measured severity and handling

The imbalance here is more extreme than most NIDS papers acknowledge. On the
primary dataset the ratio between the largest and smallest class is
**39,806:1** (`Benign` 17,514,626 vs `SQL_Injection` 440).

### 6.1 Severity tiers and per-tier policy

| Tier | Count | CICIDS2018 members | Policy |
|---|---|---|---|
| **Head** | > 1 M | benign, ddos_hoic | Subsample; cap loss contribution |
| **Body** | 50 k – 1 M | brute_ftp, ddos_loic_http, bot, brute_ssh, infiltration, dos_slowhttptest, dos_hulk, dos_goldeneye | Normal closed-set training |
| **Tail** | 1 k – 50 k | dos_slowloris, ddos_loic_udp, brute_web | Keep all rows; 1 sub-prototype; effective-number weighting |
| **Extreme** | < 1 k | brute_xss (480), sql_injection (440) | **Route to few-shot registration**, not closed-set training — see §6.3 |

ToN-IoT extreme tier: `ransomware` (3,971), `mitm` (6,013).
UNSW-NB15 extreme tier: `worms` (158), `analysis` (1,226).
BoT-IoT extreme tier: `theft` (1,615) — and `benign` is itself only 0.31%.

### 6.2 Mechanisms (applied together)

1. **Class-balanced loss targets.** All flows in an anchor bin are embedded (the
   graph needs them), but the loss is computed on a class-balanced subsample of
   at most `n_per_class = 32` targets per class per batch. This decouples graph
   density from loss balance and is the primary mechanism.
2. **Effective-number weighting**, `w_c = (1-ν)/(1-ν^{n_c})` with `ν = 0.999`,
   normalised to mean 1.
3. **Guaranteed class coverage per batch.** Construct batches so each contains
   targets from at least `min(8, C)` distinct classes. Without this, prototype
   gradients are erratic: a batch of pure benign updates only the benign
   prototypes and drags the geometry.
4. **Tier-aware sub-prototype counts** (`05_ARCHITECTURE.md` §6.2): 4 for benign,
   2 for body classes, 1 for tail.
5. **Extreme tier → few-shot.** See §6.3.

### 6.3 Extreme-tier classes become C1 demonstrations

Classes below `min_count = 100` for prototype training are **excluded from
Stage-1 closed-set training and evaluated as few-shot registration targets
instead**. On CICIDS2018 this puts `sql_injection` (440) and `brute_xss` (480) in
the few-shot protocol; on UNSW-NB15 it puts `worms` (158).

This is not a workaround. Fitting a prototype to 440 samples drawn from a
single capture day memorises those samples; the resulting closed-set F1 would be
both high and meaningless. Turning the rarest classes into a demonstration that
the model can *register* them from 20 labelled examples is the honest framing and
directly serves C1. State it explicitly in the paper.

### 6.4 Explicitly rejected techniques

| Technique | Why rejected |
|---|---|
| **SMOTE / ADASYN** | Interpolates in feature space, producing physically impossible flows — fractional packet counts, byte totals inconsistent with packet counts, IAT statistics violating `min ≤ avg ≤ max`. Worse here than usual because the constraint set in `10_ADVERSARIAL.md` §1.3 is exactly the set SMOTE violates. |
| **Random oversampling (duplication)** | With `n_c = 158` this memorises. Also corrupts the graph: duplicated flows create fake edges that change topology. |
| **Random undersampling of benign** | Discards the benign diversity the 4 benign sub-prototypes need, and would inflate apparent unknown-detection performance. |
| **Focal loss** | Considered and not adopted: it interacts badly with the evidential head, since down-weighting easy examples also suppresses the evidence magnitude that calibration depends on. Effective-number weighting is the safer choice. Revisit only if tail F1 is inadequate, and re-check ECE if so. |
| **Threshold moving on test** | Thresholds are selected on validation only (`05_ARCHITECTURE.md` §6.4). |

---

## 7. TRAP 3 — host identity leakage

The third trap, and the one most likely to survive into a published result
unnoticed.

**The failure mode.** In these captures, attacker hosts are usually a small fixed
set of machines. A model can therefore achieve excellent metrics by learning
*which hosts are malicious* instead of *which behaviour is malicious* — and such
a model transfers to no other network, which would falsify the entire
deployability claim. NF-UNSW-NB15 has 40 source IPs; memorising 40 identities is
trivial. Even on CICIDS2018 the attack infrastructure is a handful of hosts.

Graph models are **more** exposed to this than flow-independent models, because
message passing and the per-node GRU memory give identity information additional
routes into the prediction.

### 7.1 Mandatory identity-leakage audit

Run before any modelling, once per dataset. `src/argus/data/audit.py`.

```
1. Train a depth-8 decision tree on ONLY:
       (src IP, dst IP, src port bucket, dst port bucket)
   using the Protocol-A split. Report macro-F1.
   -> This is the IDENTITY FLOOR: performance obtainable with zero behavioural
      information.

2. Report the fraction of test flows whose (src IP, dst IP) pair is unseen in
   train  ->  the UNSEEN-PAIR RATE.

3. Report per-class attacker-host concentration: for each attack class, the
   number of distinct source IPs, and the share of that class's flows
   originating from its single most common source IP.
```

**Interpretation and required disclosure.** If the identity floor is high (say
macro-F1 > 0.6), then *any* model's headline number on that dataset is partly
identity memorisation, and the paper must report the floor alongside its own
results. This is not optional — a reviewer who computes it independently and
finds it undisclosed will reject the paper.

### 7.2 Mitigations

1. **Node identity never enters the feature vector.** IPs and ports are used to
   build graph structure and (for UNSW) node identity; the raw address is never a
   model input. Port *buckets* are inputs; raw port values are not.
2. **Memory dropout `p_mem = 0.1`** (`05_ARCHITECTURE.md` §4) prevents the GRU
   state from becoming a per-host label cache.
3. **Host-disjoint evaluation (Protocol D, diagnostic).** Build an additional
   split where the *attacker* source IPs in test are disjoint from those in
   train, where the dataset permits it. Report the macro-F1 drop from Protocol A
   to Protocol D as the **identity-reliance gap**. Where a dataset does not
   permit it (UNSW-NB15, BoT-IoT), say so.
4. **Cross-dataset transfer (Protocol C)** is the ultimate control: no host in
   ToN-IoT appears in CICIDS2018, so transfer performance cannot be identity
   memorisation.

The identity floor, the identity-reliance gap, and the transfer result together
form a coherent argument that ARGUS learns behaviour rather than identity. That
argument is worth a short subsection in the paper and very few NIDS papers make
it.

---

## 8. Dataset config schema

`config/dataset/cicids2018.yaml` (the primary; others follow the same shape):

```yaml
name: cicids2018
display_name: NF-CICIDS2018-v3
csv_path: dataset/NF-CICIDS2018-v3.csv
role: primary                    # primary | secondary | tertiary | degenerate
node_granularity: ip             # ip | ip_port
min_unique_src_ip: 1000          # hard-fail guard for TRAP 1
measured_unique_src_ip: 181876   # audited value; loader verifies within 1%

rows: 20115529
subsample_target: 4000000
minority_threshold: 200000       # classes at or below this are kept in full
benign_floor_fraction: 0.50      # benign gets >= 50% of the budget

benign_label: benign
n_attack_classes: 14
attack_classes:
  - ddos_hoic
  - ddos_loic_udp
  - ddos_loic_http
  - dos_hulk
  - dos_goldeneye
  - dos_slowloris
  - dos_slowhttptest
  - brute_ftp
  - brute_ssh
  - brute_web
  - brute_xss
  - sql_injection
  - infiltration
  - bot

families:                        # enables Protocol B2
  dos:   [dos_hulk, dos_goldeneye, dos_slowloris, dos_slowhttptest]
  ddos:  [ddos_hoic, ddos_loic_udp, ddos_loic_http]
  brute: [brute_ftp, brute_ssh, brute_web, brute_xss]
  web:   [sql_injection, brute_xss, brute_web]
  other: [bot, infiltration]

min_count_for_prototype: 100     # below this -> few-shot registration target
few_shot_classes: [sql_injection, brute_xss]   # extreme tier, see 6.3

split:
  protocol_a:  {train: 0.70, val: 0.15, test: 0.15}
  protocol_b:  {holdout_size: 3, repeats: 5}
  protocol_b2: {enabled: true}   # within-family holdout, CICIDS2018 only
  protocol_d:  {enabled: true}   # host-disjoint diagnostic

windows:
  short_seconds: 1
  mid_seconds: 30
  long_seconds: 300
```

Dataset-specific overrides:

| Key | cicids2018 | ton_iot | unsw_nb15 | bot_iot |
|---|---|---|---|---|
| `role` | primary | secondary | tertiary | degenerate |
| `node_granularity` | ip | ip | **ip_port** | ip_port |
| `measured_unique_src_ip` | 181,876 | 15,270 | **40** | **20** |
| `subsample_target` | 4,000,000 | 4,000,000 | null (full) | 1,000,000 |
| `n_attack_classes` | 14 | 9 | 9 | 4 |
| `protocol_b2.enabled` | true | false | false | false |
| `protocol_d.enabled` | true | true | **false** | **false** |
| `few_shot_classes` | sql_injection, brute_xss | ransomware, mitm | worms, analysis | theft |
