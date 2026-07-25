# 14 — Paper Outline

Target: IEEE/ACM conference, 8–10 pages, two-column IEEE format
(NOMS / TrustCom / ICC / EuroS&P workshop).

Working title options:

1. *"Open-World Graph Intrusion Detection: Evidential Prototypes and
   Structure-Robust Aggregation"*
2. *"ARGUS: Detecting What You Were Not Trained For, Without Retraining"*
3. *"Beyond Closed-Set GNN-NIDS: Uncertainty-Aware, Structurally Robust,
   Temporally Grounded Intrusion Detection"*

Option 1 is the safest for a systems/security venue; it names both mechanisms.

---

## 1. Section plan

| § | Section | Pages | Content |
|---|---|---:|---|
| I | Introduction | 1.0 | Problem, three deployment realities, four contributions, results preview |
| II | Background & Related Work | 1.0 | GNN-NIDS lineage; open-set NIDS (all non-graph); structural attack evidence |
| III | Threat Model & Problem Formulation | 0.5 | Attacker capabilities table; open-world problem statement |
| IV | Method | 2.5 | IV-A graph construction; IV-B temporal representation; IV-C SR-TEG; IV-D EPC head; IV-E few-shot registration |
| V | Experimental Setup | 0.75 | Datasets, the two traps and our mitigations, splits, baselines, metrics |
| VI | Results | 2.75 | VI-A closed-set; VI-B open-set; VI-C few-shot; VI-D temporal ladder; VI-E adversarial; VI-F explanation; VI-G deployment |
| VII | Discussion & Limitations | 0.5 | Robustness–accuracy frontier; path-control boundary; BoT-IoT degeneracy |
| VIII | Conclusion | 0.25 | |
| — | References | 0.75 | ~35 refs |

---

## 2. Claim → evidence map

Every claim must point at a table or figure. Anything unsupported gets cut.

| Claim | Stated in | Evidence | Backup |
|---|---|---|---|
| **C1a** Single head yields classify / defer / unknown | §IV-D | **T3**, F3, F8 | E-OS-1, E-SEL-1 |
| **C1b** Calibrated uncertainty | §IV-D | **T8**, F4 | E-CAL-1 |
| **C1c** Few-shot registration, zero forgetting | §IV-E | **T4** | E-FS-1, E-FS-BL1 |
| **C2a** Robust aggregation resists structural injection | §IV-C | **T5**, **F6** | E-ADV-A2 |
| **C2b** Stratified sampling is itself a defence | §IV-A | **F6** (A2 vs A2b gap) | E-ADV-A2b/c |
| **C2c** Provenance partition reduces reliance on forgeable features | §IV-C | T10 (`λ_ch` ablation) | E-ABL-CH |
| **C3a** NF-v3 temporal features are worth *X* once conditioned | §VI-D | **T6** (L0→L2) | E-TEMP-L0..L2 |
| **C3b** Each mechanism rescues identifiable classes | §VI-D | **F10** | E-TEMP-L3..L7 |
| **C3c** Attack classes select different time scales | §VI-D | **F5** | E-TEMP-L6 |
| **C4a** Robust aggregation denies injected edges influence | §VI-F | **F7** | E-XAI-2 |
| **C4b** Native attribution is more stable than post-hoc | §VI-F | T11 | E-XAI-1 |
| **C4c** UNKNOWNs form coherent, labelable clusters | §VI-F | T12 | E-XAI-3 |
| Deployability | §VI-G | T9 | E-DEP-1 |
| Transfer | §VI-B | T7 | E-TR-* |

---

## 3. Section notes

### §I Introduction

Open with the three deployment realities from `00_OVERVIEW.md` §1 — new attacks
appear, models must update without rebuilds, the detector is itself a target.
State that these are not three problems but one: **the model must behave sanely
on input it was not trained to expect.**

Put the novelty evidence in the introduction, concretely: *"a full-text search of
the literature returns no graph-based open-set NIDS; every open-set NIDS we found
classifies flows independently."* That single sentence does more than a paragraph
of hedged positioning.

Preview one number per claim. Do not preview all fourteen.

### §II Related Work

Three subsections mirroring `01_RELATED_WORK.md`:
1. GNN-NIDS — dense and accuracy-focused; note that all are closed-set.
2. Open-set / zero-day NIDS — capable, but all flow-independent. **This is the
   gap sentence.**
3. Adversarial robustness of GNN-NIDS — Venturi et al. and REAL-IoT.

Explicitly differentiate Dai et al. (arXiv:2606.17109) on the five points in
`01_RELATED_WORK.md` §1.1. A reviewer who knows that paper will look for this.

### §III Threat Model

Half a page, mostly the capability table from `10_ADVERSARIAL.md` §1.2. Security
venues reward an explicit threat model and punish its absence. State the
path-control exclusion here, not buried in limitations.

### §IV Method

Budget carefully — 2.5 pages for five subsections.

- **IV-A Graph construction** (0.4 p). Nodes, edges, multi-scale windows,
  degree-capped stratified sampling. Include figure F1.
- **IV-B Temporal representation** (0.5 p). TE1–TE7 compressed into a table with
  one row each: mechanism, what it captures, cost. Detail lives in §VI-D.
- **IV-C SR-TEG** (0.6 p). Provenance channels, time-decayed attention, robust
  aggregation. **State the breakdown point as a short proposition** — a boxed
  statement, not a theorem environment, since the empirical result is what we
  actually stand behind.
- **IV-D EPC head** (0.7 p). Embedding → prototypes → Dirichlet evidence →
  three-way decision. Emphasise the unification argument: defer and unknown are
  two readings of one posterior. Equations for `e_k`, `α_k`, `S`, `u`.
- **IV-E Few-shot registration** (0.3 p). Four lines of pseudocode. Make the
  zero-forgetting guarantee explicit and note it is structural, not empirical.

### §V Experimental Setup

**Do not skip the traps.** A short, confident paragraph on each:

> *"NF-UNSW-NB15 contains only 40 unique source IP addresses; an IP-level graph
> is therefore near-complete and topologically uninformative. We use IP:port
> composite nodes for that dataset and note that prior work reporting IP-level
> graph results on UNSW-NB15 should be interpreted with this in mind."*

> *"Attack classes in NF-v3 are segregated by capture day. A chronological split
> therefore places entire classes on one side of the boundary. We use a per-class
> stratified temporal split for closed-set evaluation and exploit the day
> structure deliberately for leave-class-out evaluation."*

Both paragraphs demonstrate methodological care and pre-empt reviewer objections.
They may also be the most-cited part of the paper.

### §VI Results

Lead each subsection with the finding, then the evidence.

- **VI-A Closed-set.** Expect parity. **Say "parity" plainly** — claiming a small
  win on a saturated benchmark invites scepticism about everything else.
- **VI-B Open-set.** The headline. Mean ± std over 5 holdouts; note that prior
  work reports single holdouts.
- **VI-C Few-shot.** Lead with the zero-forgetting row against the fine-tuning
  baseline's degradation.
- **VI-D Temporal ladder.** Lead with the L0→L2 finding as a named result. Then
  F10 and F5.
- **VI-E Adversarial.** F6 plus the empirical breakdown point. Include the
  robustness–accuracy frontier here, not in limitations.
- **VI-F Explanation.** F7 first — it verifies §VI-E rather than merely
  illustrating the method. Then triage purity.
- **VI-G Deployment.** T9. If ARGUS is slower than E-GraphSAGE, report the factor
  and price it against the gains.

### §VII Discussion & Limitations

Cover honestly and briefly:
1. Robustness costs clean accuracy; the frontier is the honest framing.
2. The provenance partition assumes no network-path control.
3. BoT-IoT is degenerate for graph methods — an applicability boundary.
4. Synthetic datasets; real-network validation is future work.
5. Whatever the temporal ladder refuted from `09_TEMPORAL_STUDY.md` §3.3.

Limitations sections that name real boundaries are read as competence. Vague ones
are read as evasion.

---

## 4. Figures and tables in the paper

Space is the binding constraint. Priority order:

**Must include (7 items)**
F1 architecture · T2 closed-set · **T3 open-set** · **T4 few-shot** ·
**T5/F6 adversarial** · **T6 temporal ladder** · **F7 attribution verification**

**Include if space (5)**
F3 vacuity histogram · F5 scale gate by class · F10 per-class ladder heatmap ·
T8 calibration · T9 deployment

**Appendix / supplementary**
T1 dataset stats · F2 confusion · F4 reliability · F8 risk–coverage ·
F9 openness sweep · T7 transfer · T10 ablation grid · T11 XAI quality ·
T12 triage · full per-class tables

Combine where possible: T5 and F6 as one float; F3 and F4 side by side.

---

## 5. Writing rules

1. **No claim without a table or figure reference.**
2. Report mean ± std everywhere. Single-seed numbers appear nowhere.
3. Never write "significantly" without a bootstrap CI behind it.
4. Where a result is within noise, write that it is within noise.
5. Never quote a baseline number from another paper — say "re-run on our splits".
6. Report refuted hypotheses from the temporal ladder. An honest negative
   strengthens the surrounding positives.
7. Accuracy appears nowhere as a headline number.
8. Every number traces to a `run_id`; keep the mapping in the artifact repo.

---

## 6. Pre-submission checklist

- [ ] Every claim in §I maps to a result in §VI
- [ ] Every table/figure is referenced in the text
- [ ] All numbers are mean ± std over ≥3 seeds
- [ ] Baselines re-run on our splits; stated explicitly
- [ ] Threat model explicit; exclusions stated
- [ ] Both dataset traps documented in §V
- [ ] Limitations name real boundaries
- [ ] Dai et al. cited and differentiated
- [ ] Venturi et al. and REAL-IoT cited as motivation
- [ ] NF-v3 dataset paper cited; TE7 framed as follow-up to their open problem
- [ ] Code and config release statement included
- [ ] Page limit met without shrinking margins or fonts
