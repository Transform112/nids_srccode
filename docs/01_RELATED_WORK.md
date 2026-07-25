# 01 — Related Work

All entries below were verified on 2026-07-25. Identifiers are given so no
further searching is required. Where an arXiv ID and a DOI both exist, both are
listed; cite the published venue.

---

## 1. GNN-based NIDS (the direct lineage)

| Work | ID | What it does | What it lacks | ARGUS differs by |
|---|---|---|---|---|
| **E-GraphSAGE** — Lo et al., NOMS 2022 | arXiv:2103.16329, DOI 10.1109/NOMS54207.2022.9789878 | First practical edge-feature GNN NIDS. Nodes = hosts with **constant** feature vectors; edges = flows carrying features. Mean aggregation. | Closed-set. Constant node init discards host context. Mean aggregation is unbounded-influence. No temporal modelling. Random splits. | Real node features, robust aggregation, open-world head, temporal stack |
| **E-ResGAT / modified E-GraphSAGE** — Chang & Branco | arXiv:2111.13597 | Adds residual connections to GraphSAGE and GAT for class imbalance. | Closed-set, no robustness, no temporal. | Same as above |
| **EDGMAT** — Li et al. | arXiv:2310.17348 | Edge-directed multi-head attention; weights neighbours by similarity. | Attention is *learned*, therefore attacker-influenceable. Closed-set. | Attention combined with a robust aggregator and explicit time decay |
| **Anomal-E** — Caville et al., KBS 2022 | arXiv:2207.06819, DOI 10.1016/j.knosys.2022.110030 | Self-supervised (DGI-style) edge embeddings + downstream anomaly detector. No labels needed. | Binary anomaly only, cannot name the attack class, no unknown *class* semantics. | Multi-class + explicit UNKNOWN + few-shot naming of new classes |
| **GraphIDS** — Guerra et al., NeurIPS 2025 | arXiv:2509.16625 | Masked autoencoder over local graph context + Transformer reconstructor; flags high reconstruction error. | Unsupervised binary; reconstruction error is not calibrated; no adversarial eval. | Calibrated evidential uncertainty, multi-class, adversarial eval |
| **PPT-GNN** — Van Langendonck et al. | arXiv:2406.13365 | Pre-trained spatio-temporal GNN, near-real-time, fine-tunes to unseen networks with few labels. | Fine-tuning = gradient steps = forgetting risk. Closed-set. | Registration without gradient steps; forgetting is structurally zero |
| **STEG** — Zoubir & Missaoui | arXiv:2404.10800 | Scattering transform on edge features + Node2Vec node init. | Closed-set, no robustness. | — (cite as feature-representation prior art) |
| **CAGN-GAT Fusion** — Jahin et al., IEA/AIE 2025 | arXiv:2503.00961 | Contrastive attentive graph net; studies edge perturbation and feature masking as augmentation. | Augmentation only; not an adversarial threat model. | We use perturbation as *attack*, not augmentation |
| **GTCN-G** — Xu et al., TrustCom 2025 | arXiv:2510.07285 | Gated TCN + GCN + GAT residual for imbalance. | Temporal via TCN on flow sequence, not on graph time. Closed-set. | Time-aware message passing, not a separate temporal branch |
| **XG-NID** — Farrukh et al. | arXiv:2408.16021 | Heterogeneous graph fusing flow-level and packet-level data; LLM generates explanations. | Requires packet data. LLM explanations are unverifiable and unmetriced. | Native, quantitatively-evaluated attribution; flow-level only |
| **X-CBA** — Kaya et al., ICC 2024 | DOI 10.1109/ICC51166.2024.10622177, arXiv:2402.00839 | GNN embeddings + CatBoost + XAI (local & global). | XAI is illustrative; no fidelity/stability metrics; closed-set. | Explanation metrics + explanation used to verify robustness |

### 1.1 Closest competitor — read carefully before writing

**Dai et al., "Timestamp-Aware Spatio-Temporal Graph Contrastive Learning for
Network Intrusion Detection"** — arXiv:2606.17109 (June 2026).

Builds temporal graphs from *real timestamps*, uses an E-GraphSAGE + LSTM
encoder, and multi-view (temporal / spatial / feature) graph contrastive
learning with gradient-norm adaptive loss weighting. Self-supervised; claims
performance comparable to supervised SOTA.

**This is the nearest work to C3 and must be cited and explicitly
differentiated.** Differentiation points, all defensible:

1. They use a **single** temporal graph scale; ARGUS uses three concurrent
   scales with gated fusion (TE5).
2. They feed timestamps for *ordering*; ARGUS uses a **functional Δt encoding**
   that can represent periodicity (TE3), plus a per-host **spectral** descriptor
   (TE7). Ordering ≠ rhythm.
3. They are self-supervised and **closed-set**; ARGUS is open-world with
   few-shot class registration.
4. They perform **no adversarial evaluation**.
5. They do not quantify *which* temporal component helps *which* attack class;
   ARGUS's L0→L7 ladder does, per-class.

## 2. Vulnerability evidence (motivates C2)

| Work | ID | Key result to cite |
|---|---|---|
| **Venturi et al.** | arXiv:2403.11830 (submitted IEEE TIFS) | First formalisation of adversarial attacks *specifically for GNN-based NIDS*, with problem-space constraints an attacker must respect. Finds GNN-NIDS are **more** robust than flow-independent models to classical feature-based attacks but **susceptible to structure-based attacks**. This is the single most important motivating citation for C2. |
| **REAL-IoT** — Zhan et al. | arXiv:2507.10836 | Unified cross-dataset benchmark + physical IoT testbed capture. Finds performance drops versus standard benchmarks, "quantifying susceptibility to drift and realistic attacks" and concluding the literature **overestimates GNN-NIDS resilience**. |
| **Poster: LLM agents for GNN robustness** — Zhan et al. | arXiv:2506.20806 | Same group; notes current robustness evaluations "rely on unrealistic synthetic perturbations". Cite when justifying our constrained threat model. |
| **Pujol-Perich et al.** | arXiv:2107.14756 | Claims GNN structural modelling gives robustness to packet-size/IAT manipulation where classical ML loses up to 50% F1. **Cite as the counter-claim we are testing** — our A2/A5 results will qualify it. |

## 3. Open-set / zero-day NIDS — all non-graph (motivates C1)

This is the key novelty evidence: every one of these is flow-independent.

| Work | ID | Mechanism | Graph? |
|---|---|---|---|
| **EFC-OSR** — Souza et al., Computers & Security 2025 | DOI 10.1016/j.cose.2025.104569, arXiv:2109.11224 | Single-layer energy-based flow classifier extended to open set; low temporal complexity | No |
| **CLOSR** — Wilkie et al., IEEE TNSM 2026 | DOI 10.1109/TNSM.2026.3652529, arXiv:2601.09902, code `github.com/jackwilkie/CLOSR` | Novel contrastive loss trained on benign + known-malign; extended to OSR, reports **OpenAUC** | No |
| **MSPL** — Martinez-Lopez et al., AAAI-25 AICS | arXiv:2501.00050 | Multi-space prototypical networks (Euclidean/Cosine/Chebyshev/Wasserstein), Polyak-averaged prototypes, episodic training | No |
| **Zero-X** — Amara korba et al. | arXiv:2407.02969 | Deep NN + OSR inside blockchain-enabled federated learning for IoV | No |
| **MI²DAS** — Lian & Guerra-Manzanares, ICISSP 2026 | arXiv:2602.23846 | Multi-layer: GMM anomaly → OSR known/unknown → RF fine-grained → incremental learning | No |
| **DOC++ framework** — Soltani et al. | arXiv:2108.09199 | Deep novelty classifier (DOC/DOC++/OpenMax/AutoSVM) + clustering + human relabelling | No |
| **Farrukh et al.** | arXiv:2309.07461 | Image-based packet representation + stacking + sub-clustering for unknowns | No |

**Use MSPL and CLOSR as the primary non-graph open-set baselines** — MSPL because
it is prototype-based (closest mechanism), CLOSR because it reports OpenAUC and
publishes code.

## 4. Methods imported from outside NIDS

| Method | ID | Role in ARGUS |
|---|---|---|
| **Evidential Deep Learning** — Sensoy et al., NeurIPS 2018 | — | Dirichlet head, evidence → uncertainty. Basis of the EPC head. |
| **Soft Medoid robust aggregation** — Geisler et al., NeurIPS 2020 | "Reliable Graph Neural Networks via Robust Aggregation" | Robust neighbourhood aggregation. **Never applied to NIDS** — this transfer is part of C2's novelty. |
| **TGAT functional time encoding** — Xu et al., ICLR 2020 | arXiv:2002.07962 | Bochner-theorem-based continuous time encoding. Source of TE3. |
| **Time2Vec** — Kazemi et al. | arXiv:1907.05321 | Model-agnostic learnable periodic time representation. Simpler TE3 variant; use this as the default. |
| **GNNExplainer / PGExplainer** | — | Post-hoc explainer baselines for `11_XAI.md`. |
| **ProvX** — Wu et al. | arXiv:2508.06073 | Counterfactual explanation for provenance GNN-IDS; explanation *necessity* metric. Source of our counterfactual metric and the detection–explanation–feedback loop idea. |

## 5. Dataset and methodology references

| Work | ID | Why cited |
|---|---|---|
| **NF-v3 datasets + temporal analysis** — Luay et al. | arXiv:2503.04404, DOI 10.1109/ACCESS.2026.3688204 | Our dataset source. **Also the source of the TE7 open problem** — see §6. |
| **NF-v2 standard feature set** — Sarhan et al., MONET 2022 | — | The 43-feature basis; cite for L0 rung of the temporal ladder. |
| **Efficient network representation + data leakage** — Friji et al., ACNS 2023 | DOI 10.1007/978-3-031-33488-7_20 | "Highlights a potential data leakage issue with classical evaluation procedures." Cite when justifying our split protocol. |
| **SoK: Pragmatic Assessment of ML-NIDS** — Apruzzese et al., EuroS&P 2023 | — | Evaluation-hygiene authority. |
| **Cross-evaluation of ML-NIDS** — Apruzzese et al., IEEE TNSM 19(4) 2022 | — | Justifies cross-dataset transfer protocol. |
| **FlowTransformer** — Manocchio et al., ESWA 2024, 241:122564 | — | Non-graph deep baseline family. |
| **Flow timeout matters** — Janati Idrissi et al., FGCS 2025 | `papers/` | Up to 8.77% F1 swing from exporter timeout choice; tree ensembles most stable. Source of the timeout-stability ablation and of the Extra Trees baseline requirement. |

## 6. The stated open problem we pick up (TE7 justification)

`arXiv:2503.04404` §5.5 applies spectrograms / time-frequency distributions to
NF-UNSW-NB15 and reports that attack classes have visually distinct
time-frequency signatures — "while DoS and Worms share some similarities, their
patterns still remain distinct… Fuzzers display a unique time-frequency
signature". But the authors state their "initial investigations have not yet
yielded definitive results", and the Conclusion says "further work is needed to
refine time-frequency-based approaches and evaluate their practicality in
real-time intrusion detection scenarios."

**TE7 (per-host spectral beaconing descriptor) is a direct, citable follow-up on
an open problem left by the dataset authors themselves.** This is a cheap and
defensible novelty hook — state it explicitly in the paper.

## 7. Sources deliberately NOT pursued

| Work | ID | Why excluded |
|---|---|---|
| KAIROS, threaTrace, CONTINUUM, PROVEX, ProvX (detection side) | various | **Provenance/host** graphs, not network flow graphs. Different problem, different data. Cite ProvX for explanation methodology only. |
| CyberGFM | arXiv:2601.05988 | Lateral-movement link prediction on authentication logs, not flow classification. |
| Q-AGNN | arXiv:2603.22365 | Quantum circuits; not reproducible on our compute. |
| GraphFaaS | arXiv:2511.10554 | Serverless *systems* contribution; orthogonal. Optionally cite in deployment discussion. |
| KnowGraph | arXiv:2410.08390 | Requires curated domain-knowledge logic rules we do not have. |
| PacketCLIP | arXiv:2503.03747 | Needs packet payloads. |
