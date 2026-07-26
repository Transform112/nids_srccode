# BUGS — diagnosis log

Full-codebase bug audit before the first real Kaggle training run (2026-07-26).
Method: manual review of the cache/training pipeline I built earlier this
session, plus five parallel agent audits — one per subsystem (data/features,
model architecture/graph construction, training/losses/config, eval/baselines/
streaming, adversarial/XAI) — each reading its assigned `docs/*.md` chapter(s)
in full and every file in its assigned code area, cross-checking doc claims
against actual code. All findings below were independently re-verified by
reading the actual source (not just trusting the agent's report) before being
logged as confirmed.

**Do not delete entries when they're fixed** — the reasoning is the point.
Add new dated sessions below rather than rewriting history.

Severity: **critical** (silently wrong results, or crashes the very next time
the affected code runs) / **high** (wrong results or a crash under a
realistic, not-contrived, sequence of steps) / **medium** (real gap, lower
likelihood or blast radius) / **low** (doc/consistency only).

---

## Session 2026-07-26 — fixed

### Cache / label-vocabulary consistency (found personally, before dispatching agents)

**1. [CRITICAL] `03_cache_graphs.py` derived label ids from a file that doesn't exist yet on a fresh Kaggle session**
`scripts/03_cache_graphs.py` tried to read `class_vocab.json` (written by
`04_train_encoder.py`) to build `label_to_id`, falling back to `None` if it
didn't exist — then unconditionally did `df["_label_id"].to_numpy()`, a
column that only exists if `label_to_id is not None`. On a fresh Kaggle
session (03 is meant to run *before* 04, in a CPU-only cache-building
session) this raises `KeyError: '_label_id'`. Worse case if a *stale* vocab
happened to exist: labels get silently scrambled, since `target_labels` is
serialized directly into every cached `.pt.gz` batch. This happened locally
by accident — a leftover `data/artifacts/cicids2018/class_vocab.json` from
an earlier dev run was used, and independently re-verified to happen to
match the current data (see Verification log). **Fix:** added
`argus.utils.io.derive_class_vocab(train_df)` as the single source of truth
(`sorted(train_df["canonical_label"].unique())`); `03_cache_graphs.py` and
`04_train_encoder.py` both call it directly from `train_features.parquet`,
independent of run order.

**2. [CRITICAL] Graph cache had no config fingerprint — a settings mismatch would silently replay the wrong time windows**
Bins are keyed only by a bare integer filename (`bin_000500.pt.gz`). The
local cache was built with `anchor_bin_seconds=10`/`window_short_seconds=10`
passed as one-off CLI `--set` overrides, never persisted to any config file
(`config/default.yaml` still said `1`). Had 04/05 run on Kaggle without the
exact same override, `bin_000500` under the default 1s scheme (seconds
500-501) is a completely different time window than under the 10s scheme
the cache was built with (seconds 5000-5010) — same filename, silently wrong
graph, wrong node set, wrong labels, no crash. **Fix:**
`src/argus/graph/cache.py::verify_or_write_cache_meta()` fingerprints
`node_granularity, anchor_bin_seconds, window_short/mid/long_seconds,
neighbour_cap, sampling, strata` (later extended to include
`features.te7_enabled/spectral_nbins/spectral_min_flows`, see #14 below)
into a `_meta.json` per split-cache dir; first write records it, every later
call must match exactly or raises. Wired into `CachedGraphSource.__init__`
(used by 04/05) and `03_cache_graphs.py::run()`. Backfilled `_meta.json`
into the already-built local cache reflecting the actual build config.
Also fixed in the same pass: `CachedGraphSource`'s on-the-fly cache-miss
path wrote **uncompressed** `.pt`, inconsistent with `03`'s gzip output and
reintroducing the disk-full failure mode from the original Kaggle
multiprocessing attempt — both paths now share `_save_compressed`/
`_load_compressed` from `argus.graph.cache`.
**Residual, accepted gap:** the fingerprint doesn't cover config keys that
only affect *feature engineering* upstream of the cache (e.g.
`data.subsample_target`) — out of scope by design; it only guards
graph-construction-time settings.

**3. [CRITICAL] `14_run_baselines.py` wrote derived columns back to the (Kaggle read-only) processed-data mount**
Same bug class as one already fixed earlier this session in
`04_train_encoder.py`, missed here: (a) `test_df.to_parquet(...)` ran
**unconditionally**, before any baseline trains — would crash on Kaggle even
with `--skip-egraphsage --skip-egatv2`; (b) `train_df.to_parquet(...)` ran
whenever a GNN baseline was enabled, which AGENT_GUIDE.md's P2 gate requires
to actually happen; (c) the `class_vocab.json` fallback wrote to
`artifact_dir` (read-only) whenever no vocab existed anywhere — a real case,
since AGENT_GUIDE.md says baselines are meant to run *before* encoder
training ("P2 is deliberately early"). **Fix:** removed both `to_parquet`
write-backs; `_gnn_baseline_source()` now takes an already-loaded DataFrame
directly instead of re-reading from disk; replaced the vocab read/write
dance with `derive_class_vocab(train_df)` (baselines train fresh models —
no checkpoint to keep vocab-aligned with). Re-ran
`14_run_baselines.py --skip-egraphsage --skip-egatv2` locally end-to-end
after the refactor to confirm no regression.

**4. [CRITICAL] `resolve_class_vocab()` resolved to the wrong directory — off-by-one path depth**
`src/argus/utils/io.py::resolve_class_vocab()` computed
`stage1_dir = Path(__file__).parents[2] / ...`. `io.py` lives at
`src/argus/utils/io.py`, three levels below repo root — `parents[2]` is
`.../src`, not repo root; needs `parents[3]`. Independently verified with
`Path.parents` directly. Every script "fixed" earlier this session to use
this helper for the exact purpose of finding the writable Stage-1 run
directory on Kaggle (06, 07, 08, 09, 11, 12, 13) inherited this bug — it
would never find `run_dir`'s copy, fall through to `artifact_dir` (also
absent on a real Kaggle run, since that copy's write is wrapped in
`try/except OSError: pass` for exactly this scenario), and raise
`FileNotFoundError` immediately after a *successful* training run. Only
"worked" locally because a stray `artifact_dir` copy happened to exist.
**Fix:** `parents[2]` → `parents[3]`. Verified by direct computation that it
now resolves to the real `results/runs/cicids2018_stage1/` directory.

**5. [LOW] `00_smoke_test.py` shared a run directory with production Kaggle runs**
Smoke-test overrides (`neighbour_cap=8` etc.) differ from production, but
wrote into the same `results/runs/cicids2018_stage1/` path. After fix #2,
this would now correctly *raise* on fingerprint mismatch instead of
corrupting anything, but it's a confusing crash for what should be an
isolated sanity check. **Fix:** added `run.out_dir=results/runs_smoketest`
to the smoke test's overrides.

**6. [MEDIUM] CICIDS2018 trained/cached at `anchor_bin_seconds=10` while docs and default config still said `{1,5}`**
`docs/07_HYPERPARAMETERS.md` documents `anchor_bin_seconds ∈ {1,5}`,
`window_short_seconds ∈ {0.5,1,2}`. The value actually needed to make one
epoch tractable on a single Kaggle GPU session (10/10) exceeds both ranges,
and was only ever a one-off CLI flag. (Aside: the documented ranges are also
internally inconsistent at their boundary — `anchor=5` has no valid
`window_short` partner in `{0.5,1,2}` given the hard rule
`anchor_bin_seconds <= window_short_seconds` — pre-existing docs issue.)
**Fix:** persisted `anchor_bin_seconds: 10` / `window_short_seconds: 10`
into `config/dataset/cicids2018.yaml`; documented the deviation and reason
in `docs/07_HYPERPARAMETERS.md` §3 and `docs/TODO.md` "Open decisions"
rather than silently rewriting the documented range. **This is a paper-scope
judgment call — see "Decisions needed" below.**

**7. [LOW] `docs/TODO.md` / `README.md` staleness**
`docs/TODO.md`'s P3 checklist and "Session 2026-07-25" log described an
older, different `03_cache_graphs.py` design (hash-named cache directories,
never actually built) that no longer matches the file that exists now.
`README.md`'s Status section said "77 tests passing" and listed several
modules (Anomal-E, post-hoc OSR, XAI baseline explainers, UNKNOWN triage,
temporal ladder, deployment reporting) as "not yet built" when they've
existed and been tested for a while. **Fix:** updated both to reflect
current reality; `README.md`'s "Known limitation" section rewritten to
describe the actual current bottleneck (graph construction, now mitigated
by the cache) instead of the stale "no caching yet, per-node loop
unvectorised" framing.

### Data pipeline (`scripts/01_prepare_data.py`, `src/argus/data/*`)

**8. [HIGH] Streaming subsample draws majority classes from only their first few chunks, not stratified across the full time range**
`01_prepare_data.py`'s Phase-2 streaming subsample takes `cls_rows.index[:n_needed]` — literally "first N rows encountered" — once a class's per-chunk arrivals exceed its remaining quota, every later chunk contributes nothing for that class. Verified directly against the already-produced `data/interim/cicids2018/cleaned.parquet`: **benign (2M rows) is drawn from only 2 of the dataset's 9 present days** (2018-02-14/15 only — 7 of 9 days entirely absent). The other flagged majority classes (`ddos_hoic`, `brute_ftp`, `ddos_loic_http`, `bot`, `dos_hulk`) turned out to be single-day *in the raw data itself* (TRAP 2 — CICIDS2018's attacks are inherently day-segregated by capture design, confirmed via AGENT_GUIDE.md's own TRAP-2 table), so their single-day appearance here is expected/unavoidable, not a subsampling artifact. **Benign specifically is the real, confirmed problem** — it's genuine background traffic recorded throughout the whole capture, and the subsample has silently destroyed most of its temporal diversity, directly undermining the documented "benign is strongly multi-modal, needs 4 sub-prototypes" design rationale and weakening any temporal-ladder finding that involves benign. **Not fixed — this affects data already produced and used to build the 3.3GB local graph cache; redoing it costs real time. Flagged as a decision, see below.** (The underlying streaming/RAM-bounded quota code itself was already a known, disclosed simplification of the doc's exact bin-stratified method — see `docs/BUGS.md` history in this file's predecessor summary — its severity just wasn't fully appreciated until this measurement.)

**9. [MEDIUM] Benign quota could be silently clobbered when benign itself qualifies as a minority class**
The minority-class loop (`quotas[c] = class_counts[c]` for `count <= minority_threshold`, correctly including benign when small) was unconditionally overwritten right after by `quotas["benign"] = benign_quota` (a smaller, floor-based number) — no guard, unlike the (unused) `data/subsample.py::stratified_temporal_subsample` which has one. Reachable exactly by the script's own documented smoke-test invocation (`--nrows 200000` with default `minority_threshold=200000`): not triggered at the real 4M-row/17.5M-benign production scale. **Fix:** only compute the floor-based benign quota when `"benign" not in minority_classes`.

**10. [MEDIUM] `remaining` budget could go negative and silently zero out every other attack class**
`remaining -= benign_quota` was never floored at 0; if minority + benign quotas together exceeded target, every non-benign majority class's quota computation (`if total_non_benign > 0 and remaining > 0`) would fail and silently assign 0 to all of them. Not triggered at production settings; reachable with a smaller `--set data.subsample_target=...`. **Fix:** `remaining = max(remaining - benign_quota, 0)`.

**11. [MEDIUM] `01_prepare_data.py` crashed on `data.subsample_target: null` ("use in full")**
`docs/02_DATASETS.md` §5.3 documents `subsample_target: null` for UNSW-NB15 (use the whole dataset, no subsampling) — but `target - minority_quota` etc. would `TypeError` on `None` immediately; the "null" case had no code path at all. **Fix:** added an explicit `if target is None: quotas = dict(class_counts)` branch (keep every row); `min(target, nrows)` guarded against `target=None` too.

**12. [MEDIUM] ton_iot/unsw_nb15/bot_iot dataset configs never set `data.subsample_target` — all three silently inherited CICIDS2018's 4M**
`docs/02_DATASETS.md` §5.3's override table specifies 4,000,000 / null / 1,000,000 for ton_iot/unsw_nb15/bot_iot respectively; none of the three `config/dataset/*.yaml` files set it, so all three got the CICIDS2018 default by coincidence (only invisible because CICIDS2018, the one dataset actually run this session, happens to match). **Fix:** added the documented values to all three YAMLs (verified all four configs now load with the correct resolved value).

### Training / losses / config (`src/argus/train/*`, `src/argus/losses/*`, `src/argus/config.py`)

**13. [CRITICAL] `DistanceThresholdHead.forward()` — the documented Stage-2 non-convergence fallback — crashes immediately**
Returns `{"logits", "p_hat", "cos_c", "d_c"}` but not `"z"`; `stage1_encoder.py`'s loss closure unconditionally does `z = outputs["z"]`. AGENT_GUIDE.md §7 row 2 names `--set head.type=distance_threshold` as the sanctioned, "do not let this block the paper" fallback if the evidential head won't converge — it was completely broken. Zero test coverage (`grep` for `distance_threshold` in `tests/` found nothing). **Fix:** added `"z": z` to the returned dict.

**14. [HIGH] `te7_enabled`/`spectral_nbins`/`spectral_min_flows` never reached `AnchorBinGraphSource`; `f_v` hardcoded to 18 everywhere**
`04_train_encoder.py`, `05_train_head.py` (shares 04's `_load_source`), `03_cache_graphs.py`, `11_run_adversarial.py`, `12_run_xai.py`, `07_eval_open_set.py`, `09_eval_transfer.py` all constructed `AnchorBinGraphSource` without passing these three `cfg.features.*` values (always using the constructor defaults `True`/`64`/`8`) and separately hardcoded `f_v=18` at the `ArgusModel` call site. This breaks the documented L3 temporal-ladder rung (`--set features.te7_enabled=false`) silently: the ablation would train with TE7 fully active regardless of the override — no crash, no warning, numbers indistinguishable from the TE7-on baseline. **Fix:** added `src/argus/graph/node_features.py::node_feature_dim(te7_enabled)` as the single source of truth for `f_v` (12, or 18 with TE7); threaded `te7_enabled`/`spectral_nbins`/`spectral_min_flows` into every `AnchorBinGraphSource(...)` call site listed above and replaced every hardcoded `f_v=18` with `node_feature_dim(cfg.features.te7_enabled)`. Extended the cache fingerprint (#2 above) to include these three keys, since they change cached node-feature tensor width/content — regenerated the local cache's `_meta.json` accordingly (no rebuild needed; the actual build already used the defaults these now make explicit).

**15. [MEDIUM] `tau_hat` was gradient-trained by AdamW every step, fighting the mandated anneal schedule**
`EPCHead.tau_hat` is a plain `nn.Parameter`; `stage2_head.py`'s `head_params = [p for p in model.head.parameters() if p.requires_grad]` included it, so AdamW updated it every one of ~9,375 steps/epoch, while `anneal_tau(epoch)` force-resets it to the documented geometric schedule only once per epoch. Between resets, gradient descent silently fought the schedule that exists specifically to prevent the Stage-2 NaN/exploding-gradient failure mode (AGENT_GUIDE.md §7 row 1). **Fix:** exclude `tau_hat` by name when building `head_params` (verified the name via `named_parameters()`).

**16. [MEDIUM] Config schema validator had no entry for the `classes` section — the one documented "raise on unknown keys" rule with a real gap**
`"classes"` is a valid top-level `SCHEMA_KEYS` entry (used by Protocol B2), but `SECTION_SCHEMA` had no matching entry, so `validate()`'s nested-key loop never checked it — a typo'd key under any dataset config's `classes:` block would pass silently, contradicting AGENT_GUIDE.md §8's explicit "must raise on unknown keys — catches typo'd overrides." **Fix:** added `"classes": {"benign", "attacks", "canonical", "families"}` to `SECTION_SCHEMA` (verified against all four `config/dataset/*.yaml` files' actual keys).

**17. [MEDIUM] `EPCHead.decide()`'s threshold fallback used Python truthiness instead of an explicit None-check**
`self.theta_unknown or 0.5` would silently replace a legitimately-calibrated `theta_unknown == 0.0` with the default `0.5`. Same pattern found and fixed in `11_run_adversarial.py`'s `theta_unknown or 0.5`. **Fix:** `self.theta_unknown if self.theta_unknown is not None else 0.5` in both places.

**18. [MEDIUM] Post-optimizer-step prototype renormalisation gated on an exact class-name string**
`train/loop.py` did `if model.head.__class__.__name__ == "EPCHead": ... post_step_normalize()`. `DistanceThresholdHead` (the fallback fixed in #13, now actually usable) also owns a `PrototypeBank` whose unit-norm invariant `cosine_to_classes` depends on, but was never renormalised — weight decay/gradient steps would silently drift it off the unit hypersphere over training. **Fix:** `hasattr(model.head, "prototype_bank")` instead of the class-name check.

**19. [MEDIUM] `PrototypeBank.diversity_loss` summed the full symmetric similarity matrix, double-counting every pair**
The spec (`Σ_{i<j}`) sums each unordered prototype pair once; the code zeroed only the diagonal and summed the whole matrix, where `sim[i,j] == sim[j,i]` — silently doubling the effective `lambda_div`. **Fix:** mask to `torch.triu(diagonal=1)` before summing.

**20. [MEDIUM] `EPCHead` computed `p_hat`/`vacuity`/`log_p_hat`/`logits` via real-space division + re-logging, not the mandated log-space subtraction**
`docs/05_ARCHITECTURE.md` §6.3 rule 2 requires `log p̂_c = log α_c − log S`. The code computed `S = exp(log_S)` then divided in real space (`p_hat = alpha / S`), and derived `log_p_hat` by re-logging that result — meanwhile `log_alpha = torch.log1p(e)` (exactly the log-space value needed) was computed and never used. Didn't overflow at the current `log_evidence_clamp=15` (verified numerically), but reintroduces the `exp()`-mediated gradient path the log-space mandate exists specifically to avoid, and would degrade if the clamp or `tau_min` are ever loosened in an ablation. **Fix:** `log_p_hat = log_alpha - log_S`, `log_vacuity = log_c - log_S`; `p_hat`/`vacuity` now derived from these via one final `exp()`; `logits` is `log_p_hat` directly (no re-log). Verified against `test_epc.py`/`test_model.py`/`test_train_loop.py`/`test_thresholds.py` — all still pass (mathematically equivalent, just no longer round-tripping through real space).

### Adversarial (`src/argus/attacks/*`, `scripts/11_run_adversarial.py`, `scripts/12_run_xai.py`)

**21. [CRITICAL] `11_run_adversarial.py`/`12_run_xai.py` compared canonical labels against the raw config string — guaranteed crash / silent corruption**
Both scripts used `cfg.classes.benign` (`"Benign"`, the raw label) where every canonicalised `class_names`/`canonical_label` value is lowercase `"benign"`. Scripts 07/08/09 already used the literal `"benign"` correctly — 11/12 were the odd ones out. `11_run_adversarial.py` raised `KeyError: 'Benign'` on `label_to_id[cfg.classes.benign]` before any attack could run; worse, `attack_classes = [c for c in class_names if c != cfg.classes.benign]` (always true, since nothing equals the raw string) would have **included benign itself as an "attack class"** had the KeyError not fired first. `12_run_xai.py` didn't crash but would have silently computed an all-NaN IG baseline and sampled every benign flow as an "attack" flow. Invisible in tests because `tests/test_adversarial.py` hand-builds `class_names = ["Benign", ...]` directly, bypassing real canonicalisation. **Fix:** introduced a `BENIGN_LABEL = "benign"` constant in both scripts, replacing every `cfg.classes.benign` comparison.

**22. [CRITICAL] `11_run_adversarial.py`/`12_run_xai.py` reconstructed `ArgusModel` without `class_counts` — cannot load a real checkpoint**
`PrototypeBank.__init__` defaults every class to 2 sub-prototypes when `class_counts=None`, but a real checkpoint (trained by 04/05, which correctly pass true per-class counts) has 1 sub-prototype for any class below `large_threshold=10,000` — a near-certainty for several of CICIDS2018's 15 classes. `load_checkpoint`'s `strict=True` state-dict load would raise a size-mismatch `RuntimeError` on any realistically-trained checkpoint. Invisible in tests (every test builds a fresh model, never loads an externally-trained checkpoint with different class_counts). **Fix:** both scripts now compute `class_counts` from `train_features.parquet` (matching what 04/05 used) and pass it through.

**23. [CRITICAL] A2 structural injection hardcoded `anchor_bin_seconds=1` when rebuilding the post-injection graph source**
`build_injected_source()` computes the target bin's start time correctly using `source.anchor_bin_seconds` (the real value), but then constructed the *returned* `AnchorBinGraphSource` with a hardcoded `anchor_bin_seconds=1`. `evaluate_flow()` then looks up the **same** `bin_id` in both the clean and injected sources — but `bin_id` is only comparable between them if `anchor_bin_seconds` agrees. This was invisible only because the (now-changed, see #6) old default *was* 1; with the real CICIDS2018 setting of 10, this would look up an unrelated 1-second time window post-injection, corrupting the entire A2 budget sweep (the headline C2 attack) and its breakdown-point measurement. **Fix:** `anchor_bin_seconds=source.anchor_bin_seconds`.

**24. [MEDIUM] A2 hardcoded the benign class id to `0` instead of using the real one**
`inj_labels = np.zeros(budget, ...)` assumed benign is always class index 0 — true today only because `derive_class_vocab`'s alphabetical sort happens to put "benign" first for all four datasets' actual class name sets; not guaranteed. **Fix:** added a `benign_class_id` parameter to `build_injected_source`/threaded from `run_a2_budget_sweep` (which already received it) instead of hardcoding.

**25. [MEDIUM] A2's benign injection pool was sampled from the test split being attacked, not train**
Gives the simulated attacker perfect knowledge of the true test-time benign distribution — a real attacker limited to observing training-time traffic wouldn't have this. **Fix:** `11_run_adversarial.py` now samples `benign_pool` from `train_features.parquet`.

**26. [MEDIUM] `attacks/constraints.py`: MTU-cap projection order left both packet-size columns above the cap**
`SHORTEST_FLOW_PKT`/`LONGEST_FLOW_PKT`: the old code took `min`/`max` of the *unclamped* inputs, then clipped only the max to 1514 — if both inputs already exceeded 1514 (plausible for uniform-MTU bulk flows under an aggressive perturbation), the result was `SHORTEST > LONGEST > 1514`, violating both the ordering and MTU constraints simultaneously. Not caught by `assert_feasible`, which didn't check either invariant. **Fix:** clip both columns to the MTU ceiling *before* taking min/max; added both invariants to `assert_feasible`.

**27. [MEDIUM] `attacks/constraints.py` never enforced Σ`NUM_PKTS_*` bins == `IN_PKTS + OUT_PKTS`**
Explicitly required by `docs/10_ADVERSARIAL.md` §1.3; the five packet-size-bin columns were independently perturbable with no link back to the totals, letting A1/A4 report "successful evasion" on a physically-impossible packet-size histogram. **Fix:** `project()` now proportionally rescales the five bins to sum to `IN_PKTS+OUT_PKTS` (falling back to dumping the total into the smallest bin if all bins were zero); `assert_feasible` checks the sum within a small per-bin rounding tolerance. Verified `tests/test_adversarial.py` (10/10) still passes with the new, stricter checks active.

**28. [CRITICAL, found by baselines agent] `models/baselines/posthoc_osr.py`: OpenMax per-class Weibull recalibration was overwritten by the last loop iteration's value for every top-alpha class**
The `for c in alpha_cls:` loop correctly computed `recal[c] = v[c] * (1.0 - w_score)` per class — then three redundant lines immediately after unconditionally reassigned `recal[alpha_cls]` using only the *last* iteration's leaked `w_score` for every one of those classes, discarding the correct per-class values for all but one. Weakens the OpenMax baseline unfairly, inflating ARGUS's relative OpenAUC margin in exactly the comparison AGENT_GUIDE.md §10.2 says must be fair. **Fix:** deleted the three redundant lines; the loop body already did the correct assignment.

### Eval / selective prediction / streaming (`src/argus/eval/*`, `src/argus/streaming/*`, `scripts/07`, `scripts/09`)

**29. [CRITICAL] `eval/selective.py::aurc()`/`e_aurc()` crash outright on the installed numpy, and were sign-inverted even if fixed naively**
`np.trapz` was removed in numpy 2.x (installed: 2.4.6, permitted by `pyproject.toml`'s `numpy>=1.26,<3`) — confirmed by executing it (`AttributeError`). Independently, `risk_coverage_curve` returns `coverage` **descending** (100%→0%); integrating a descending x-array with `trapezoid` returns the *negative* of the true area (confirmed numerically: constant risk=0.3 over the full range returned -0.3, not +0.3), which would flip "lower is better" to mean the opposite once the crash was naively patched. No test exercises `selective.py` (`grep` found nothing in `tests/`), so this was invisible. **Fix:** `np.trapz` → `np.trapezoid`; both functions now sort by `coverage` (`np.argsort`) before integrating, correct regardless of input order.

**30. [HIGH] `07_eval_open_set.py`: unknown-detection threshold computed from the test split itself, with an inverted percentile, and a fake one-hot score matrix**
Three compounding bugs in one function: (a) `theta = np.percentile(unknown_scores, 100 * target_false_unknown_rate)` used the **test** split's own scores — never a validation split — violating standing rule 6 (thresholds selected on validation only); (b) with `unknown_score = 1-max(prob)` (higher = more unknown) and the decision rule `unknown_score > theta → UNKNOWN`, hitting a 5% false-unknown rate needs the **95th** percentile of *known-only* samples, not the 5th percentile of the whole (known+unknown) test set — the direction was backwards, so ~95% of all flows would have been flagged UNKNOWN; (c) `open_auc()` was called with a synthetic one-hot `score_matrix` (1.0 at the argmax, 0 elsewhere) instead of the real softmax probabilities `_infer()` already computed and silently discarded, collapsing the AUROC to a single fixed operating point. Together these would have made the C1 open-set headline table (T3) both statistically invalid (threshold leaked from the scored data) and numerically close to meaningless (~95% UNKNOWN rate). **Fix:** now loads `val.parquet` alongside `test.parquet`, applies the same per-holdout "mark holdout classes UNKNOWN" transform to both, runs inference on validation first to calibrate `theta` at the 95th percentile of *known-only* validation scores, then scores test with that fixed threshold; `open_auc` now receives the real per-class probability matrix. Also fixed the same `f_v=18`/missing `te7_enabled` threading and missing `class_counts` bugs as #14/#22 in this same file.

**31. [HIGH] `09_eval_transfer.py`: same threshold-from-test-not-validation pattern, plus a positional-alignment bug in both the old and new known/unknown masking**
The unknown threshold was `np.percentile(unknown_scores, 95)` computed from the target test split being scored (percentile direction happened to be correct here, unlike #30, but the data-source violation of standing rule 6 was the same). While implementing the fix I found a second, independent, pre-existing bug: the *known-mask* was read back as `test_df["_is_known"].values[:len(y_true)]` — a **positional slice of the original, unsorted dataframe** — but `_build_source` sorts rows by timestamp internally before bins are built, so the inference output (`y_true`, `unknown_scores`, ...) is in a different row order than `test_df`. The two would silently misalign for any target dataset whose test split isn't already globally time-sorted on disk. A third, unconditional bug in the same block (`truly_unknown = ~known_mask.values[...]`, calling `.values` on an already-plain-ndarray) would have raised `AttributeError` on every single run that reached it, regardless of any of the above. **Fix:** extracted a shared `_run_inference()` helper; threshold now calibrated on `val_features.parquet` (loaded via the target dataset's own validation split) using known/unknown status derived from the *returned* family-id array (`np.isin(y_true, known_family_ids)`) — guaranteed aligned with the inference output — rather than re-indexing the input dataframe by position. Also fixed the same `f_v=18`/`te7_enabled`/`class_counts` gaps as #14/#22.

**32. [MEDIUM] `streaming/detector.py`: per-flow latency measurement excluded the model's forward pass**
`elapsed_ms` was snapshotted immediately after `build_bin_batch()`, before `model_inputs_from_batch`/`self.model(*inputs)` ran — every `Verdict.latency_ms` therefore excluded the entire encoder+EPC-head forward pass, while `measure_streaming_throughput`'s own wall-clock timing (wraps the whole `push()` call) correctly included it — making throughput and reported per-flow latency internally inconsistent in the same deployment report, understating ARGUS's real latency exactly where docs/08_EVALUATION.md §4.6 says "if ARGUS is slower, report honestly." **Fix:** moved the `elapsed_ms` snapshot to after the forward pass (the early-return "no targets" branch already measured correctly, since there's no forward pass to wait for there).

---

## Found this session, not yet fixed — decisions needed

These either require a scope/priority call only you can make, or are
substantial-enough rewrites that attempting them in the same pass as ~30
other fixes risked rushing core architecture. Ranked by how much they affect
what you're about to spend Kaggle GPU hours on.

### 1. Local subsample's benign class lacks temporal diversity (see fixed-bug #8 above)
The 4M-row subsample already used to build splits, features, and the 3.3GB
graph cache has benign drawn from only 2 of 9 present days. The *code* bug is
fixed (future runs will stratify properly), but the *data already on disk*
is not affected retroactively.
**Your call:** (a) accept it and note as a limitation, (b) rebuild
`01_prepare_data.py` → `02_build_splits.py` → `03_fit_features.py` →
`03_cache_graphs.py` locally with the fix (another ~1-1.5h of local compute,
matching what was already spent), or (c) something in between (e.g. only
re-stratify benign, keep everything else). I did not rebuild without asking,
since it invalidates already-completed local work.

### 2. Per-node GRU memory (TE6) is fully inert — never receives gradient, ever
Confirmed independently by two separate agents (architecture + training
audits), from different code paths: `model_inputs_from_batch()` never
includes a `memory` argument in the tuple passed to `model(*inputs)`, so
`ArgusModel.forward`'s `memory` parameter is always `None`; the internal
`NodeMemory` GRUCell (~99k params per the doc's budget) is constructed but
its `forward()` is never called anywhere in the real training path. The
`memory_state` dict `train/loop.py` builds and "clears" every `bptt_chunk`
bins is discarded, never threaded anywhere. A `te6_enabled=true` vs `false`
ablation would currently be bit-identical. This is a core, named part of the
architecture (AGENT_GUIDE.md §5: "per-node GRU memory with dropout and
eviction") and directly affects any temporal-memory contribution claim (C3).
**Not fixed** — wiring a real per-node memory tensor through the BPTT loop
correctly (build before `forward_scale`, thread the updated dict back out,
`.detach()` per-value at chunk boundaries rather than `.clear()`, feed the
true aggregate not the post-residual state) is a genuine architecture change
that deserves dedicated attention, not a rushed patch at the tail of this
session. **Recommend fixing before spending Kaggle GPU time**, since any
checkpoint trained before this fix would need retraining anyway once it's
addressed.

### 3. Time-decayed attention decay is never normalised by scale duration
`decay = -(lambda * dt)` uses raw seconds directly; the doc (§3.4) explicitly
requires `Δt` normalised by scale duration fed into *both* Time2Vec and the
decay term, specifically because "a shared λ cannot serve both" short and
long scales. At the long scale (300s), a neighbour 100s old gets
`decay≈-1109` for some heads — softmax underflows every edge but the single
most recent to ~0 weight, degenerating mid/long-scale attention to a hard
recency pick. This directly undermines the multi-scale design exactly where
it matters for C3, and is invisible to tests because the test fixtures reuse
one small `edge_dt` tensor for all three scales. **Not fixed** — same
reasoning as #2 (needs the per-scale `lambda_hat` re-initialisation question
resolved properly, not patched under time pressure).

### 4. `soft_medoid` aggregation is silently discarded whenever multi-aggregator readout is on (the default)
`RobustAggregator.forward` computes `soft_medoid(...)` when configured, then
unconditionally returns `self.multi(...)` instead whenever
`multi_aggregator=true` (the actual default) — and `MultiAggregator` is
hardcoded to always use `trimmed_mean`/`trimmed_std` regardless of which
aggregator was requested. Setting `model.aggregation=soft_medoid` with
default config currently produces numerically identical output to
`trimmed`. This breaks AGENT_GUIDE.md's own aggregation ablation ("Run all
three in the C2 ablation. Expect soft_medoid to be the most robust") — the
ablation arm is dead. **Not fixed.**

### 5. `recency_stratified` sampling stratifies by array position (count), not by time
`sample_neighbours()` only receives `dst_ids`, no timestamps, so its 4
"strata" are equal-*count* slices of the time-sorted array, not the
documented "4 equal time sub-intervals." A large enough flood of injected
flows can spill across multiple count-based strata and capture well over the
intended 25% share, partially reproducing the "most-recent-K" attack pattern
AGENT_GUIDE.md §5.1(4) explicitly calls out as voiding claim C2. The
existing test encodes the same count-based assumption, so it can't catch
this. **Not fixed** — requires threading timestamps into the sampler
(signature change cascades to callers) and implementing the documented
shortfall-redistribution rule.

### 6. Node features are blind to a node's role as a flood/scan *victim*
`byte_vol`, `short_scale_count`/`mid_scale_count`, and `gaps` in
`compute_node_features_block1` are only accumulated on the *source* side of
each flow, never the destination side — even though the spec defines
`byte_volume` as "total bytes across incident flows" generally. A node that
is purely the target of a DDoS/scan flood gets these features ≈0, i.e. the
features most relevant to detecting that exact attack pattern are silently
dead for the victim. Separately, `distinct_dst_ports` isn't port data at
all — it adds destination *node ids* to a set indexed by that same node,
converging to a size-1 set on any incoming flow (the code's own comment
admits "placeholder"). **Not fixed** — needs a real port array threaded in
and dual-sided (src+dst) accumulation.

### 7. Run registry (`results/runs/registry.jsonl`) used by ~1 of 13 scripts; no per-epoch checkpoint/resume in 04/05
`src/argus/utils/registry.py` is fully implemented and tested, but almost
nothing calls it — `docs/12_IMPLEMENTATION_PLAN.md` §4.5 and AGENT_GUIDE.md
§7 row 10 both describe per-run registry entries + per-epoch checkpointing
as *the* mitigation for "Kaggle session dies mid-run," and neither is wired
up: `04_train_encoder.py`/`05_train_head.py` only call `save_checkpoint`
once, after all epochs finish (and even then construct a **fresh,
never-stepped** optimizer just to fill the checkpoint's optimizer field —
the real trained optimizer state is discarded); `train.checkpoint_every_epoch`
is declared in the config schema but never read anywhere. Given your actual
12h-Kaggle-session constraint, a session dying at epoch 20/30 currently
loses everything. **Not fixed** — this is a real, scoped feature addition
(per-epoch save inside `train_stage1`/`train_stage2`, a `--resume` flag,
skip-if-already-registered logic in the longer-running scripts like
`10_run_temporal_ladder.py`/`11_run_adversarial.py`), not a quick patch.
**Recommend prioritizing this highly given your Kaggle session-length
constraint** — the cost of hitting this without it is a fully lost session.

### 8. Gates G0–G7 (`train/gates.py`) are fully implemented and never called
AGENT_GUIDE.md's own fresh-agent prompt (§12) says "Run gate G0 — the
1,000-sample overfit capacity check — before any full training run, and
every time the architecture changes" — this is the cheapest, highest-value
safety net for exactly the failure mode you already hit once this session
(silent pipeline bugs burning hours before anyone notices). No script
invokes any gate. **Recommend running G0 manually (or wiring a
`--gate0-only` preflight into `04_train_encoder.py`) before starting the
real Stage-1 Kaggle run**, given how cheap it is relative to a wasted
GPU session.

### 9. C4/F7 (XAI ↔ adversarial linkage) is unimplemented, not merely "not yet run"
`docs/11_XAI.md`'s edge-level attribution (`attr_edge`) and the
`injected_mass_fraction` metric for figure F7 don't exist anywhere in
`src/` — `evidence_attrib.py` only implements embedding-level and
Integrated-Gradients feature-level attribution. `12_run_xai.py` never
imports anything from `attacks/a2_structural_injection.py`; the two
"headline" modules for C2 and C4 were built and tested in complete isolation
from each other. **Not fixed** — this is new development (implement
`attr_edge`, expose an injected-edge mask on `AnchorBinGraphSource`, wire
`12_run_xai.py` to call A2 and compute the fraction), not a bug fix.

### 10. Open-set/few-shot eval (07/08) score checkpoints that already saw the "held-out" classes during training
There's no mechanism to actually train a model with a given holdout set's
classes excluded and evaluate that specific checkpoint — `02_build_splits.py
--protocol B` only materializes the *first* holdout set, and 07/08 each load
one generic Stage-2 checkpoint (trained on all classes) and relabel
different classes as UNKNOWN post hoc for each of 5 sampled holdout sets.
The model saw every "held-out" class's flows during training, so OpenAUC/
unknown-TPR (T3) and few-shot zero-forgetting numbers (T4) will look better
than genuine zero-shot generalisation would produce. **Not fixed** — needs
either a real class-exclusion training path (5 distinct checkpoints) or an
explicit, disclosed descoping of how strong the C1 claim can be stated.

### 11. Other confirmed-real, lower-urgency findings (not on your critical path to the next Kaggle run)
- `08_eval_few_shot.py`'s "zero parameter change" check is a raw `numel()`
  delta (always positive by construction when new rows are added) — the
  rigorous state-dict-diff check exists in `tests/test_streaming.py` but was
  never ported into the script that generates the paper's T4 numbers.
- `15_make_tables.py` has no flattener for `06`/`07`/`08`/`09`'s output —
  it silently omits exactly the claim-critical T2/T3/T4/T7 tables while
  reporting success.
- `PrototypeBank.register_class` reassigns `self.bank` to a brand-new
  `nn.Parameter` with `requires_grad=False` for the *entire* bank (not just
  the new rows) — harmless today since nothing calls it mid-training, but a
  footgun for any future live-registration-during-training path.
- Multi-aggregator readout's third term concatenates the bare scalar
  `a_scale` where `docs/05_ARCHITECTURE.md` §3.5b specifies the
  elementwise vector product `a_scale · a_robust` — internally consistent
  (the `nn.Linear` is sized to match) but not what the spec computes, a
  plausible contributing factor if the P3 E-GraphSAGE-parity gate
  underperforms.
- `docs/09_TEMPORAL_STUDY.md`/`docs/11_XAI.md` cite stale runner filenames
  (`09_run_temporal_ladder.py`/`11_run_xai.py`; actual: `10_`/`12_`).
- `features/pipeline.py`'s `split_hash_` provenance field is always `None`
  — the "verify the pipeline was fit on train only" safety net
  (`docs/03_FEATURE_ENGINEERING.md` §1, described as a hard error) is a
  complete no-op. No live leak today (`03_fit_features.py`'s actual
  fit/transform usage is correct), but the regression-safety-net doesn't
  exist.
- `data/audit.py::near_duplicate_rate()` is implemented but never called
  from `02_build_splits.py`, so `split_audit.json` is missing a disclosure
  `docs/02_DATASETS.md` §4.2 calls mandatory.
- `docs/12_IMPLEMENTATION_PLAN.md`'s illustrative directory tree
  (`data/processed/<dataset>/{train,val,test}/` as subdirectories,
  `data/artifacts/<run_id>/`) doesn't match the actual, consistently-used
  on-disk layout (flat `{split}_features.parquet` files,
  `data/artifacts/<dataset>/`) — doc-only staleness, every script agrees
  with the real convention.

---

## Session 2026-07-26, continued — architecture fixes + subsample re-stratification

Per your explicit decision (redo the subsample now; fix the 5 architecture
gaps before Kaggle), completed all of the following. All are additionally
covered by the full local pipeline rebuild queued after these changes (see
Verification log).

**33. [HIGH, FIXED] Streaming subsample's core sampling bug (root cause of #8 above)**
The fix for #8 only corrected quota *arithmetic*; the actual "take first N
rows encountered" chronological-bias logic was untouched — simply redoing
the pipeline would have reproduced the exact same benign-day-collapse.
Implemented real per-class **time**-bin quotas: Phase 1 now also tracks each
class's observed `(min, max)` timestamp; Phase 2 buckets each class's quota
across `time_bins_for_subsample` (config, default 100) equal-*width* time
bins (an approximation of the doc's equal-*count* bins, tractable without a
second full-data pass) with a vectorized `groupby().cumcount()` rank-within-bin
instead of a per-row Python loop. Verified on a synthetic 9-day/450K-row
benign-like class with a 20K quota: old logic keeps 100% from day 0; new
logic keeps all 9 days at ~2,200-2,400 each.

**34. [HIGH, FIXED] `soft_medoid` aggregation silently discarded under `multi_aggregator=true`; multi-aggregator's third term was a scalar, not the documented vector product**
`RobustAggregator.forward()` now passes its actual computed aggregate (`agg`
— whichever of `trimmed_mean`/`soft_medoid` was configured) into
`MultiAggregator.forward(msgs, weights, a_robust)`, which builds on top of
it instead of silently recomputing `trimmed_mean` internally regardless of
configuration. Also fixed the third concatenated term from the bare scalar
`a_scale` to the documented elementwise vector `a_scale * a_robust`
(`docs/05_ARCHITECTURE.md` §3.5b), resizing `project` from `2*d_h+1` to
`3*d_h` accordingly. `MultiAggregator` has no external callers besides
`RobustAggregator` (verified by grep), so this signature change is fully
contained. `tests/test_aggregation.py` (6/6) still passes.

**35. [HIGH, FIXED] Time-decayed attention decay used raw-second `dt`, not normalised by scale duration**
`SRTEGLayer.forward()`'s inline attention block used `dt = neigh_dt[v, m]`
directly in `decay = -(lambdas * dt)`. Root cause: `SRTEGLayer` (and its
`TimeDecayedAttention`) is always constructed with `scale_duration=1.0`
(weights are shared across S/M/L scales, so there's one shared `lambda_hat`,
correctly initialised assuming a **normalised** `[0,1]`-ish Δt range) —
`layer.attn.scale_duration = scale_duration` is reassigned per real scale in
`forward_scale()`, but nothing read that attribute, so decay always saw raw
seconds against a lambda calibrated for normalised time. Concretely, this
made mid/long-scale attention degenerate to a near-hard most-recent-edge
pick (a neighbour 100s old got `decay≈-1109` for some heads). **Fix:**
`dt = neigh_dt[v, m] / self.attn.scale_duration` — one line, makes the
already-correct init calibration and the already-being-set (but previously
unread) `scale_duration` attribute both actually matter. `tests/test_model.py`,
`test_graph.py`, `test_time_encoding.py`, `test_aggregation.py` (18/18) pass.

**36. [HIGH, FIXED] `recency_stratified` sampling stratified by array position (count), not by time**
`sample_neighbours_recency_stratified` only received `dst_ids` — no
timestamps — so its "4 strata" were equal-*count* slices of whatever
edges were present, not the documented "4 equal time sub-intervals of the
window." A burst large enough to span multiple count-quantiles could claim
more than 1/Q of a node's sampled neighbourhood despite arriving within a
narrow instant, weakening the anti-burst-injection guarantee C2 relies on.
**Fix:** threaded `age_seconds` (seconds before window end — `batching.py`
already computed this, just never passed it on) and the scale's *declared*
`window_seconds` (attacker-independent, not the observed edge span) through
`_cap_and_extract` → `sample_neighbours` → `sample_neighbours_recency_stratified`;
buckets are now real time sub-intervals of the fixed window, with the
documented shortfall-redistribution to the most-recent stratum. Falls back
to the old count-based behaviour if the new params are omitted (backward
compatible for any other caller). Verified with a synthetic burst-injection
scenario (200 background edges spread across a 300s window + 100 attacker
edges in the last 2 seconds): attacker captured 18.75% of the sample
(bounded near 1/Q), confined to stratum 0. `tests/test_graph.py` (6/6) pass.

**37. [HIGH, FIXED] Node features blind to a node's role as a flood/scan *victim***
`byte_vol`, `short_scale_count`/`mid_scale_count`, and `gaps` in
`compute_node_features_block1` were accumulated only on the edge's *source*
side, never the destination side, despite the spec defining `byte_volume`
as "total bytes across **incident** flows" generally. A pure DDoS/scan
*target* got these features ≈0 — exactly the features most relevant to
detecting that attack pattern, silently dead for the victim. **Fix:**
accumulate over both `src` and `dst` per edge (deduped for the rare
`src==dst` self-loop case). Verified with a synthetic 50-attacker,
single-victim flood: victim's `byte_volume` and `short_scale_burst` went
from a hardcoded 0.0 to 0.62 and 30.0 respectively. **Not fixed in the same
pass** (larger, separate blast radius): `distinct_dst_ports` still isn't
real port data (adds destination *node ids* to a set, not ports) — its own
code comment already discloses this as a placeholder. Fixing it properly
needs raw per-edge port values threaded through the entire graph
construction pipeline (`AnchorBinGraphSource` doesn't currently carry raw
ports at all, only transformed `edge_features`), a much wider change than
the dual-sided accumulation fix. Flagged, not attempted, given time
constraints — see "Decisions needed" below (superseded — moved to fixed
list above from the prior pass, port-data gap remains open).

**38. [FIXED, with an honest caveat] Per-node GRU memory (TE6) was completely inert; now genuinely influences predictions, but its own weights still can't receive gradient under the current training loop**
This was the largest of the five architecture items and took three attempts
to get right — recorded in full because the reasoning matters for whoever
revisits it:
- Threaded a real `node_ids` tensor through `AnchorBinGraphSource.build_bin_batch()`'s
  return dict (global node id per row of `node_feat` — required because a
  bin's local node-index space is remapped independently every bin, so it
  cannot key a cross-bin memory dict).
- Fixed `ArgusModel.forward()`'s `mem_tensor` construction, which was
  hardcoded to `None` regardless of input (self-admitted in-code: "memory
  not used here"), and its memory-update call, which used
  `torch.arange(node_feat.shape[0])` (local indices) instead of the real
  global `node_ids` — meaning even if the dead code above had been live, it
  would have mixed up different hosts' memory across bins.
- Threaded `memory_state` + `node_ids` through `train/loop.py::run_epoch()`'s
  `model(...)` call (previously never passed at all, so `memory` was always
  `None` regardless of the `memory_state` dict the loop maintained).
- **First attempt** (detach `memory_state` every bin after backward): fixed
  nothing — crashed immediately with "Trying to backward through the graph
  a second time," because this loop calls `optimizer.step()` **every bin**
  (not accumulated over `bptt_chunk` bins), and detaching *after* backward
  was already too late — the shared-ancestor graph between bin *k*'s memory
  write and bin *k*'s own loss was already double-touched.
- **Second attempt** (detach the GRU's own inputs — previous hidden state
  and the long-scale aggregate — inside `NodeMemory.forward`/`ArgusModel.forward`,
  leaving the *read* of stored memory un-detached so gradient could reach
  the GRU's weights): fixed the graph-freeing crash, but hit a *second*,
  more fundamental one — "modified by an inplace operation" — because
  `optimizer.step()` for bin *k-1* mutates the GRU's weight tensors in
  place, and bin *k*'s backward pass (reading a value the GRU produced
  during bin *k-1*, before that step) then finds those weights at the wrong
  version. **This is not fixable by placing detach() calls more cleverly —
  it's a structural incompatibility**: no tensor connected to a parameter
  can survive across an `optimizer.step()` on that parameter. Genuine
  multi-bin backprop-through-time into the GRU's weights requires
  restructuring `run_epoch()` to accumulate loss across a `bptt_chunk` of
  bins before a single `backward()`+`step()`, not per-bin stepping.
- **Final, shipped fix:** detach the *read* too (`mem_tensor` used in
  `x = x + mem_tensor` is `.detach()`ed). This means the GRU's own weights
  get no gradient from this recurrent connection and stay at their random
  initial values — but memory's *values* are now real, computed every bin
  from the true long-scale aggregate, and genuinely change what the encoder
  sees. Empirically verified: `te6_enabled=true` vs `false` now produce
  measurably different loss trajectories on identical data/seed (previously
  bit-identical, per two independent agents' findings) — the "provably
  inert regardless of the ablation flag" bug is fixed. The "GRU is a fixed
  random projection, not a learned one" limitation is new information this
  session surfaced and is **not** something a config change or a smaller
  patch can close — see "Decisions needed" below.

All of `tests/test_graph.py`, `test_model.py`, `test_train_loop.py`,
`test_thresholds.py`, `test_streaming.py`, `test_channel_penalty.py`, and
the full 97-test suite pass after every fix in this section (re-run after
each one, not batched).

---

## Session 2026-07-26, part 3 — the four deferred items + learned TE6 memory

Per your explicit instruction ("fix these 4 issues and also try to fix the
issue of gru weights ... kaggle has computational limitations ... we can do
it in parts, running only one script in a session"), this pass closed every
remaining item from "Decisions needed". See docs/KAGGLE_PLAN.md for the
session-by-session execution plan this enables.

**39. [FIXED] TE6 GRU weights now receive real gradient — `run_epoch` restructured to true 8-bin truncated BPTT**
Supersedes #38's "GRU is a fixed random projection" caveat. The structural
incompatibility identified there (no tensor connected to a parameter survives
an `optimizer.step()` on that parameter) was resolved the only correct way:
`train/loop.py::run_epoch` now accumulates per-bin losses across a
`bptt_chunk` (=8, config) window and performs ONE `backward()` +
`clip_grad_norm_` + `step()` + prototype renormalisation per chunk — the
loop structure docs/06_TRAINING.md §5 specified all along. The `.detach()`
calls #38 added in `NodeMemory.forward` (GRU inputs) and `ArgusModel.forward`
(memory read) are removed; truncation now happens exactly once, at chunk
boundaries, by detaching every tensor in `memory_state` after the chunk's
step. Within a chunk, gradient flows from later bins' losses through the
recurrence into the GRU's weights and earlier bins' encoders — genuine BPTT.
The channel penalty's per-bin double-backward still works (its
`autograd.grad(..., create_graph=True, retain_graph=True)` keeps the bin's
graph alive until the chunk backward). Loss reported per chunk is the mean
over its bins, keeping gradient magnitude comparable to the old per-bin
stepping. **Verified:** `model.memory.gru.weight_ih` changes during training
(max |Δ| ≈ 5e-3 over one 40-bin epoch; was provably frozen before); two
epochs + eval run crash-free; full test suite passes.

**40. [HIGH, FIXED] `AnchorBinGraphSource._prev_active_nodes` leaked across epochs — uncached training silently diverged from cached training**
Found while proving the resume test's trajectory equality: with everything
else (weights, optimizer state, all four RNG streams) verified bit-identical
at the epoch boundary, resumed runs still diverged from uninterrupted runs at
the first resumed epoch. Bisected to `node_feat` of epoch k>0's first bins:
the "newly active node" feature reads `_prev_active_nodes`, which was never
reset between epoch passes — so epoch 2's first bin saw epoch 1's LAST bin
as "the previous bin" (an artifact of the epoch loop, not real temporality).
This also meant uncached training (whose source object persists across
epochs) computed different features than cached training (cache built once,
fresh pass) for the same nominal data. **Fix:**
`AnchorBinGraphSource.reset_epoch_state()` (+ `CachedGraphSource` delegate),
called at the top of `run_epoch`. The streaming detector is untouched — it
genuinely wants cross-bin continuity. **Verified:** the kill-and-resume test
now reproduces the uninterrupted trajectory within the doc's 1e-3 tolerance
(previously off by ~3e-2 via this artifact).

**41. [FIXED] Per-epoch checkpoint/resume + run registry (former decision item "run registry / checkpointing")**
- `train_stage1`/`train_stage2` now save `stage{1,2}_ckpt_last.pt` every
  `train.checkpoint_every_epoch` epochs (the previously-declared-but-never-read
  config key) with the REAL optimizer state, epoch, best-val/patience/history,
  and all four RNG streams; `stage{1,2}_ckpt_best.pt` on val improvement.
  `save_checkpoint` writes atomically (tmp + `os.replace`) so a session dying
  mid-save can't corrupt the only resumable checkpoint.
- `--resume` on 04/05 continues from `ckpt_last` (epoch, early-stop state and
  RNG restored); G0 preflight is skipped on resume.
- On return the model carries the BEST epoch's weights (early stopping with
  patience deliberately trains past the best epoch; previously the final —
  known-worse — weights were what 04/05 saved to `stage{1,2}_final.pt`).
  `stage{1,2}_final.pt` also now stores the real, stepped optimizer (was: a
  fresh never-stepped AdamW constructed inline at save time).
- 04/05 register completed runs in `results/runs/registry.jsonl`
  (`RunRegistry` was fully implemented but called by ~nothing) and skip
  already-registered runs unless `--force` — the doc's "never repeat a
  completed run across Kaggle sessions" behaviour.
- TE6 memory is deliberately NOT in the checkpoint: it re-initialises to
  zeros at split start (docs/05_ARCHITECTURE.md §4), which happens at every
  epoch top, so an epoch-boundary checkpoint has no memory state to carry.
- **Verified:** new `tests/test_checkpoint_resume.py` — kill-after-2-epochs /
  resume-to-4 reproduces the uninterrupted 4-epoch trajectory (≤1e-3, per
  docs/06_TRAINING.md §6.2), best-weight restore matches `ckpt_best`
  bitwise, checkpointed optimizer state is non-empty/stepped.

**42. [FIXED] Gates G0–G7 wired in (former decision item "gates never invoked")**
- **G0** (1,000-flow overfit capacity, §8.1): `04_train_encoder.py` runs it by
  default before full training — class-stratified subset, ALL regularisation
  disabled via config overrides, throwaway model trained up to 60 subset
  epochs; hard-fails the run (RuntimeError) if `train_acc < 0.99` so a broken
  architecture can't burn a 12h GPU session. `--skip-gate0` opts out; skipped
  automatically on `--resume`; smoke test opts out explicitly.
- **G1** (epoch 5), **G7** (every epoch), **G5** (channel ratio, every epoch,
  via a new `ChannelPenaltyLoss.last_ratio` side-channel), **G6** (non-finite
  loss → hard RuntimeError, every epoch) inside `train_stage1`; **G2/G2b**
  (prototype geometry, via new `gates.prototype_gate_stats`) after Stage 1;
  **G6** per epoch + **G3/G4** (known vacuity / synthetic-unknown vacuity,
  via new `stage2_head.evaluate_vacuity`) after Stage 2.
- All results append to `<run_dir>/gates_report.json` via `gates.record_gate`.
- Disclosed proxies: G1/G7 use the val-accuracy proxy this trainer already
  early-stops on (doc says macro-F1 — pre-existing deviation, now recorded);
  G3/G4 skip cleanly under the `distance_threshold` fallback head (no
  evidence path to gate).

**43. [FIXED] C4/F7 XAI↔adversarial linkage implemented (former decision item 9)**
- `AnchorBinGraphSource` accepts and threads an `is_injected` provenance mask
  (reordered with the time sort — injected rows interleave, they are NOT a
  contiguous suffix) through `_cap_and_extract`'s sampling into a per-scale
  `batch["edge_injected"]` bool tensor. `build_injected_source` marks its
  synthetic rows.
- `SRTEGLayer` gained a `record_messages` side-channel (mirrors
  `record_attention`): retains the [N,K,d_h] message tensor + grad.
- `evidence_attrib.py` gained `edge_attribution_for_victim` (attr_edge =
  α·‖∂score/∂m‖₂ over the victim's long-scale neighbourhood, normalised to
  sum 1 — the doc's per-neighbour Jacobian norm is replaced by the
  predicted-class-logit saliency, a disclosed standard reduction),
  `victim_slot_flags` (dense-slot alignment mirroring `_edges_to_dense`),
  and `injected_mass_fraction`.
- `12_run_xai.py` now runs the full F7 sweep: correctly-detected malicious
  flows × budgets {4,8,16,32} × aggregation {mean, trimmed, soft_medoid}
  (switched at inference on the same weights — a mechanism comparison, noted
  in the output), writing per-row and aggregate results into
  `xai_results.json["f7_injected_mass"]`. Benign injection pool now sampled
  from TRAIN (same attacker-knowledge rationale as #25).
- **Verified:** new `tests/test_xai_adversarial_linkage.py` (3 tests): mask
  survives the sort and reaches the right scale/victim edges; clean sources
  are all-False; attr_edge normalises to 1 with the fraction in [0,1] and
  side-channels reset after use.

**44. [FIXED] Open-set/few-shot eval no longer scores checkpoints that saw the "held-out" classes (former decision item 10)**
The full per-holdout training path now exists:
- `02_build_splits.py --protocol B --holdout-index i` writes splits to
  `data/processed/<ds>/holdout_b<i>/` (previously protocol B silently
  OVERWROTE the Protocol-A train/val/test parquets) and persists ALL sampled
  holdout sets to `holdout_sets.json` — fixing a second, subtle bug: 02
  sampled holdouts from `unique()` (first-seen) order while 07 sampled from
  the alphabetised vocab, so the same seed produced DIFFERENT holdout sets in
  the two scripts; the persisted file is now the single source of truth, and
  the input list is sorted before sampling.
- `03_fit_features.py --holdout-index` refits the feature pipeline on the
  holdout-excluded train split (fit-on-train-only applies per holdout);
  `03_cache_graphs.py`/`04`/`05 --holdout-index` build/train into
  `<ds>_stage1_b<i>`/`<ds>_stage2_b<i>` with suffixed registry run-ids.
- `07_eval_open_set.py` rewritten: reads `holdout_sets.json`, loads each
  holdout's OWN checkpoint/vocab/features, verifies the vocab contains no
  held-out class (refuses to report leaked numbers otherwise), calibrates
  theta on that holdout's validation split, and aggregates mean±std over
  however many per-holdout checkpoints exist (skipping missing ones loudly).
  The post-hoc-relabel path is gone entirely.
- `08_eval_few_shot.py` rewritten on the same per-holdout basis, also fixing
  three pre-existing internal bugs: (a) registration embeddings were taken
  from arbitrary flows of the FIRST TEST BIN, not flows of the class being
  registered — now the class's n chronologically-earliest test rows are
  embedded (and excluded from post-registration scoring); (b) the "zero
  parameter change" check was a `numel()` delta (always nonzero by
  construction) — now a bitwise state-dict diff over all pre-existing
  tensors including the bank's original rows; (c) registrations accumulated
  across n-shot settings on one model instance — now a fresh checkpoint is
  loaded per (holdout, n) configuration. `protocol_b_split` preserves the
  true class in a new `label_pre_holdout` column (relabelling to "UNKNOWN"
  had erased the identity T4's per-class F1 needs), threaded through
  `03_fit_features.py`.

**Remaining known deviations (not in scope of this pass, disclosed):**
- LR scheduler (cosine + 3-epoch warmup, docs/06_TRAINING.md §5) is still
  not implemented — training runs at constant LR.
- DropEdge (regularisation table §5.1) exists only as a config key.
- `batch_anchor_bins=64` batching and AMP bf16 are not implemented (loop is
  one-bin-at-a-time, fp32).
- Stage-1/2 early stopping and gates G1/G7 use val accuracy, not the
  documented macro-F1 / OpenAUC.
- `distinct_dst_ports` node feature remains a disclosed placeholder (#37).

---

## Session 2026-07-26, part 4 — the feature-scale collapse behind G0 failure

The part-3 fixes added a G0 capacity preflight (04 refuses to spend GPU hours
if the model can't memorise ~1000 flows). On the first real Kaggle validation
kernel it did its job: **G0 failed, train_acc ≈ 0.06** (the model collapsed to
predicting one class). That is not a Kaggle artifact — it reproduced locally,
and bisection found it was never the BPTT change (fails identically at
`bptt_chunk=1`). It was two independent *unscaled-feature* bugs, plus a third
latent crash found along the way. Diagnosis method: instrumented the forward
pass and measured across-sample variance at every stage. Embeddings `z` were
collapsed — between-class cosine 0.93, within-class 0.97, `z.std ≈ 0.03` — and
the collapse was present **at initialisation**, before any training, which
ruled out the optimiser/loss and pointed at the feature scale.

**45. [CRITICAL, FIXED] Node feature `fanout_ratio` (Block-1 #5) exploded to ~1e6 for zero-out-degree nodes → total embedding collapse → G0 unlearnable**
`compute_node_features_block1` computed `out[i,4] = len(peers[i]) / (out_deg +
1e-6)`. For any node with **out-degree 0** — a pure *destination*, i.e. exactly
a scan/flood *victim* — this divides by `1e-6` and returns ~1e6–3e6 (measured
per-column max 3.0e6, mean 6.6e5, on a 40/class subset). Node features are
otherwise all O(1); edge features are clipped to ±5. That one column dwarfed
everything: after the `W_v` input projection every node state pointed the same
direction (`meandir_dominance = 1.000`), so by the edge readout 95% of every
flow embedding was one shared direction (`dominance = 0.950`) and all classes
were indistinguishable. This directly violates the invariant docs/04 §3.1
states for itself — "every Block-1 feature is degree-capped or log-compressed
so a flooding attacker cannot drive any of them arbitrarily." **Fix
(`graph/node_features.py`):** normalise by *total* degree —
`len(peers) / (out_deg + in_deg + 1e-6)` — which is provably in [0,1] (every
distinct peer comes from an incident edge, so `len(peers) ≤ out_deg + in_deg`),
stays defined for out-degree-0 nodes, and preserves the scanning signal (same
`x/(in+out+eps)` shape as feature #11 `reverse_ratio`). **Verified:**
`node_feat` abs-max across bins dropped from ~3e6 to 30.0; G0 subset train_acc
rose from 0.056 (stuck/collapsed) to 0.98+ and is climbing (see verification
log). This is distinct from fixed-item **#6** above (victim-side accumulation /
`distinct_dst_ports` placeholder): #6 is about *missing* victim signal, #45 is
about a victim node's *existing* feature blowing up the whole vector.

**46. [HIGH, FIXED] RobustScaler "bounded" block escaped the ±5 clip — `DNS_QUERY_ID` etc. reached the model at 65535**
`features/pipeline.py::transform` clips the TE1 and TE2 conditioned blocks to
`±clip_post_transform` (=5.0) but applied the `RobustScaler` "bounded" block
with no clip at all. Five `BOUNDED_NUMERIC` columns are >75% zeros
(`DNS_QUERY_ID`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_TYPE`,
`DNS_TTL_ANSWER`): their IQR is 0, so `RobustScaler` sets `scale_ = 1.0` and
passes the raw value straight through — `DNS_QUERY_ID` then reaches the model
at up to 65535, four orders of magnitude past the ±5 bound every other numeric
block obeys. **Fix:** store `clip_post_transform` on the pipeline and
`np.clip(bounded_arr, -c, +c)` before assembling the block. **Impact:**
requires re-running `03_fit_features` (re-transform; scaler params unchanged)
so the cached parquets no longer contain the 65535 values, then rebuilding
graph caches.

**47. [LOW, FIXED] `norm_node = layernorm | rmsnorm` crashed — `batch` passed to a norm that doesn't take it**
`SRTEGLayer.forward` always called `self.upd_norm(concat, batch)`, but only
`GraphNorm.forward` accepts `batch`; `nn.LayerNorm`/`RMSNorm` take `x` alone —
`TypeError: forward() takes 2 positional arguments but 3 were given`. So the
`norm_node=layernorm`/`rmsnorm` ablations have never been runnable (found while
bisecting #45). **Fix (`models/srteg.py`):** pass `batch` only when
`isinstance(self.upd_norm, GraphNorm)`.

**48. [CRITICAL, FIXED] `SRTEGLayer.forward` looped over nodes in Python — ~8.25 ms/node, ~3 h/epoch, 30 epochs unreachable**
Found while diagnosing "stalls" in the first full Stage-1 run. `forward` ran
`for v in range(N)`, and for each node issued a handful of tiny GPU ops
(`W_Q`/`W_K`, an einsum, softmax, then `trimmed_mean` + `trimmed_std` +
projection). On a T4 that measured **8.25 ms per node**, and it is
latency-bound on kernel launches rather than FLOPs, so a bigger GPU would not
have helped.

CICIDS2018 bins are extremely uneven — median 14 nodes, mean 132, p95 1172,
max 1799 — totalling ~1.22 M node-iterations per epoch. Fitting
`t = 77 ms/bin + 8.25 ms/node` against the Kaggle log projects **~3.0 h/epoch,
~90 h for 30 epochs, ~4 epochs per 12 h session** — i.e. eight sessions for one
protocol-A run, repeated for each of the five protocol-B holdouts. Note this is
*not* what made the first run look erratic to the eye: the diagnosis initially
blamed Kaggle host contention (see docs/KAGGLE_PLAN.md correction), because
progress was being read in bins rather than nodes.

**Fix:** the loop is unnecessary — the layer already materialises dense
`msgs [N, K, d_h]` with a `[N, K]` validity mask, so attention and aggregation
are one batched pass. Added `trimmed_mean_batched`, `trimmed_std_batched`,
`mean_aggregate_batched`, `soft_medoid_batched`, and `forward_batched` on
`MultiAggregator`/`RobustAggregator` (`models/aggregation.py`), and rewrote
`SRTEGLayer.forward` to use them. Semantics preserved exactly, including the
`int(beta * n_v)` truncation (computed in float64 so it matches Python's `int()`
across the rounding boundary), the `n_v - 2k < 1` fallback with its
*unnormalised* weighted sums, per-node `a_scale = log1p(n_v)/log1p(K)`, and
zero output rows for isolated nodes (the loop's `continue` left `out[v]` at its
zero init). Masked-softmax uses a finite `-1e30` sentinel rather than `-inf` so
an all-masked row yields finite weights instead of NaN.

Two incidental findings: the loop computed `attended = W_O(head_msgs)` and
**discarded it** (the trimmed-mean aggregate is used instead, deliberately —
it preserves the breakdown-point argument), so that work is simply not done now;
`self.attn.W_O` is left defined so existing checkpoints stay loadable. And the
`record_attention` XAI side channel keeps its per-node dict, built only when
explicitly enabled, so training pays nothing for it.

**Measured on the T4** (same 400-bin range, loop vs batched, identical data):

| | loop | batched | |
|---|---|---|---|
| per-node cost | 8.26 ms | **0.62 ms** | 13.3× |
| per-bin fixed cost | 77 ms | 95 ms | — |
| projected epoch | 3.01 h | **0.46 h** | 6.6× |
| 30 epochs | ~90 h | **~13.7 h** | |

Light bins (2 nodes) are unchanged at ~0.095 s/bin — they were never node-bound.
Heavy bins (170 nodes) went 74.1 s → 10.0 s for 50 bins (7.4×). Note the fixed
per-bin cost now dominates (**54%** of an epoch): the next optimisation target
is per-bin overhead (cache load + per-bin Python/optimizer work), not per-node
work.

**Memory caveat — batching trades memory for speed.** The dense `[N, K, d_h]`
layout allocates all K=32 neighbour slots while only **5.14 are valid on
average** (an 8.2× waste); the loop only ever touched the valid ones, so it was
memory-frugal *because* it was sparse. Autograd retains those tensors for the
whole BPTT chunk, and `bptt_chunk=8` needed ~21 GB on a chunk of large bins —
OOM on a 14.5 GB T4, twice. Mitigations applied: `trimmed_mean_std_batched`
shares one sort between the mean and the multi-aggregator's spread term (the
two were sorting the same `[N,K,d]` tensor twice), and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `bptt_chunk` is reduced
from the documented T_bptt=8 (2 verified to fit; 4 in use) — **a disclosed
deviation**, shortening the window over which gradient flows through the TE6
recurrence. Truncating K to the per-bin max was measured and rejected (1.1×;
81% of bins have at least one node with all 32 slots full). The clean fix that
would restore T_bptt=8 is to compute on the flat valid-edge list (`[E, d]`,
E ≈ N·5.14) with segment-wise softmax and trimmed mean — 8.2× less memory *and*
less compute than dense. Not yet implemented.

**Verified:** `tests/test_vectorised_layer.py` transcribes the original loop as
a reference and asserts equivalence across {trimmed, mean, soft_medoid} ×
{te4 on, off}, over inputs that include an isolated node, a single-neighbour
node (which triggers the trimming fallback), and a fully-connected node; plus
gradient flow to inputs and `lambda_hat`, and no NaN when every node is
isolated. Parameters are unchanged, so Stage-1 checkpoints remain compatible.

---

## Decisions needed (updated after part 3)

Every item from the original "Decisions needed" list is now RESOLVED
(#39-#44 above). What remains is the disclosed-deviations list at the end of
the part-3 section (LR scheduler, DropEdge, 64-bin batching/AMP, accuracy-
proxy early stopping, `distinct_dst_ports` placeholder) — none block the
Kaggle runs, all should stay disclosed in the paper's implementation notes.

One operational note: Protocol-B open-set numbers now require 5 holdout
training runs (scripts 02→05 with `--holdout-index 0..4`). This is the real
cost of leakage-free C1 claims — docs/KAGGLE_PLAN.md sequences it across
sessions.

---

## Verification log

- Full pytest suite: **97 passed**, unchanged count, throughout every fix in
  this session (re-run after each logical group of changes, not just once
  at the end).
- Independently re-derived `sorted(train_features.parquet["canonical_label"].unique())`
  from the current on-disk data and confirmed it's **identical** to the
  stray `data/artifacts/cicids2018/class_vocab.json` the local cache build
  happened to use — the existing 12,623-bin local cache (train: 9,375
  bins/2.7GB, val: 3,248 bins/580MB) has correct labels and does not need a
  rebuild.
- Independently measured the real per-day distribution of majority classes
  in `data/interim/cicids2018/cleaned.parquet` (pandas groupby on the actual
  parquet file, not a re-statement of the agent's claim) — confirmed bug #8
  exactly: benign present on 2/9 days, other flagged classes present on
  their single TRAP-2-documented day (not a subsampling artifact for those).
- Independently executed `Path(...).parents[N]` for `resolve_class_vocab`'s
  file location to confirm the off-by-one before and after the fix (bug #4).
- Independently executed the exact `01_prepare_data.py` benign-quota-clobber
  scenario (bug #9) with synthetic counts matching the documented
  `--nrows 200000` smoke test, before and after the fix.
- Independently executed `np.trapz`/`np.trapezoid` against the installed
  numpy 2.4.6 to confirm both the crash and the sign inversion (bug #29)
  before writing the fix.
- Independently confirmed `EPCHead.named_parameters()` yields `"tau_hat"`
  as the exact string needed for the exclusion filter (bug #15).
- Simulated the fixed time-stratified subsampler (bug #33) on synthetic
  9-day data with a realistic quota; confirmed all 9 days represented
  (2,200-2,400 rows each) vs. 100% from day 0 under the old logic.
- Simulated a burst-injection attack (bug #36) with real timestamps; the
  attacker's sample share came out at 18.75%, correctly bounded near 1/Q=25%
  and confined to the most-recent stratum.
- Simulated a 50-attacker single-victim flood (bug #37); victim's
  `byte_volume`/`short_scale_burst` went from a hardcoded 0.0 to 0.62/30.0.
- Directly inspected `model.memory.gru.weight_ih.grad` after a training
  step (bug #38) at each of the three attempted fixes — confirmed the first
  two crashed with different, diagnostic errors before landing on the
  version that runs cleanly; confirmed the final version's GRU gradient is
  `None` (documented limitation) while independently confirming
  `te6_enabled=true` vs `false` now produce different loss trajectories on
  identical data/seed (the actual bug being fixed).
- Full 97-test pytest suite re-run after every fix in both architecture-fix
  batches, not just once at the end.

### Part-3 verification log (2026-07-26, continued)

- BPTT restructure (#39): directly confirmed `model.memory.gru.weight_ih`
  changes after one training epoch (max |Δ| ≈ 5.0e-3; previously `grad is
  None`/frozen), across two consecutive epochs plus an eval pass with no
  autograd errors.
- Resume trajectory (#40/#41): two identical uninterrupted runs are
  bit-identical (rules out ambient nondeterminism); weights, optimizer state
  (tensor-by-tensor), and all RNG streams verified bit-identical at the
  epoch-2 boundary between interrupted and uninterrupted runs; remaining
  divergence bisected to `node_feat` of the first resumed bin via forward-
  input hashing, root-caused to `_prev_active_nodes`, and eliminated —
  `tests/test_checkpoint_resume.py` (3 tests) green.
- Gates: end-to-end stage1+stage2 synthetic training run green with all
  gate hooks active; G0 path import-checked and exercised structurally (its
  full data path runs as part of any real 04 invocation).
- F7 (#43): `tests/test_xai_adversarial_linkage.py` — synthetic injection of
  12 flows: mask count survives re-sort exactly (12), marked long-scale
  edges all point at the victim, targets count unchanged post-injection
  (injected flows never become targets), attr_edge sums to 1.0, fraction
  bounded, side-channels reset.
- Holdout path (#44): all seven touched scripts import-check clean; 02
  `--protocol B --holdout-index 0` + 03 `--holdout-index 0` executed against
  the real regenerated cicids2018 data (see session log for output).
- Full pytest suite re-run at the end of the pass (includes the 6 new
  tests).
