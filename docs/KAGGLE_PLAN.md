# Kaggle execution plan — one script per session, nothing lost to the 12h cap

Written 2026-07-26, after the part-3 fixes (docs/BUGS.md #39–#44). Assumes
the workflow you described: run one script per Kaggle session, download the
outputs locally between sessions (or via the Kaggle CLI), and feed them back
as an input dataset to the next session.

## Part-4 update (2026-07-26): data must be rebuilt before uploading

The first real validation kernel's G0 preflight failed (train_acc ≈ 0.06) and
caught two unscaled-feature bugs (docs/BUGS.md #45 node `fanout_ratio` → ~1e6;
#46 bounded block escaped the ±5 clip → `DNS_QUERY_ID` at 65535). Both are
fixed in code. Because the cached parquets and graph caches were built by the
old pipeline, **the previously-uploaded `argus-data` is stale** and must be
regenerated locally before any GPU session:

    python scripts/03_fit_features.py  --dataset cicids2018      # re-transform w/ clip
    python scripts/03_cache_graphs.py  --dataset cicids2018      # rebuild node feats
    # then per holdout i in 0..4 (protocol-B track, can follow later):
    #   03_fit_features / 03_cache_graphs --holdout-index i

Node features are recomputed at cache-build time, so the `fanout_ratio` fix
lands via `03_cache_graphs`; the clip fix lands via `03_fit_features`. G0 now
passes locally (0.06 → 0.993 by epoch ~82) with the hardened preflight
(150 epochs + cosine LR decay, local to the throwaway check). Re-version the
`argus-code` dataset (code changed) and replace `argus-data` before rerunning
the validation kernel.

## The graph cache MUST be uploaded, not rebuilt on Kaggle (learned run 1)

The first full Stage-1 attempt was launched without a cache dataset, letting
epoch 0 build all 9,268 train bins into `/kaggle/working`. Two problems, both
fatal to the multi-session plan:

1. **`/kaggle/working` is ephemeral.** The ~900 MB cache built during a session
   is discarded at session end, so every resumed session — and each of the 5
   protocol-B holdout runs — would rebuild it from scratch.
2. ~~**Kaggle's shared host stalls unpredictably.**~~ **This was wrong** — see
   the correction below. The uneven bin timings were data-dependent, not host
   contention.

Ruled out during the diagnosis (recorded so it isn't re-investigated): the
untimed gzip cache-write is only ~9 min total across 9,268 bins, so it never
explained the stalls. `CachedGraphSource` now reports `write=` separately from
`build=` and `load=` so this is attributable straight from the log next time.

### Correction: the "stalls" were the per-node loop, not the host

The first diagnosis blamed Kaggle host contention because bin timings looked
work-independent (bins 400→450 took 325 s, 450→500 took 8 s). That was an
artifact of only comparing *bin counts*. Across three separate runs the slow
stretch landed in exactly the same place (bins 350→400: 74 s / 78 s / 80 s),
which rules out ambient contention — and the cached data explains it exactly:

| bins | nodes/bin | s/bin |
|---|---|---|
| 50–350 | 2 | 0.094 |
| 350–400 | 170 (max 837) | 1.48 |

85× the nodes, 85× the time. Bins are wildly uneven (median 14 nodes, mean 132,
p95 1172, max 1799), so *bins processed* is a near-useless progress unit; the
real unit is **nodes processed**. Root cause was the `for v in range(N)` loop in
`SRTEGLayer.forward` at ~8.25 ms/node — see docs/BUGS.md #48 for the fix.

**Therefore:** always attach `harshitpachahara/argus-cache` (908 MB, protocol-A
train+val) and pass `cache_input` to `kaggle_bootstrap.bootstrap()`. The
bootstrap symlinks it to `<out_dir>/<dataset>_stage1/cache`, read-only, so every
epoch replays at ~14 ms/bin with zero build cost. Upload each holdout's cache
before starting the protocol-B track.

Note when wiring resume: detect a resumable prior run by globbing for an actual
`stage1_ckpt_last.pt`, not for a `results/runs` path — the cache dataset also
matches the latter and would pass `--resume` with nothing to resume from.

**Kaggle rewrites the cache dataset's layout on mount — do not hardcode paths.**
Two transformations happen, both caught by a wasted run:

1. **The zip-name level is stripped.** `cicids2018_stage1.zip` containing
   `cache/train/...` mounts as `<mount>/cache/train/...`, NOT
   `<mount>/cicids2018_stage1/cache/train/...`.
2. **`.pt.gz` bins are decompressed to `.pt`.** Kaggle unpacks gzip members, so
   the mounted cache holds `bin_000123.pt`. This is harmless —
   `CachedGraphSource.build_bin_batch` already falls back to the uncompressed
   path and it loads *faster* (no decompress) — but any glob written for
   `*.pt.gz` alone will silently find nothing.

Also note this dataset mounts at `/kaggle/input/argus-cache` (top level), while
argus-code/argus-data mount under `/kaggle/input/datasets/<owner>/<slug>`. Mount
layout is not consistent across datasets, so always *discover* the cache by
globbing for `**/cache/train/bin_*` and passing the directory that contains
`cache/` as `cache_input`. Print the resolved root and fail loudly when it is
`None` — a silent `None` means the session rebuilds ~9.3 K bins and throws them
away, which is exactly the failure this section exists to prevent.

## Kaggle environment facts (learned the hard way via the validation kernel)

Three environment gotchas were caught by a cheap validation kernel BEFORE any
long run (see docs/BUGS.md "Kaggle bring-up" section):

1. **Accelerator MUST be forced to T4.** Kaggle's default GPU for this account
   is a Tesla P100 (compute capability sm_60), but the pre-installed
   torch 2.10 dropped sm_60 (supports sm_70+). The P100 loads but every CUDA
   op fails. Fix: request the T4 (sm_75) explicitly —
   `kaggle kernels push --accelerator NvidiaTeslaT4`, and/or
   `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json. Valid values:
   `NvidiaTeslaT4`, `NvidiaTeslaP100`.
2. **Dataset mount path is not `/kaggle/input/<slug>`.** These datasets mount
   at `/kaggle/input/datasets/<owner>/<slug>`. `scripts/kaggle_bootstrap.py`
   `find_input_dataset()` discovers the real location by recursive search;
   always resolve mounts through it rather than hardcoding.
3. **Deps:** torch/pandas/sklearn/pyarrow/scipy/joblib are pre-installed;
   `omegaconf` is not — `pip install -q omegaconf>=2.3` at kernel start
   (internet enabled).

Kaggle datasets are created with `--dir-mode zip` (the CLI 2.2.3 on Windows
mangles temp paths for individual top-level files; zip-per-subdir sidesteps
it — and dir-mode-zip datasets DO mount extracted, verified). Keep every
top-level entry a directory; do not ship top-level loose files.

## Why this is now safe

- **Per-epoch checkpoint/resume**: 04/05 write `stage{1,2}_ckpt_last.pt`
  every epoch (atomic writes, real optimizer + RNG state). If a session dies
  at epoch 20/30, the next session runs the same command plus `--resume` and
  continues from epoch 21 with the trajectory reproduced (tested to ≤1e-3).
- **Run registry**: completed runs are recorded in
  `results/runs/registry.jsonl`; re-running a finished script skips instead
  of retraining (override with `--force`). Safe to re-submit a session
  blindly.
- **G0 preflight**: 04 refuses to start a full run if the model can't
  memorise 1,000 flows (minutes, on-GPU). A broken config costs minutes, not
  a session.
- **G6 hard-stop**: a NaN/Inf loss kills the run immediately instead of
  burning the remaining hours.
- **Cache fingerprint**: a config drift between cache-build and training
  raises immediately instead of silently training on wrong windows.

## What to upload once (input dataset, built locally — already done)

    data/processed/cicids2018/*.parquet          (protocol-A splits + features)
    data/artifacts/cicids2018/                    (pipeline, manifest)
    results/runs/cicids2018_stage1/cache/         (train+val graph cache, ~0.9 GB)

Between sessions, download the session's `results/runs/**` output and add it
to the next session's inputs (or use `kaggle datasets version` to update a
"argus-runs" dataset from the CLI).

## Main (protocol-A) sequence

| # | Session command | Hardware | Est. time | Download afterwards |
|---|---|---|---|---|
| 1 | `python scripts/04_train_encoder.py --dataset cicids2018 --set run.device=cuda` | GPU | hours; resumable | `cicids2018_stage1/` (ckpts, history, gates_report, class_vocab) |
| 1b | (only if session 1 died) same command + `--resume` | GPU | remaining epochs | same |
| 2 | `python scripts/05_train_head.py --dataset cicids2018 --set run.device=cuda` (+`--resume` if died) | GPU | shorter than stage 1 | `cicids2018_stage2/` (ckpt, thresholds, gates G3/G4) |
| 3 | `python scripts/06_eval_closed_set.py --dataset cicids2018` | GPU | ~1h | `cicids2018_closed_set/` |
| 4 | `python scripts/11_run_adversarial.py --dataset cicids2018` | GPU | ~1-2h | `cicids2018_adversarial/` |
| 5 | `python scripts/12_run_xai.py --dataset cicids2018` (now includes the F7 sweep) | GPU | ~1h | `cicids2018_xai/` |
| 6 | `python scripts/13_measure_deployment.py --dataset cicids2018` | GPU | <1h | deployment report |

Notes:
- Stage-1 epoch count vs the 12h cap: with the cached graphs an epoch is
  I/O + GPU only. If the first session shows >30 min/epoch, either lower
  `train.stage1_epochs` via `--set` or just let it die and `--resume` — both
  are now safe.
- The G0 preflight runs at the start of session 1 (minutes). If it fails,
  STOP and debug locally — that is the gate doing its job.

## Protocol-B (open-set, T3/T4) sequence — 5 holdout runs

Local, once (CPU, ~15 min total): build all five split/feature sets

    for i in 0 1 2 3 4:
        python scripts/02_build_splits.py  --dataset cicids2018 --protocol B --holdout-index $i
        python scripts/03_fit_features.py  --dataset cicids2018 --holdout-index $i

Local or Kaggle-CPU (no GPU quota), once per i (~12 min each):

    python scripts/03_cache_graphs.py --dataset cicids2018 --holdout-index $i

Upload `data/processed/cicids2018/holdout_b*/`, `data/artifacts/cicids2018/holdout_b*/`,
`data/processed/cicids2018/holdout_sets.json`, and the per-holdout caches.

Then one GPU session per holdout (each independently resumable/skippable):

    python scripts/04_train_encoder.py --dataset cicids2018 --holdout-index $i --set run.device=cuda
    python scripts/05_train_head.py    --dataset cicids2018 --holdout-index $i --set run.device=cuda

(04+05 for one holdout fit comfortably in one session — the reduced train
split is smaller than protocol-A's. If pressed for quota, run holdouts at
reduced `--set train.stage1_epochs=...` and disclose.)

Finally (GPU, one session):

    python scripts/07_eval_open_set.py --dataset cicids2018     # T3, needs all/most holdout ckpts
    python scripts/08_eval_few_shot.py --dataset cicids2018     # T4, uses holdouts 0-1

07/08 skip missing holdouts loudly and aggregate over what exists — partial
results are fine mid-way through the 5 runs.

## Ordering across the two tracks

1. Sessions 1–2 (protocol-A stage 1+2) first — they validate the whole
   training path and produce the checkpoint every other script needs.
2. Then interleave: protocol-B holdout sessions are independent of the
   protocol-A eval sessions (3–6), so use whichever quota is available.
3. 07/08 last among the B-track; 15_make_tables.py at the very end, locally.

## If a session dies

1. Re-attach the previous session's output as input (or re-version the runs
   dataset via CLI).
2. Re-run the exact same command with `--resume` added (04/05 only; eval
   scripts are cheap enough to just re-run).
3. Never delete `stage*_ckpt_last.pt` / `registry.jsonl` from the runs
   dataset — they are the resume state.
