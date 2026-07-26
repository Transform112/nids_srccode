"""Kaggle session bootstrap: map read-only input datasets into the paths the
ARGUS scripts expect, and hand back the config overrides to use.

Kaggle mounts input datasets read-only at /kaggle/input/<slug>/ and gives one
writable dir, /kaggle/working. Rather than copy the code or data around, we
exploit pathlib's "join-with-absolute discards the left side" rule: passing
absolute `paths.*` / `run.out_dir` overrides makes `resolved_path` (which
prepends the repo root) return the absolute path verbatim. So the only real
filesystem work here is (a) making a previously-uploaded graph cache visible
at the writable run dir via symlink, and (b) copying a prior session's run
dir into the writable tree so `--resume` can overwrite checkpoints in place.

Typical use inside a kernel script:

    from scripts.kaggle_bootstrap import bootstrap
    ov = bootstrap(
        dataset="cicids2018",
        data_input="/kaggle/input/argus-data",
        cache_input="/kaggle/input/argus-cache",     # optional
        prior_runs_input="/kaggle/input/argus-runs",  # optional, for --resume
        working="/kaggle/working",
    )
    # ov is a list like ["paths.processed_dir=...", "run.out_dir=...", ...]
    # pass it to the script: python scripts/04_train_encoder.py --dataset ... \
    #     $(printf ' --set %s' "${ov[@]}")
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path


def find_input_dataset(slug: str, marker: str | None = None) -> str | None:
    """Locate a mounted Kaggle input dataset by slug, tolerating mount-layout
    differences (/kaggle/input/<slug> vs /kaggle/input/datasets/<owner>/<slug>).

    If `marker` is given (a relative path that must exist inside the dataset,
    e.g. "processed"), only a candidate containing it is accepted — this
    disambiguates when several mounts share a basename.
    """
    candidates = [f"/kaggle/input/{slug}"]
    candidates += sorted(glob.glob(f"/kaggle/input/**/{slug}", recursive=True))
    for c in candidates:
        if not os.path.isdir(c):
            continue
        if marker is None or os.path.exists(os.path.join(c, marker)):
            return c
    return None


def bootstrap(
    dataset: str,
    data_input: str,
    working: str = "/kaggle/working",
    cache_input: str | None = None,
    prior_runs_input: str | None = None,
    holdout_index: int | None = None,
) -> list[str]:
    """Prepare the working tree and return config-override strings.

    Args:
        dataset: dataset name (e.g. "cicids2018").
        data_input: mount of the data dataset; must contain `processed/` and
            `artifacts/` subdirs (as staged by the local uploader).
        working: writable dir (Kaggle: /kaggle/working).
        cache_input: mount of the graph-cache dataset, if training from a
            prebuilt cache. Must contain `<dataset>_stage1[_b<i>]/cache/`.
        prior_runs_input: mount of a previous session's run outputs, copied
            into the writable results tree so `--resume` can continue.
        holdout_index: Protocol-B holdout index, if any (selects the
            `_b<i>` run-dir suffix and the cache subdir).

    Returns:
        List of `key=value` strings to pass as repeated `--set` overrides.
    """
    work = Path(working)
    results = work / "results" / "runs"
    results.mkdir(parents=True, exist_ok=True)

    suffix = "" if holdout_index is None else f"_b{holdout_index}"

    # Tolerate the /kaggle/input/datasets/<owner>/<slug> mount layout: if the
    # given path has no processed/, rediscover it by basename.
    if not (Path(data_input) / "processed").is_dir():
        found = find_input_dataset(Path(data_input).name, marker="processed")
        if found is not None:
            print(f"[bootstrap] data_input {data_input} -> discovered {found}")
            data_input = found

    processed = Path(data_input) / "processed"
    artifacts = Path(data_input) / "artifacts"
    if not processed.is_dir() or not artifacts.is_dir():
        listing = [p.name for p in Path(data_input).iterdir()] if Path(data_input).is_dir() else "(missing)"
        raise FileNotFoundError(
            f"data_input {data_input} must contain processed/ and artifacts/ (found: {listing})"
        )

    if cache_input is not None and not Path(cache_input).is_dir():
        cache_input = find_input_dataset(Path(cache_input).name)
    if prior_runs_input is not None and not Path(prior_runs_input).is_dir():
        prior_runs_input = find_input_dataset(Path(prior_runs_input).name)

    # Restore a prior session's run dir into the writable tree for --resume.
    if prior_runs_input is not None and Path(prior_runs_input).is_dir():
        for run_dir in Path(prior_runs_input).iterdir():
            if not run_dir.is_dir():
                if run_dir.name == "registry.jsonl":
                    shutil.copy2(run_dir, results / "registry.jsonl")
                continue
            dst = results / run_dir.name
            if dst.exists() or dst.is_symlink():
                continue
            shutil.copytree(run_dir, dst)
            print(f"[bootstrap] restored prior run dir {run_dir.name} for resume")

    # Symlink a prebuilt cache into the run dir the scripts derive
    # (out_dir/<dataset>_stage1[_b<i>]/cache). Read-only target is fine — a
    # complete cache has no misses and only reads.
    if cache_input is not None and Path(cache_input).is_dir():
        stage1_run = results / f"{dataset}_stage1{suffix}"
        stage1_run.mkdir(parents=True, exist_ok=True)
        link = stage1_run / "cache"
        cache_src = _find_cache_dir(Path(cache_input), dataset, suffix)
        if cache_src is not None and not link.exists() and not link.is_symlink():
            os.symlink(cache_src, link)
            print(f"[bootstrap] linked cache {cache_src} -> {link}")

    overrides = [
        f"paths.processed_dir={processed}",
        f"paths.artifact_dir={artifacts}",
        f"run.out_dir={results}",
    ]
    print("[bootstrap] overrides:")
    for o in overrides:
        print(f"[bootstrap]   {o}")
    return overrides


def _find_cache_dir(cache_input: Path, dataset: str, suffix: str) -> Path | None:
    """Locate the cache dir inside the mounted cache dataset, tolerating a few
    plausible layouts (…/<dataset>_stage1[_b<i>]/cache, …/cache, or the mount
    root already being the cache)."""
    candidates = [
        cache_input / f"{dataset}_stage1{suffix}" / "cache",
        cache_input / "cache",
        cache_input,
    ]
    for c in candidates:
        if (c / "train").is_dir() or any(c.glob("*/bin_*.pt.gz")) or any(c.glob("bin_*.pt.gz")):
            return c
    return None
