"""13 — Measure deployment throughput/latency/memory for ARGUS and E-GraphSAGE
on the same host, via the real `StreamingDetector` push() contract.

Requires a trained checkpoint (scripts/04 + 05). Usage:
    python scripts/13_measure_deployment.py --dataset cicids2018 --n-flows 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from argus.config import load_config, resolved_path  # noqa: E402
from argus.eval.deployment import measure_streaming_throughput, model_size  # noqa: E402
from argus.graph.builder import assign_node_ids  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.models.baselines.egraphsage import EGraphSAGE  # noqa: E402
from argus.streaming.detector import StreamingDetector  # noqa: E402
from argus.train.checkpoint import load_checkpoint  # noqa: E402


def run(dataset: str, n_flows: int = 2000, overrides: list[str] | None = None) -> dict:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]
    with open(artifact_dir / "class_vocab.json") as f:
        class_names = json.load(f)

    feat_test = pd.read_parquet(processed_dir / "test_features.parquet")
    feat_test = feat_test.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    feat_test = feat_test.iloc[:n_flows]

    src_ids, dst_ids, _ = assign_node_ids(
        feat_test, node_granularity=cfg.graph.node_granularity,
        src_ip_col="IPV4_SRC_ADDR", dst_ip_col="IPV4_DST_ADDR",
        src_port_col="L4_SRC_PORT", dst_port_col="L4_DST_PORT",
    )
    times_ms = feat_test["FLOW_START_MILLISECONDS"].to_numpy()
    feature_rows = feat_test[feature_names].to_numpy(dtype=np.float32)

    device = torch.device("cpu")  # deployment measurement is a single-host CPU/GPU comparison; CPU is the common baseline
    results: dict[str, dict] = {}

    argus_model = ArgusModel(cfg, f_e=f_e, f_v=18, class_names=class_names).to(device)
    stage2_ckpt = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage2" / "stage2_final.pt"
    if stage2_ckpt.exists():
        load_checkpoint(stage2_ckpt, argus_model)
    else:
        print(f"[13] Warning: no checkpoint at {stage2_ckpt}; measuring an untrained ARGUS model's cost profile.")
    argus_detector = StreamingDetector(
        argus_model, class_names, node_granularity=cfg.graph.node_granularity,
        anchor_bin_seconds=cfg.graph.anchor_bin_seconds,
        window_short_seconds=cfg.graph.window_short_seconds,
        window_mid_seconds=cfg.graph.window_mid_seconds,
        window_long_seconds=cfg.graph.window_long_seconds,
        neighbour_cap=cfg.graph.neighbour_cap, device=device,
    )
    print(f"[13] Measuring ARGUS on {n_flows} flows ...")
    results["argus"] = asdict(
        measure_streaming_throughput(argus_detector, feature_rows, times_ms, src_ids, dst_ids)
    )

    egraphsage_model = EGraphSAGE(f_e=f_e, d_h=64, num_classes=len(class_names), layers=2).to(device)
    egraphsage_detector = StreamingDetector(
        egraphsage_model, class_names, node_granularity=cfg.graph.node_granularity,
        anchor_bin_seconds=cfg.graph.anchor_bin_seconds,
        window_short_seconds=cfg.graph.window_mid_seconds,  # E-GraphSAGE: single (mid) scale
        window_mid_seconds=cfg.graph.window_mid_seconds,
        window_long_seconds=cfg.graph.window_mid_seconds,
        neighbour_cap=10**6, device=device,
    )
    # EGraphSAGE.forward doesn't implement .decide(); StreamingDetector falls
    # back to argmax, which is fine for a throughput/latency comparison.
    print(f"[13] Measuring E-GraphSAGE on {n_flows} flows ...")
    results["egraphsage"] = asdict(
        measure_streaming_throughput(egraphsage_detector, feature_rows, times_ms, src_ids, dst_ids)
    )

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "deployment_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for name, r in results.items():
        print(f"[13] {name}: {r['throughput_flows_per_sec']:.1f} flows/s, "
              f"p50={r['latency_p50_ms']:.2f}ms, p99={r['latency_p99_ms']:.2f}ms, "
              f"{r['model_size_params']:,} params ({r['model_size_mb']:.2f} MB)")
    print(f"[13] Wrote {out_dir / 'deployment_report.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-flows", type=int, default=2000)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    run(args.dataset, n_flows=args.n_flows, overrides=args.overrides)
