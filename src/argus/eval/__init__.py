"""Evaluation: metrics, calibration, open-set, selective prediction, continual
learning, bootstrap CIs, deployment measurement, and run report assembly.

See docs/08_EVALUATION.md.
"""

from argus.eval.bootstrap import bootstrap_ci, paired_bootstrap_ci
from argus.eval.calibration import (
    brier_score,
    calibration_report,
    ece,
    mce,
    nll,
    reliability_diagram_data,
)
from argus.eval.continual import backward_transfer, few_shot_report, forgetting
from argus.eval.deployment import (
    DeploymentReport,
    measure_streaming_throughput,
    model_size,
)
from argus.eval.metrics import closed_set_report, per_tier_macro_f1
from argus.eval.openset import open_auc, open_set_report, openness, unknown_tpr_fpr
from argus.eval.report import (
    aggregate_over_seeds,
    assemble_run_report,
    save_run_report,
)
from argus.eval.selective import (
    aurc,
    deferral_precision,
    e_aurc,
    risk_at_coverage,
    risk_coverage_curve,
    selective_report,
)

__all__ = [
    # closed-set
    "closed_set_report",
    "per_tier_macro_f1",
    # calibration
    "ece",
    "mce",
    "brier_score",
    "nll",
    "reliability_diagram_data",
    "calibration_report",
    # open-set
    "unknown_tpr_fpr",
    "open_auc",
    "openness",
    "open_set_report",
    # selective
    "risk_coverage_curve",
    "aurc",
    "e_aurc",
    "risk_at_coverage",
    "deferral_precision",
    "selective_report",
    # continual
    "forgetting",
    "backward_transfer",
    "few_shot_report",
    # deployment
    "DeploymentReport",
    "measure_streaming_throughput",
    "model_size",
    # bootstrap
    "paired_bootstrap_ci",
    "bootstrap_ci",
    # report
    "assemble_run_report",
    "save_run_report",
    "aggregate_over_seeds",
]
