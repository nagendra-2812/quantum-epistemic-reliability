from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

SEED = 2026
EPS = 1e-8

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

RUN34B = (
    ROOT
    / "experiments"
    / "STEP34B_FINAL_MATCHED_MLP_VQC"
)

RUN35A = (
    ROOT
    / "experiments"
    / "STEP35A_UNCERTAINTY"
)

RUN35B = (
    ROOT
    / "experiments"
    / "STEP35B_PARAMETER_PERTURBATION"
)

RUN37 = (
    ROOT
    / "experiments"
    / "STEP37_CONFORMAL"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP38_SELECTIVE_PREDICTION"
)

SOURCE_DIR = OUT / "source_data"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
METRIC_DIR = OUT / "metrics"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    SOURCE_DIR,
    TABLE_DIR,
    FIG_DIR,
    METRIC_DIR,
    CONFIG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def entropy_from_probability(p):

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return (
        -p * np.log(p)
        -
        (
            1.0 - p
        )
        *
        np.log(
            1.0 - p
        )
    )


def error_detection_auroc(
    y,
    p,
    uncertainty,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    error = (
        pred != y
    ).astype(int)

    if len(
        np.unique(
            error
        )
    ) < 2:

        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            error,
            uncertainty,
        )
    )


def risk_curve(
    y,
    p,
    uncertainty,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    errors = (
        pred != y
    ).astype(float)

    # lowest uncertainty = most trusted
    order = np.argsort(
        uncertainty,
        kind="mergesort",
    )

    ordered_errors = (
        errors[
            order
        ]
    )

    ordered_indices = (
        order
    )

    n = len(
        ordered_errors
    )

    coverage = np.arange(
        1,
        n + 1,
        dtype=float,
    ) / n

    cumulative_error = np.cumsum(
        ordered_errors
    )

    risk = (
        cumulative_error
        /
        np.arange(
            1,
            n + 1,
            dtype=float,
        )
    )

    aurc = float(
        np.trapezoid(
            risk,
            coverage,
        )
    )

    return (
        coverage,
        risk,
        aurc,
        ordered_indices,
    )


def risk_at_coverage(
    y,
    p,
    uncertainty,
    target,
):

    coverage, risk, aurc, order = (
        risk_curve(
            y,
            p,
            uncertainty,
        )
    )

    idx = int(
        np.argmin(
            np.abs(
                coverage
                -
                target
            )
        )
    )

    return {

        "requested_coverage":
            float(target),

        "actual_coverage":
            float(
                coverage[idx]
            ),

        "risk":
            float(
                risk[idx]
            ),

        "accuracy":
            float(
                1.0
                -
                risk[idx]
            ),
    }


def build_patient_level(
    frame,
    probability_column,
):

    rows = []

    for patient, group in frame.groupby(
        "patient_id"
    ):

        p = float(
            group[
                probability_column
            ].mean()
        )

        label = int(
            group[
                "label"
            ].iloc[0]
        )

        rows.append({

            "patient_id":
                str(patient),

            "label":
                label,

            "probability":
                p,

        })

    result = pd.DataFrame(
        rows
    )

    return result


def print_metric_block(
    name,
    y,
    p,
    uncertainty,
):

    auc = error_detection_auroc(
        y,
        p,
        uncertainty,
    )

    coverage, risk, aurc, order = (
        risk_curve(
            y,
            p,
            uncertainty,
        )
    )

    print()
    print(
        name
    )

    print(
        "  error-detection AUROC:",
        auc,
    )

    print(
        "  AURC:",
        aurc,
    )

    for target in [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]:

        item = risk_at_coverage(
            y,
            p,
            uncertainty,
            target,
        )

        print(
            f"  coverage {target:.0%}: "
            f"actual={item['actual_coverage']:.4f}, "
            f"risk={item['risk']:.4f}, "
            f"accuracy={item['accuracy']:.4f}"
        )

    return (
        auc,
        coverage,
        risk,
        aurc,
    )


# ============================================================
# INPUT FILES
# ============================================================

SHOT_SOURCE = (
    RUN35A
    / "source_data"
    / "VQC_SHOT_LEVEL_UNCERTAINTY_SOURCE.csv"
)

PARAM_SOURCE = (
    RUN35B
    / "source_data"
    / "PARAMETER_PERTURBATION_SOURCE_DATA.csv"
)

MLP_PRED = (
    RUN34B
    / "predictions"
    / "MLP_RECORD_LEVEL_PREDICTIONS.csv"
)

VQC_PRED = (
    RUN34B
    / "predictions"
    / "VQC_RECORD_LEVEL_PREDICTIONS.csv"
)

for path in [
    SHOT_SOURCE,
    PARAM_SOURCE,
    MLP_PRED,
    VQC_PRED,
]:

    if not path.is_file():

        raise RuntimeError(
            f"Required input not found: {path}"
        )


print()
print("=" * 100)
print(
    "STEP 38 - SELECTIVE PREDICTION"
)
print("=" * 100)


# ============================================================
# LOAD 34B PREDICTIONS
# ============================================================

mlp = pd.read_csv(
    MLP_PRED
)

vqc = pd.read_csv(
    VQC_PRED
)

required_prediction_columns = {
    "patient_id",
    "label",
    "probability",
}

for name, frame in [
    (
        "MLP",
        mlp,
    ),
    (
        "VQC",
        vqc,
    ),
]:

    missing = (
        required_prediction_columns
        -
        set(
            frame.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{name} prediction file missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# LOAD 35A SHOT DATA
# ============================================================

shot = pd.read_csv(
    SHOT_SOURCE
)

required_shot_columns = {
    "shot_budget",
    "record_index",
    "patient_id",
    "label",
    "mean_probability",
    "shot_variance",
    "predictive_entropy",
}

missing = (
    required_shot_columns
    -
    set(
        shot.columns
    )
)

if missing:

    raise RuntimeError(
        "35A source missing columns: "
        + str(
            sorted(
                missing
            )
        )
    )


shot4096 = shot[
    shot[
        "shot_budget"
    ]
    ==
    4096
].copy()

if len(
    shot4096
) == 0:

    raise RuntimeError(
        "No 4096-shot records found in 35A source."
    )


# ============================================================
# LOAD 35B PARAMETER VARIABILITY
# ============================================================

param = pd.read_csv(
    PARAM_SOURCE
)

required_param_columns = {
    "model",
    "perturbation_sigma",
    "record_index",
    "patient_id",
    "label",
    "perturbed_mean_probability",
    "parameter_variance",
    "predictive_entropy",
}

missing = (
    required_param_columns
    -
    set(
        param.columns
    )
)

if missing:

    raise RuntimeError(
        "35B source missing columns: "
        + str(
            sorted(
                missing
            )
        )
    )


# Use sigma = 0.10 as the primary sensitivity level.
param010 = param[
    np.isclose(
        param[
            "perturbation_sigma"
        ],
        0.10,
    )
].copy()

if len(
    param010
) == 0:

    raise RuntimeError(
        "35B sigma=0.10 records not found."
    )


# ============================================================
# BUILD COMMON RECORD-LEVEL TABLE
# ============================================================

# 34B prediction files preserve the internal-test row order.
# Add an explicit record_index so repeated patients are matched
# record-to-record rather than patient-to-many-record.
if len(mlp) != len(shot4096):
    raise RuntimeError(
        "MLP prediction count does not match 4096-shot record count."
    )

mlp_common = mlp[
    [
        "patient_id",
        "label",
        "probability",
    ]
].copy()

mlp_common.insert(
    0,
    "record_index",
    np.arange(
        len(mlp_common)
    ),
)

mlp_common = (
    mlp_common
    .rename(
        columns={
            "probability":
                "mlp_probability",
        }
    )
)

if len(vqc) != len(shot4096):
    raise RuntimeError(
        "VQC prediction count does not match 4096-shot record count."
    )

vqc_common = vqc[
    [
        "patient_id",
        "label",
        "probability",
    ]
].copy()

vqc_common.insert(
    0,
    "record_index",
    np.arange(
        len(vqc_common)
    ),
)

vqc_common = (
    vqc_common
    .rename(
        columns={
            "probability":
                "vqc_probability",
        }
    )
)

shot_common = shot4096[
    [
        "record_index",
        "patient_id",
        "label",
        "mean_probability",
        "shot_variance",
        "predictive_entropy",
    ]
].copy()

shot_common = (
    shot_common
    .rename(
        columns={
            "mean_probability":
                "vqc_shot_probability",

            "shot_variance":
                "vqc_shot_variance",

            "predictive_entropy":
                "vqc_shot_entropy",
        }
    )
)

param_vqc = param010[
    param010[
        "model"
    ]
    ==
    "VQC"
][
    [
        "record_index",
        "patient_id",
        "label",
        "parameter_variance",
        "parameter_std",
        "predictive_entropy",
    ]
].copy()

param_vqc = (
    param_vqc
    .rename(
        columns={
            "parameter_variance":
                "vqc_parameter_variance",

            "parameter_std":
                "vqc_parameter_std",

            "predictive_entropy":
                "vqc_parameter_entropy",
        }
    )
)


# MLP entropy from the frozen prediction.
mlp_common[
    "mlp_entropy"
] = entropy_from_probability(
    mlp_common[
        "mlp_probability"
    ].to_numpy()
)


# ============================================================
# MERGE
# ============================================================

records = (
    shot_common
    .merge(
        param_vqc,
        on=[
            "record_index",
            "patient_id",
            "label",
        ],
        how="inner",
        validate="one_to_one",
    )
)

records = (
    records
    .merge(
        mlp_common,
        on=[
            "record_index",
            "patient_id",
            "label",
        ],
        how="inner",
        validate="one_to_one",
    )
)

if len(
    records
) != len(
    shot_common
):

    raise RuntimeError(
        "Record-level merge changed the expected test cohort."
    )


# Verify VQC 34B predictions against the same record ordering.
records = (
    records
    .merge(
        vqc_common[
            [
                "record_index",
                "patient_id",
                "label",
                "vqc_probability",
            ]
        ],
        on=[
            "record_index",
            "patient_id",
            "label",
        ],
        how="inner",
        validate="one_to_one",
    )
)

if len(records) != len(shot_common):
    raise RuntimeError(
        "Record-level VQC merge changed the expected cohort."
    )

records[
    "mlp_entropy"
] = entropy_from_probability(
    records[
        "mlp_probability"
    ].to_numpy()
)


# ============================================================
# VALIDATION
# ============================================================

print()
print(
    "Common record count:",
    len(records),
)

print(
    "Unique patients:",
    records[
        "patient_id"
    ].nunique(),
)

print(
    "Labels:",
    records[
        "label"
    ].value_counts()
    .to_dict(),
)

if records[
    "patient_id"
].nunique() != 227:

    print(
        "WARNING: expected 227 internal-test patients; "
        "actual:",
        records[
            "patient_id"
        ].nunique(),
    )


# ============================================================
# RECORD-LEVEL SELECTIVE ANALYSIS
# ============================================================

record_specs = [

    (
        "MLP_entropy",
        "MLP entropy",
        "mlp_probability",
        "mlp_entropy",
    ),

    (
        "VQC_entropy",
        "VQC predictive entropy",
        "vqc_shot_probability",
        "vqc_shot_entropy",
    ),

    (
        "VQC_shot_variance",
        "VQC shot variance",
        "vqc_shot_probability",
        "vqc_shot_variance",
    ),

    (
        "VQC_parameter_variance",
        "VQC parameter variance",
        "vqc_shot_probability",
        "vqc_parameter_variance",
    ),

]


record_summary_rows = []
record_curve_rows = []


for (
    name,
    label,
    probability_column,
    uncertainty_column,
) in record_specs:

    y = records[
        "label"
    ].to_numpy()

    p = records[
        probability_column
    ].to_numpy()

    uncertainty = records[
        uncertainty_column
    ].to_numpy()

    auc, coverage, risk, aurc = (
        print_metric_block(
            label,
            y,
            p,
            uncertainty,
        )
    )

    record_summary_rows.append({

        "analysis":
            name,

        "level":
            "record",

        "error_detection_auroc":
            float(auc),

        "aurc":
            float(aurc),

        "mean_uncertainty":
            float(
                np.mean(
                    uncertainty
                )
            ),

        "median_uncertainty":
            float(
                np.median(
                    uncertainty
                )
            ),

    })


    for c, r in zip(
        coverage,
        risk,
    ):

        record_curve_rows.append({

            "analysis":
                name,

            "level":
                "record",

            "coverage":
                float(c),

            "risk":
                float(r),

        })


# ============================================================
# PATIENT-LEVEL TABLE
# ============================================================

patient = (
    records
    .groupby(
        "patient_id"
    )
    .agg(

        label=(
            "label",
            "first",
        ),

        mlp_probability=(
            "mlp_probability",
            "mean",
        ),

        vqc_probability=(
            "vqc_shot_probability",
            "mean",
        ),

        mlp_entropy=(
            "mlp_entropy",
            "mean",
        ),

        vqc_entropy=(
            "vqc_shot_entropy",
            "mean",
        ),

        vqc_shot_variance=(
            "vqc_shot_variance",
            "mean",
        ),

        vqc_parameter_variance=(
            "vqc_parameter_variance",
            "mean",
        ),

        vqc_parameter_std=(
            "vqc_parameter_std",
            "mean",
        ),

    )
    .reset_index()
)


patient_specs = [

    (
        "MLP_entropy",
        "MLP entropy",
        "mlp_probability",
        "mlp_entropy",
    ),

    (
        "VQC_entropy",
        "VQC predictive entropy",
        "vqc_probability",
        "vqc_entropy",
    ),

    (
        "VQC_shot_variance",
        "VQC shot variance",
        "vqc_probability",
        "vqc_shot_variance",
    ),

    (
        "VQC_parameter_variance",
        "VQC parameter variance",
        "vqc_probability",
        "vqc_parameter_variance",
    ),

]


patient_summary_rows = []
patient_curve_rows = []


for (
    name,
    label,
    probability_column,
    uncertainty_column,
) in patient_specs:

    y = patient[
        "label"
    ].to_numpy()

    p = patient[
        probability_column
    ].to_numpy()

    uncertainty = patient[
        uncertainty_column
    ].to_numpy()

    auc, coverage, risk, aurc = (
        print_metric_block(
            "PATIENT - " + label,
            y,
            p,
            uncertainty,
        )
    )

    patient_summary_rows.append({

        "analysis":
            name,

        "level":
            "patient",

        "error_detection_auroc":
            float(auc),

        "aurc":
            float(aurc),

        "mean_uncertainty":
            float(
                np.mean(
                    uncertainty
                )
            ),

        "median_uncertainty":
            float(
                np.median(
                    uncertainty
                )
            ),

    })

    for c, r in zip(
        coverage,
        risk,
    ):

        patient_curve_rows.append({

            "analysis":
                name,

            "level":
                "patient",

            "coverage":
                float(c),

            "risk":
                float(r),

        })


# ============================================================
# COMBINED RESULTS
# ============================================================

summary_df = pd.DataFrame(
    record_summary_rows
    +
    patient_summary_rows
)

curve_df = pd.DataFrame(
    record_curve_rows
    +
    patient_curve_rows
)


# ============================================================
# COVERAGE TABLE
# ============================================================

coverage_rows = []

for level, frame in [
    (
        "record",
        records,
    ),
    (
        "patient",
        patient,
    ),
]:

    specs = (
        record_specs
        if level == "record"
        else patient_specs
    )

    for (
        name,
        label,
        probability_column,
        uncertainty_column,
    ) in specs:

        y = frame[
            "label"
        ].to_numpy()

        p = frame[
            probability_column
        ].to_numpy()

        u = frame[
            uncertainty_column
        ].to_numpy()

        for target in [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
        ]:

            item = risk_at_coverage(
                y,
                p,
                u,
                target,
            )

            coverage_rows.append({

                "analysis":
                    name,

                "level":
                    level,

                **item,

            })


coverage_df = pd.DataFrame(
    coverage_rows
)


# ============================================================
# SAVE SOURCE/TABLES
# ============================================================

records.to_csv(
    SOURCE_DIR
    / "SELECTIVE_RECORD_LEVEL_SOURCE_DATA.csv",
    index=False,
)

patient.to_csv(
    SOURCE_DIR
    / "SELECTIVE_PATIENT_LEVEL_SOURCE_DATA.csv",
    index=False,
)

summary_df.to_csv(
    TABLE_DIR
    / "TABLE_38_SELECTIVE_SUMMARY.csv",
    index=False,
)

coverage_df.to_csv(
    TABLE_DIR
    / "TABLE_38_RISK_AT_COVERAGE.csv",
    index=False,
)

curve_df.to_csv(
    SOURCE_DIR
    / "RISK_COVERAGE_CURVE_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# FIGURE 1 - RECORD RISK COVERAGE
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for analysis in [
    "MLP_entropy",
    "VQC_entropy",
    "VQC_shot_variance",
    "VQC_parameter_variance",
]:

    subset = curve_df[
        (
            curve_df[
                "level"
            ]
            ==
            "record"
        )
        &
        (
            curve_df[
                "analysis"
            ]
            ==
            analysis
        )
    ]

    if len(subset) == 0:
        continue

    plt.plot(
        subset[
            "coverage"
        ],
        subset[
            "risk"
        ],
        linewidth=1.8,
        label=analysis,
    )

plt.xlabel(
    "Coverage"
)

plt.ylabel(
    "Risk"
)

plt.title(
    "Record-level selective risk-coverage"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_38_RECORD_RISK_COVERAGE.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_RECORD_RISK_COVERAGE.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_RECORD_RISK_COVERAGE_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 - PATIENT RISK COVERAGE
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for analysis in [
    "MLP_entropy",
    "VQC_entropy",
    "VQC_shot_variance",
    "VQC_parameter_variance",
]:

    subset = curve_df[
        (
            curve_df[
                "level"
            ]
            ==
            "patient"
        )
        &
        (
            curve_df[
                "analysis"
            ]
            ==
            analysis
        )
    ]

    if len(subset) == 0:
        continue

    plt.plot(
        subset[
            "coverage"
        ],
        subset[
            "risk"
        ],
        linewidth=1.8,
        label=analysis,
    )

plt.xlabel(
    "Coverage"
)

plt.ylabel(
    "Risk"
)

plt.title(
    "Patient-level selective risk-coverage"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_38_PATIENT_RISK_COVERAGE.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_PATIENT_RISK_COVERAGE.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_PATIENT_RISK_COVERAGE_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3 - ERROR DETECTION
# ============================================================

plot_df = summary_df[
    summary_df[
        "level"
    ]
    ==
    "patient"
].copy()

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

plt.bar(
    np.arange(
        len(plot_df)
    ),
    plot_df[
        "error_detection_auroc"
    ],
)

plt.xticks(
    np.arange(
        len(plot_df)
    ),
    plot_df[
        "analysis"
    ],
    rotation=25,
    ha="right",
)

plt.ylabel(
    "Error-detection AUROC"
)

plt.title(
    "Patient-level uncertainty error detection"
)

plt.ylim(
    0.0,
    1.0,
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_38_ERROR_DETECTION_AUROC.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_ERROR_DETECTION_AUROC.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_38_ERROR_DETECTION_AUROC_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP38_COMPLETE",

    "dataset":
        "CBIS-DDSM",

    "record_count":
        int(
            len(records)
        ),

    "patient_count":
        int(
            len(patient)
        ),

    "uncertainty_methods":
        [
            "MLP predictive entropy",
            "VQC predictive entropy",
            "VQC finite-shot variance",
            "VQC parameter-induced variance",
        ],

    "primary_parameter_sigma":
        0.10,

    "primary_shot_budget":
        4096,

    "record_level":
        record_summary_rows,

    "patient_level":
        patient_summary_rows,

    "interpretation":
        (
            "Selective prediction evaluates whether low-uncertainty "
            "predictions can be preferentially retained to reduce "
            "classification risk. Parameter-induced variability is "
            "reported as model sensitivity and is not treated as a "
            "Bayesian posterior."
        ),

    "artifacts":
        {

            "record_source":
                str(
                    SOURCE_DIR
                    / "SELECTIVE_RECORD_LEVEL_SOURCE_DATA.csv"
                ),

            "patient_source":
                str(
                    SOURCE_DIR
                    / "SELECTIVE_PATIENT_LEVEL_SOURCE_DATA.csv"
                ),

            "summary":
                str(
                    TABLE_DIR
                    / "TABLE_38_SELECTIVE_SUMMARY.csv"
                ),

            "coverage":
                str(
                    TABLE_DIR
                    / "TABLE_38_RISK_AT_COVERAGE.csv"
                ),

            "curves":
                str(
                    SOURCE_DIR
                    / "RISK_COVERAGE_CURVE_SOURCE_DATA.csv"
                ),

            "figures":
                str(
                    FIG_DIR
                ),
        },
}


(
    METRIC_DIR
    / "STEP38_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CONFIGURATION
# ============================================================

config = {

    "seed":
        SEED,

    "primary_shots":
        4096,

    "primary_parameter_sigma":
        0.10,

    "uncertainty_methods":
        [
            "MLP entropy",
            "VQC entropy",
            "VQC shot variance",
            "VQC parameter variance",
        ],

    "risk_coverage":
        True,

    "error_detection_auroc":
        True,

    "coverage_targets":
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
        ],

    "source_runs":
        {
            "34B":
                str(RUN34B),

            "35A":
                str(RUN35A),

            "35B":
                str(RUN35B),
        },

    "inference_only":
        True,
}


(
    CONFIG_DIR
    / "STEP38_CONFIGURATION.json"
).write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


environment = {

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "python":
        sys.version,

    "platform":
        platform.platform(),

    "numpy":
        np.__version__,

    "pandas":
        pd.__version__,

}


(
    CONFIG_DIR
    / "ENVIRONMENT.json"
).write_text(
    json.dumps(
        environment,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SHA256
# ============================================================

hash_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
        continue

    hash_rows.append({

        "relative_path":
            str(
                path.relative_to(
                    OUT
                )
            ),

        "bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(
                path
            ),
    })


pd.DataFrame(
    hash_rows
).to_csv(
    OUT
    / "SHA256_INVENTORY.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print(
    "STEP 38 COMPLETE"
)
print("=" * 100)

print()
print(
    "Summary:",
    TABLE_DIR
    / "TABLE_38_SELECTIVE_SUMMARY.csv",
)

print(
    "Risk-coverage:",
    TABLE_DIR
    / "TABLE_38_RISK_AT_COVERAGE.csv",
)

print(
    "Source:",
    SOURCE_DIR
    / "SELECTIVE_PATIENT_LEVEL_SOURCE_DATA.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print(
    "Results:",
    METRIC_DIR
    / "STEP38_FINAL_RESULTS.json",
)

print()
print(
    "STATUS: STEP38_COMPLETE"
)