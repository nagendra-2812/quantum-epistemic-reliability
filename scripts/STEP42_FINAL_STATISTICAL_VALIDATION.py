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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 2026
N_BOOTSTRAP = 2000
EPS = 1e-8

PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

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

RUN38 = (
    ROOT
    / "experiments"
    / "STEP38_SELECTIVE_PREDICTION"
)

RUN40 = (
    ROOT
    / "experiments"
    / "STEP40_QUANTUM_NOISE_ROBUSTNESS"
)

RUN41B = (
    ROOT
    / "experiments"
    / "STEP41B_BREAKHIS_EXTERNAL_INFERENCE"
)

RUN41C = (
    ROOT
    / "experiments"
    / "STEP41C_THERMOGRAPHY_EXTERNAL_INFERENCE"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP42_FINAL_STATISTICAL_VALIDATION"
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


def safe_auc(
    y,
    p,
):

    try:

        if len(
            np.unique(
                y
            )
        ) < 2:

            return float(
                "nan"
            )

        return float(
            roc_auc_score(
                y,
                p,
            )
        )

    except Exception:

        return float(
            "nan"
        )


def safe_auprc(
    y,
    p,
):

    try:

        if len(
            np.unique(
                y
            )
        ) < 2:

            return float(
                "nan"
            )

        return float(
            average_precision_score(
                y,
                p,
            )
        )

    except Exception:

        return float(
            "nan"
        )


def brier(
    y,
    p,
):

    return float(
        np.mean(
            (
                p - y
            ) ** 2
        )
    )


def nll(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return float(
        -np.mean(
            y * np.log(p)
            +
            (
                1.0 - y
            )
            *
            np.log(
                1.0 - p
            )
        )
    )


def ece(
    y,
    p,
    bins=10,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(y)
    value = 0.0

    for i in range(
        bins
    ):

        lo = edges[i]
        hi = edges[i + 1]

        if i == bins - 1:

            mask = (
                (p >= lo)
                &
                (p <= hi)
            )

        else:

            mask = (
                (p >= lo)
                &
                (p < hi)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        accuracy = float(
            y[mask].mean()
        )

        confidence = float(
            p[mask].mean()
        )

        value += (
            count / total
        ) * abs(
            accuracy - confidence
        )

    return float(
        value
    )


def metric_value(
    metric,
    y,
    p,
):

    if metric == "roc_auc":

        return safe_auc(
            y,
            p,
        )

    if metric == "auprc":

        return safe_auprc(
            y,
            p,
        )

    if metric == "brier":

        return brier(
            y,
            p,
        )

    if metric == "nll":

        return nll(
            y,
            p,
        )

    if metric == "ece":

        return ece(
            y,
            p,
        )

    if metric == "accuracy":

        return float(
            np.mean(
                (
                    p >= 0.5
                )
                ==
                y
            )
        )

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def bootstrap_metric(
    y,
    p,
    metric,
    rng,
    n_bootstrap=N_BOOTSTRAP,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    n = len(y)

    values = []

    for _ in range(
        n_bootstrap
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        value = metric_value(
            metric,
            y[idx],
            p[idx],
        )

        if np.isfinite(
            value
        ):

            values.append(
                value
            )

    if not values:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(
            metric_value(
                metric,
                y,
                p,
            )
        ),
        float(low),
        float(high),
    )


def grouped_patient_frame(
    frame,
    patient_col,
    label_col,
    probability_col,
):

    if patient_col not in frame.columns:

        raise RuntimeError(
            f"Patient column missing: {patient_col}"
        )

    rows = []

    for patient_id, group in frame.groupby(
        patient_col
    ):

        labels = (
            group[
                label_col
            ]
            .astype(int)
            .to_numpy()
        )

        probabilities = (
            group[
                probability_col
            ]
            .astype(float)
            .to_numpy()
        )

        if len(
            np.unique(
                labels
            )
        ) != 1:

            raise RuntimeError(
                f"Patient has inconsistent labels: {patient_id}"
            )

        rows.append({

            "patient_id":
                str(patient_id),

            "label":
                int(
                    labels[0]
                ),

            "probability":
                float(
                    probabilities.mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


def find_file(
    candidates,
):

    for path in candidates:

        if path.is_file():

            return path

    return None


def find_prediction_columns(
    frame,
):

    cols = {
        c.lower(): c
        for c in frame.columns
    }

    label_candidates = [
        "label",
        "binary_label",
        "y",
        "target",
    ]

    probability_candidates = [
        "probability",
        "predicted_probability",
        "mlp_probability",
        "vqc_probability",
        "mean_probability",
    ]

    patient_candidates = [
        "patient_id",
        "patient",
        "subject_id",
        "participant_id",
    ]

    label_col = None
    probability_col = None
    patient_col = None

    for c in label_candidates:

        if c in cols:

            label_col = cols[c]
            break

    for c in probability_candidates:

        if c in cols:

            probability_col = cols[c]
            break

    for c in patient_candidates:

        if c in cols:

            patient_col = cols[c]
            break

    return (
        label_col,
        probability_col,
        patient_col,
    )


def analyse_dataset(
    frame,
    dataset_name,
    model_name,
    label_col,
    probability_col,
    patient_col=None,
):

    y = (
        frame[
            label_col
        ]
        .astype(int)
        .to_numpy()
    )

    p = (
        frame[
            probability_col
        ]
        .astype(float)
        .to_numpy()
    )

    rng = np.random.default_rng(
        SEED
        +
        abs(
            hash(
                dataset_name
                +
                model_name
            )
        )
        %
        100000
    )

    rows = []

    for metric in [
        "roc_auc",
        "auprc",
        "accuracy",
        "brier",
        "nll",
        "ece",
    ]:

        estimate, low, high = (
            bootstrap_metric(
                y,
                p,
                metric,
                rng,
            )
        )

        rows.append({

            "dataset":
                dataset_name,

            "model":
                model_name,

            "level":
                "record",

            "metric":
                metric,

            "n":
                len(y),

            "estimate":
                estimate,

            "ci95_low":
                low,

            "ci95_high":
                high,

        })


    if patient_col is not None:

        patient_df = (
            grouped_patient_frame(
                frame,
                patient_col,
                label_col,
                probability_col,
            )
        )

        py = patient_df[
            "label"
        ].to_numpy()

        pp = patient_df[
            "probability"
        ].to_numpy()

        patient_rng = np.random.default_rng(
            SEED
            +
            100000
            +
            abs(
                hash(
                    dataset_name
                    +
                    model_name
                )
            )
            %
            100000
        )

        for metric in [
            "roc_auc",
            "auprc",
            "accuracy",
            "brier",
            "nll",
            "ece",
        ]:

            estimate, low, high = (
                bootstrap_metric(
                    py,
                    pp,
                    metric,
                    patient_rng,
                )
            )

            rows.append({

                "dataset":
                    dataset_name,

                "model":
                    model_name,

                "level":
                    "patient",

                "metric":
                    metric,

                "n":
                    len(py),

                "estimate":
                    estimate,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

            })

    return rows


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print(
    "STEP 42 - FINAL STATISTICAL VALIDATION"
)
print("=" * 100)

all_rows = []


# ============================================================
# CBIS 34B
# ============================================================

print()
print(
    "CBIS-DDSM"
)

mlp_path = find_file([
    RUN34B
    / "predictions"
    / "MLP_RECORD_LEVEL_PREDICTIONS.csv",
    RUN34B
    / "predictions"
    / "MLP_PREDICTIONS.csv",
])

vqc_path = find_file([
    RUN34B
    / "predictions"
    / "VQC_RECORD_LEVEL_PREDICTIONS.csv",
    RUN34B
    / "predictions"
    / "VQC_PREDICTIONS.csv",
])

if mlp_path is not None:

    mlp = pd.read_csv(
        mlp_path
    )

    (
        label_col,
        probability_col,
        patient_col,
    ) = find_prediction_columns(
        mlp
    )

    if (
        label_col is not None
        and
        probability_col is not None
    ):

        all_rows.extend(
            analyse_dataset(
                mlp,
                "CBIS-DDSM",
                "MLP",
                label_col,
                probability_col,
                patient_col,
            )
        )


if vqc_path is not None:

    vqc = pd.read_csv(
        vqc_path
    )

    (
        label_col,
        probability_col,
        patient_col,
    ) = find_prediction_columns(
        vqc
    )

    if (
        label_col is not None
        and
        probability_col is not None
    ):

        all_rows.extend(
            analyse_dataset(
                vqc,
                "CBIS-DDSM",
                "VQC",
                label_col,
                probability_col,
                patient_col,
            )
        )


# ============================================================
# BREAKHIS
# ============================================================

print()
print(
    "BreaKHis"
)

breakhis_path = (
    RUN41B
    / "source_data"
    / "BREAKHIS_EXTERNAL_PREDICTIONS.csv"
)

if breakhis_path.is_file():

    bh = pd.read_csv(
        breakhis_path
    )

    for model_name, probability_column in [

        (
            "MLP",
            "mlp_probability",
        ),

        (
            "VQC",
            "vqc_probability",
        ),

    ]:

        if probability_column not in bh.columns:

            continue

        all_rows.extend(
            analyse_dataset(
                bh,
                "BreaKHis",
                model_name,
                "label",
                probability_column,
                "__patient",
            )
        )


# ============================================================
# DMR-IR
# ============================================================

print()
print(
    "DMR-IR Healthy/Sick"
)

thermo_path = (
    RUN41C
    / "source_data"
    / "THERMOGRAPHY_EXTERNAL_PREDICTIONS.csv"
)

if thermo_path.is_file():

    thermo = pd.read_csv(
        thermo_path
    )

    dmr = thermo[
        thermo[
            "source"
        ]
        ==
        "DMR_IR"
    ].copy()

    for model_name, probability_column in [

        (
            "MLP",
            "mlp_probability",
        ),

        (
            "VQC",
            "vqc_probability",
        ),

    ]:

        if probability_column not in dmr.columns:

            continue

        all_rows.extend(
            analyse_dataset(
                dmr,
                "DMR-IR Healthy/Sick",
                model_name,
                "label",
                probability_column,
                "subject_id",
            )
        )


# ============================================================
# MENDELEY
# ============================================================

print()
print(
    "Mendeley Benign/Malignant"
)

if thermo_path.is_file():

    mend = thermo[
        thermo[
            "source"
        ]
        ==
        "THERMOGRAPHY_BENIGN_MALIGNANT"
    ].copy()

    for model_name, probability_column in [

        (
            "MLP",
            "mlp_probability",
        ),

        (
            "VQC",
            "vqc_probability",
        ),

    ]:

        if probability_column not in mend.columns:

            continue

        all_rows.extend(
            analyse_dataset(
                mend,
                "Mendeley Benign/Malignant",
                model_name,
                "label",
                probability_column,
                "subject_id",
            )
        )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_rows
)

if len(
    results_df
) == 0:

    raise RuntimeError(
        "No compatible prediction sources were found."
    )


results_df = results_df.sort_values(
    [
        "dataset",
        "level",
        "model",
        "metric",
    ]
).reset_index(
    drop=True
)


# ============================================================
# DELTA ANALYSIS
# ============================================================

delta_rows = []

for dataset in (
    results_df[
        "dataset"
    ]
    .unique()
):

    for level in (
        results_df[
            "level"
        ]
        .unique()
    ):

        subset = results_df[
            (
                results_df[
                    "dataset"
                ]
                ==
                dataset
            )
            &
            (
                results_df[
                    "level"
                ]
                ==
                level
            )
        ]

        for metric in (
            subset[
                "metric"
            ]
            .unique()
        ):

            m = subset[
                subset[
                    "model"
                ]
                ==
                "MLP"
            ]

            v = subset[
                subset[
                    "model"
                ]
                ==
                "VQC"
            ]

            if len(m) == 1 and len(v) == 1:

                delta_rows.append({

                    "dataset":
                        dataset,

                    "level":
                        level,

                    "metric":
                        metric,

                    "mlp":
                        float(
                            m.iloc[0][
                                "estimate"
                            ]
                        ),

                    "vqc":
                        float(
                            v.iloc[0][
                                "estimate"
                            ]
                        ),

                    "vqc_minus_mlp":
                        float(
                            v.iloc[0][
                                "estimate"
                            ]
                            -
                            m.iloc[0][
                                "estimate"
                            ]
                        ),
                })


delta_df = pd.DataFrame(
    delta_rows
)


# ============================================================
# SAVE SOURCE / TABLES
# ============================================================

results_df.to_csv(
    TABLE_DIR
    / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv",
    index=False,
)

delta_df.to_csv(
    TABLE_DIR
    / "TABLE_42_VQC_MLP_DELTA_SUMMARY.csv",
    index=False,
)

results_df.to_csv(
    SOURCE_DIR
    / "FINAL_STATISTICAL_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# PRIMARY CBIS FIGURE
# ============================================================

primary = results_df[
    (
        results_df[
            "dataset"
        ]
        ==
        "CBIS-DDSM"
    )
    &
    (
        results_df[
            "level"
        ]
        ==
        "patient"
    )
    &
    (
        results_df[
            "metric"
        ]
        ==
        "roc_auc"
    )
].copy()

if len(primary):

    fig = plt.figure(
        figsize=(
            6.8,
            5.2,
        )
    )

    positions = np.arange(
        len(primary)
    )

    plt.bar(
        positions,
        primary[
            "estimate"
        ],
    )

    plt.errorbar(
        positions,
        primary[
            "estimate"
        ],
        yerr=[
            primary[
                "estimate"
            ]
            -
            primary[
                "ci95_low"
            ],

            primary[
                "ci95_high"
            ]
            -
            primary[
                "estimate"
            ],
        ],
        fmt="none",
        capsize=4,
    )

    plt.xticks(
        positions,
        primary[
            "model"
        ],
    )

    plt.ylabel(
        "Patient-level ROC-AUC"
    )

    plt.title(
        "CBIS-DDSM patient-level ROC-AUC with 95% bootstrap CI"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_CBIS_PATIENT_AUROC_CI.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_CBIS_PATIENT_AUROC_CI.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_CBIS_PATIENT_AUROC_CI_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# EXTERNAL AUROC FIGURE
# ============================================================

ext = results_df[
    (
        results_df[
            "level"
        ]
        ==
        "record"
    )
    &
    (
        results_df[
            "metric"
        ]
        ==
        "roc_auc"
    )
    &
    (
        results_df[
            "dataset"
        ]
        !=
        "CBIS-DDSM"
    )
].copy()

if len(ext):

    labels = [
        f"{a}\n{b}"
        for a, b
        in zip(
            ext[
                "dataset"
            ],
            ext[
                "model"
            ],
        )
    ]

    fig = plt.figure(
        figsize=(
            8.5,
            5.5,
        )
    )

    positions = np.arange(
        len(ext)
    )

    plt.bar(
        positions,
        ext[
            "estimate"
        ],
    )

    plt.errorbar(
        positions,
        ext[
            "estimate"
        ],
        yerr=[
            ext[
                "estimate"
            ]
            -
            ext[
                "ci95_low"
            ],

            ext[
                "ci95_high"
            ]
            -
            ext[
                "estimate"
            ],
        ],
        fmt="none",
        capsize=4,
    )

    plt.xticks(
        positions,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "External cross-domain transfer ROC-AUC"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_EXTERNAL_AUROC_CI.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_EXTERNAL_AUROC_CI.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_42_EXTERNAL_AUROC_CI_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FINAL JSON
# ============================================================

result = {

    "status":
        "STEP42_COMPLETE",

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "confidence_interval":
        0.95,

    "confidence_interval_method":
        "percentile bootstrap",

    "datasets":
        sorted(
            results_df[
                "dataset"
            ]
            .unique()
            .tolist()
        ),

    "models":
        sorted(
            results_df[
                "model"
            ]
            .unique()
            .tolist()
        ),

    "levels":
        sorted(
            results_df[
                "level"
            ]
            .unique()
            .tolist()
        ),

    "metrics":
        sorted(
            results_df[
                "metric"
            ]
            .unique()
            .tolist()
        ),

    "interpretation_note":
        (
            "Bootstrap confidence intervals quantify uncertainty "
            "around already-observed evaluation metrics. No "
            "training, model adaptation, calibration fitting, "
            "or hyperparameter optimization was performed in "
            "Step 42."
        ),

    "external_dataset_note":
        (
            "BreaKHis, DMR-IR, and Mendeley thermography results "
            "are external cross-domain or cross-modal transfer "
            "stress tests of the frozen CBIS-developed pipeline."
        ),

    "results":
        results_df.to_dict(
            orient="records"
        ),

    "vqc_minus_mlp":
        delta_df.to_dict(
            orient="records"
        ),

    "artifacts":
        {

            "bootstrap_table":
                str(
                    TABLE_DIR
                    / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv"
                ),

            "delta_table":
                str(
                    TABLE_DIR
                    / "TABLE_42_VQC_MLP_DELTA_SUMMARY.csv"
                ),

            "source":
                str(
                    SOURCE_DIR
                    / "FINAL_STATISTICAL_SOURCE_DATA.csv"
                ),

            "figures":
                str(
                    FIG_DIR
                ),
        },
}


(
    METRIC_DIR
    / "STEP42_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CONFIG
# ============================================================

config = {

    "seed":
        SEED,

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "confidence_level":
        0.95,

    "bootstrap_method":
        "percentile",

    "metrics":
        [
            "ROC-AUC",
            "AUPRC",
            "accuracy",
            "Brier",
            "NLL",
            "ECE",
        ],

    "levels":
        [
            "record",
            "patient",
        ],

    "training":
        False,

    "model_adaptation":
        False,

    "hyperparameter_optimization":
        False,

    "source_runs":
        {
            "34B":
                str(RUN34B),

            "35A":
                str(RUN35A),

            "35B":
                str(RUN35B),

            "37":
                str(RUN37),

            "38":
                str(RUN38),

            "40":
                str(RUN40),

            "41B":
                str(RUN41B),

            "41C":
                str(RUN41C),
        },
}


(
    CONFIG_DIR
    / "STEP42_CONFIGURATION.json"
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
    "STEP 42 COMPLETE"
)
print("=" * 100)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    "Bootstrap table:",
    TABLE_DIR
    / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv",
)

print(
    "Delta table:",
    TABLE_DIR
    / "TABLE_42_VQC_MLP_DELTA_SUMMARY.csv",
)

print(
    "Source:",
    SOURCE_DIR
    / "FINAL_STATISTICAL_SOURCE_DATA.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print(
    "Results:",
    METRIC_DIR
    / "STEP42_FINAL_RESULTS.json",
)

print()
print(
    "STATUS: STEP42_COMPLETE"
)