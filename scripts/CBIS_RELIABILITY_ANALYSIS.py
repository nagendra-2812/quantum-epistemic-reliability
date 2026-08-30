from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

VQC_FILE = (
    ROOT
    / "experiments"
    / "cbis_vqc_cpu_pilot"
    / "VQC_PILOT_PREDICTIONS.csv"
)

CLASSICAL_FILE = (
    ROOT
    / "experiments"
    / "cbis_matched_classical_224"
    / "MATCHED_CLASSICAL_224_PREDICTIONS.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_reliability_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRICS
# ============================================================

def ece(
    y,
    p,
    bins=10,
):

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(y)
    value = 0.0

    for i in range(bins):

        if i == bins - 1:

            mask = (
                (p >= edges[i])
                & (p <= edges[i + 1])
            )

        else:

            mask = (
                (p >= edges[i])
                & (p < edges[i + 1])
            )

        n = int(mask.sum())

        if n == 0:
            continue

        pred = (
            p[mask] >= 0.5
        ).astype(int)

        accuracy = float(
            np.mean(
                pred == y[mask]
            )
        )

        confidence = float(
            np.mean(
                np.maximum(
                    p[mask],
                    1.0 - p[mask],
                )
            )
        )

        value += (
            n / total
        ) * abs(
            accuracy - confidence
        )

    return float(value)


def binary_entropy(p):

    p = np.clip(
        np.asarray(p, dtype=float),
        1e-7,
        1.0 - 1e-7,
    )

    return -(
        p * np.log(p)
        + (1.0 - p)
        * np.log(1.0 - p)
    )


def risk_coverage(
    y,
    p,
    points=101,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    prediction = (
        p >= 0.5
    ).astype(int)

    error = (
        prediction != y
    ).astype(float)

    confidence = np.maximum(
        p,
        1.0 - p,
    )

    order = np.argsort(
        -confidence
    )

    error = error[
        order
    ]

    cumulative = np.cumsum(
        error
    )

    n = len(y)

    coverages = np.linspace(
        1.0 / n,
        1.0,
        points,
    )

    risks = []

    for coverage in coverages:

        k = max(
            1,
            int(
                round(
                    coverage * n
                )
            ),
        )

        risks.append(
            float(
                cumulative[k - 1]
                / k
            )
        )

    aurc = float(
        np.trapezoid(
            risks,
            coverages,
        )
    )

    return {
        "coverage":
            coverages.tolist(),
        "risk":
            risks,
        "aurc":
            aurc,
    }


def summarize(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return {
        "n":
            int(len(y)),

        "positive_count":
            int(y.sum()),

        "negative_count":
            int(len(y) - y.sum()),

        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "nll":
            float(
                log_loss(
                    y,
                    np.clip(
                        p,
                        1e-7,
                        1.0 - 1e-7,
                    ),
                    labels=[0, 1],
                )
            ),

        "ece_10bin":
            ece(
                y,
                p,
            ),

        "mean_entropy":
            float(
                binary_entropy(
                    p
                ).mean()
            ),

        "mean_confidence":
            float(
                np.maximum(
                    p,
                    1.0 - p,
                ).mean()
            ),

        "aurc":
            float(
                risk_coverage(
                    y,
                    p,
                )["aurc"]
            ),
    }


# ============================================================
# LOAD
# ============================================================

def load_file(
    path,
    probability_name,
):

    df = pd.read_csv(
        path
    )

    required = {
        "split",
        "patient_id",
        "label",
        "probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {missing}"
        )

    df = df.rename(
        columns={
            "probability":
                probability_name
        }
    )

    return df[
        [
            "split",
            "patient_id",
            "label",
            probability_name,
        ]
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CBIS-DDSM RELIABILITY ANALYSIS")
    print("=" * 80)

    vqc = load_file(
        VQC_FILE,
        "vqc_probability",
    )

    classical = load_file(
        CLASSICAL_FILE,
        "classical_probability",
    )

    print(
        "VQC rows:",
        len(vqc),
    )

    print(
        "Classical rows:",
        len(classical),
    )

    if len(vqc) != len(classical):
        raise RuntimeError(
            "Prediction files have different row counts."
        )

    # --------------------------------------------------------
    # Record-level alignment.
    #
    # Duplicate patient IDs are EXPECTED.
    # We therefore verify row-wise identity of the
    # split/patient/label keys rather than uniqueness.
    # --------------------------------------------------------

    key_vqc = vqc[
        [
            "split",
            "patient_id",
            "label",
        ]
    ].reset_index(
        drop=True
    )

    key_classical = classical[
        [
            "split",
            "patient_id",
            "label",
        ]
    ].reset_index(
        drop=True
    )

    if not key_vqc.equals(
        key_classical
    ):

        raise RuntimeError(
            "VQC and classical prediction rows are not aligned "
            "in split/patient/label order."
        )

    merged = key_vqc.copy()

    merged[
        "vqc_probability"
    ] = vqc[
        "vqc_probability"
    ].to_numpy()

    merged[
        "classical_probability"
    ] = classical[
        "classical_probability"
    ].to_numpy()

    print()
    print(
        "Prediction alignment: PASS"
    )

    # --------------------------------------------------------
    # Split audit
    # --------------------------------------------------------

    split_counts = (
        merged["split"]
        .value_counts()
        .to_dict()
    )

    print(
        "Record split counts:",
        split_counts,
    )

    assert split_counts.get(
        "calibration",
        0,
    ) == 537

    assert split_counts.get(
        "internal_test",
        0,
    ) == 602

    # --------------------------------------------------------
    # Record-level results
    # --------------------------------------------------------

    record_results = {}

    for model, column in [
        (
            "matched_classical_224",
            "classical_probability",
        ),
        (
            "vqc_6q_depth2",
            "vqc_probability",
        ),
    ]:

        record_results[model] = {}

        for split in [
            "calibration",
            "internal_test",
        ]:

            x = merged[
                merged["split"] == split
            ]

            record_results[model][split] = (
                summarize(
                    x["label"].to_numpy(),
                    x[column].to_numpy(),
                )
            )

    # --------------------------------------------------------
    # Patient-level aggregation
    #
    # Primary policy:
    # arithmetic mean probability across all
    # records belonging to the patient.
    # --------------------------------------------------------

    patient_frames = []

    for model, column in [
        (
            "matched_classical_224",
            "classical_probability",
        ),
        (
            "vqc_6q_depth2",
            "vqc_probability",
        ),
    ]:

        grouped = (
            merged
            .groupby(
                [
                    "split",
                    "patient_id",
                ],
                sort=True,
            )
            .agg(
                label=(
                    "label",
                    "first"
                ),
                probability=(
                    column,
                    "mean"
                ),
                record_count=(
                    column,
                    "size"
                ),
            )
            .reset_index()
        )

        grouped[
            "model"
        ] = model

        patient_frames.append(
            grouped
        )

    patient = pd.concat(
        patient_frames,
        ignore_index=True,
    )

    patient_results = {}

    for model in [
        "matched_classical_224",
        "vqc_6q_depth2",
    ]:

        patient_results[model] = {}

        for split in [
            "calibration",
            "internal_test",
        ]:

            x = patient[
                (patient["model"] == model)
                & (
                    patient["split"] == split
                )
            ]

            patient_results[model][split] = (
                summarize(
                    x["label"].to_numpy(),
                    x["probability"].to_numpy(),
                )
            )

    # --------------------------------------------------------
    # Patient-count audit
    # --------------------------------------------------------

    patient_counts = (
        patient
        .groupby(
            [
                "model",
                "split",
            ]
        )["patient_id"]
        .nunique()
        .to_dict()
    )

    print()
    print(
        "Patient counts:"
    )

    for key, value in patient_counts.items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Save aligned record predictions
    # --------------------------------------------------------

    aligned_file = (
        OUT
        / "ALIGNED_RECORD_PREDICTIONS.csv"
    )

    merged.to_csv(
        aligned_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save patient predictions
    # --------------------------------------------------------

    patient_file = (
        OUT
        / "PATIENT_LEVEL_PREDICTIONS_MEAN.csv"
    )

    patient.to_csv(
        patient_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment":
            "CBIS-DDSM reliability analysis",

        "prediction_unit":
            "patient-primary / record-secondary",

        "patient_aggregation":
            "mean probability across patient records",

        "record_split_counts":
            split_counts,

        "patient_counts":
            {
                str(k): int(v)
                for k, v in patient_counts.items()
            },

        "record_level":
            record_results,

        "patient_level":
            patient_results,

        "primary_evaluation":
            "patient_level",

        "status":
            "COMPLETE",
    }

    results_file = (
        OUT
        / "RELIABILITY_RESULTS.json"
    )

    results_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PRIMARY PATIENT-LEVEL RESULTS")
    print("=" * 80)

    for model in [
        "matched_classical_224",
        "vqc_6q_depth2",
    ]:

        test = patient_results[
            model
        ][
            "internal_test"
        ]

        print()
        print(
            model
        )

        print(
            "  n patients:",
            test["n"],
        )

        print(
            "  ROC-AUC:",
            test["roc_auc"],
        )

        print(
            "  AUPRC:",
            test["auprc"],
        )

        print(
            "  Brier:",
            test["brier"],
        )

        print(
            "  NLL:",
            test["nll"],
        )

        print(
            "  ECE:",
            test["ece_10bin"],
        )

        print(
            "  Mean entropy:",
            test["mean_entropy"],
        )

        print(
            "  Mean confidence:",
            test["mean_confidence"],
        )

        print(
            "  AURC:",
            test["aurc"],
        )

    print()
    print(
        "Aligned record predictions:",
        aligned_file,
    )

    print(
        "Patient-level predictions:",
        patient_file,
    )

    print(
        "Results:",
        results_file,
    )

    print()
    print(
        "STATUS: RELIABILITY ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()