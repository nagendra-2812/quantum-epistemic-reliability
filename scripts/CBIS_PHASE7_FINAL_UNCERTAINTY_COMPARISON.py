from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

VQC_FILE = (
    ROOT
    / "experiments"
    / "cbis_phase7_quantum_uncertainty"
    / "PHASE7_VQC_UNCERTAINTY_PER_RECORD.csv"
)

MC_FILE = (
    ROOT
    / "experiments"
    / "cbis_matched_mc_dropout_224"
    / "MATCHED_MC_DROPOUT_224_PER_RECORD.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_phase7_final_uncertainty_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def safe_auc(y, score):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_auprc(y, score):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y,
            score,
        )
    )


def entropy(p):

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-7,
        1.0 - 1e-7,
    )

    return -(
        p * np.log(p)
        + (1.0 - p)
        * np.log(1.0 - p)
    )


def aggregate_patient(
    df,
    probability_column,
    uncertainty_columns,
):

    grouped = (
        df.groupby(
            "patient_id",
            sort=True,
        )
        .agg(
            label=(
                "label",
                "first",
            ),
            probability=(
                probability_column,
                "mean",
            ),
            record_count=(
                probability_column,
                "size",
            ),
            **{
                column: (
                    column,
                    "mean",
                )
                for column in uncertainty_columns
            },
        )
        .reset_index()
    )

    return grouped


def evaluate(
    df,
    score_columns,
):

    y = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    prediction = (
        df[
            "probability"
        ].to_numpy(
            dtype=float
        )
        >= 0.5
    ).astype(int)

    error = (
        prediction
        != y
    ).astype(int)

    results = {}

    for name, column in score_columns.items():

        score = df[
            column
        ].to_numpy(
            dtype=float
        )

        results[name] = {
            "error_detection_auroc":
                safe_auc(
                    error,
                    score,
                ),

            "error_detection_auprc":
                safe_auprc(
                    error,
                    score,
                ),

            "mean_uncertainty_correct":
                float(
                    score[
                        error == 0
                    ].mean()
                ),

            "mean_uncertainty_incorrect":
                float(
                    score[
                        error == 1
                    ].mean()
                ),
        }

    return results


def main():

    print("=" * 80)
    print(
        "FINAL PATIENT-LEVEL UNCERTAINTY COMPARISON"
    )
    print("=" * 80)

    vqc = pd.read_csv(
        VQC_FILE
    )

    mc = pd.read_csv(
        MC_FILE
    )

    # --------------------------------------------------------
    # Internal-test sanity checks
    # --------------------------------------------------------

    if len(vqc) != 602:
        raise RuntimeError(
            f"Expected 602 VQC records, got {len(vqc)}"
        )

    if len(mc) != 602:
        raise RuntimeError(
            f"Expected 602 MC records, got {len(mc)}"
        )

    vqc_keys = vqc[
        [
            "patient_id",
            "label",
        ]
    ].reset_index(
        drop=True
    )

    mc_keys = mc[
        [
            "patient_id",
            "label",
        ]
    ].reset_index(
        drop=True
    )

    if not vqc_keys.equals(
        mc_keys
    ):

        raise RuntimeError(
            "VQC and MC-Dropout records are not aligned."
        )

    # --------------------------------------------------------
    # Add common predictive entropy
    # --------------------------------------------------------

    vqc[
        "predictive_entropy"
    ] = entropy(
        vqc[
            "ideal_probability"
        ].to_numpy(
            dtype=float
        )
    )

    mc[
        "predictive_entropy"
    ] = entropy(
        mc[
            "mean_probability"
        ].to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Patient aggregation
    # --------------------------------------------------------

    vqc_patient = aggregate_patient(
        vqc,
        "ideal_probability",
        [
            "predictive_entropy",
            "shots_100_probability_variance",
            "shots_500_probability_variance",
            "shots_1000_probability_variance",
            "parameter_ensemble_probability_variance",
            "epistemic_information_gain",
            "parameter_ensemble_predictive_entropy",
            "parameter_ensemble_probability_std",
        ],
    )

    mc_patient = aggregate_patient(
        mc,
        "mean_probability",
        [
            "predictive_entropy",
            "predictive_variance",
            "predictive_std",
            "expected_entropy",
            "mutual_information",
        ],
    )

    # --------------------------------------------------------
    # Patient count
    # --------------------------------------------------------

    if len(vqc_patient) != 235:
        raise RuntimeError(
            f"Expected 235 VQC patients, got {len(vqc_patient)}"
        )

    if len(mc_patient) != 235:
        raise RuntimeError(
            f"Expected 235 MC patients, got {len(mc_patient)}"
        )

    if not vqc_patient[
        [
            "patient_id",
            "label",
        ]
    ].equals(
        mc_patient[
            [
                "patient_id",
                "label",
            ]
        ]
    ):

        raise RuntimeError(
            "Patient-level VQC and MC cohorts do not match."
        )

    # --------------------------------------------------------
    # VQC uncertainty comparison
    # --------------------------------------------------------

    vqc_scores = evaluate(
        vqc_patient,
        {
            "predictive_entropy":
                "predictive_entropy",

            "shot_variance_100":
                "shots_100_probability_variance",

            "shot_variance_500":
                "shots_500_probability_variance",

            "shot_variance_1000":
                "shots_1000_probability_variance",

            "parameter_variance":
                "parameter_ensemble_probability_variance",

            "epistemic_information_gain":
                "epistemic_information_gain",

            "parameter_predictive_entropy":
                "parameter_ensemble_predictive_entropy",

            "parameter_std":
                "parameter_ensemble_probability_std",
        },
    )

    # --------------------------------------------------------
    # MC-Dropout uncertainty comparison
    # --------------------------------------------------------

    mc_scores = evaluate(
        mc_patient,
        {
            "predictive_entropy":
                "predictive_entropy",

            "predictive_variance":
                "predictive_variance",

            "predictive_std":
                "predictive_std",

            "expected_entropy":
                "expected_entropy",

            "mutual_information":
                "mutual_information",
        },
    )

    # --------------------------------------------------------
    # Direct common-measure comparison
    # --------------------------------------------------------

    common = {
        "vqc_predictive_entropy":
            vqc_scores[
                "predictive_entropy"
            ],

        "mc_dropout_predictive_entropy":
            mc_scores[
                "predictive_entropy"
            ],
    }

    # --------------------------------------------------------
    # Save patient predictions
    # --------------------------------------------------------

    vqc_patient[
        "model"
    ] = "vqc_6q_depth2"

    mc_patient[
        "model"
    ] = "matched_mc_dropout_224"

    patient_file = (
        OUT
        / "PATIENT_LEVEL_UNCERTAINTY_COMPARISON.csv"
    )

    combined = pd.concat(
        [
            vqc_patient,
            mc_patient,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        patient_file,
        index=False,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    summary = {
        "experiment":
            "Final patient-level uncertainty comparison",

        "internal_test_patients":
            235,

        "internal_test_records":
            602,

        "vqc_parameters":
            224,

        "mc_dropout_parameters":
            224,

        "primary_common_measure":
            "predictive entropy",

        "vqc_error_detection":
            vqc_scores,

        "mc_dropout_error_detection":
            mc_scores,

        "direct_common_comparison":
            common,

        "interpretation": {
            "predictive_entropy":
                "common uncertainty measure used for direct comparison",

            "vqc_shot_variance":
                "finite-shot measurement variability",

            "vqc_epistemic_information_gain":
                "epistemic-style estimator under the explicitly defined local parameter perturbation model",

            "mc_mutual_information":
                "MC-Dropout model-disagreement quantity",

            "caution":
                "model-specific uncertainty quantities are reported separately and are not treated as physically identical uncertainty sources",
        },

        "status":
            "PHASE7_FINAL_COMPARISON_COMPLETE",
    }

    results_file = (
        OUT
        / "PHASE7_FINAL_UNCERTAINTY_COMPARISON.json"
    )

    results_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print(
        "Patients:",
        len(vqc_patient),
    )

    print()
    print(
        "COMMON PREDICTIVE-ENTROPY COMPARISON"
    )

    print(
        "VQC error-detection AUROC:",
        common[
            "vqc_predictive_entropy"
        ][
            "error_detection_auroc"
        ],
    )

    print(
        "MC-Dropout error-detection AUROC:",
        common[
            "mc_dropout_predictive_entropy"
        ][
            "error_detection_auroc"
        ],
    )

    print()
    print(
        "VQC UNCERTAINTY"
    )

    for name, value in vqc_scores.items():

        print(
            f"{name}: "
            f"AUROC={value['error_detection_auroc']:.6f} "
            f"AUPRC={value['error_detection_auprc']:.6f}"
        )

    print()
    print(
        "MC-DROPOUT UNCERTAINTY"
    )

    for name, value in mc_scores.items():

        print(
            f"{name}: "
            f"AUROC={value['error_detection_auroc']:.6f} "
            f"AUPRC={value['error_detection_auprc']:.6f}"
        )

    print()
    print(
        "Patient-level uncertainty:",
        patient_file,
    )

    print(
        "Results:",
        results_file,
    )

    print()
    print(
        "STATUS: PHASE7_FINAL_COMPARISON_COMPLETE"
    )


if __name__ == "__main__":
    main()