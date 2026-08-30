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

BASE_FILE = (
    ROOT
    / "experiments"
    / "cbis_reliability_analysis"
    / "PATIENT_LEVEL_PREDICTIONS_MEAN.csv"
)

VQC_FILE = (
    ROOT
    / "experiments"
    / "cbis_phase7_final_uncertainty_comparison"
    / "PATIENT_LEVEL_UNCERTAINTY_COMPARISON.csv"
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
    / "cbis_phase8_9_conformal_selective"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ALPHAS = (
    0.10,
    0.05,
)

COVERAGE_TARGETS = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)


# ============================================================
# METRICS
# ============================================================

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


def entropy_binary(p):

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


# ============================================================
# CONFORMAL
# ============================================================

def conformity_scores(
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

    true_probability = np.where(
        y == 1,
        p,
        1.0 - p,
    )

    return 1.0 - true_probability


def conformal_quantile(
    scores,
    alpha,
):

    scores = np.sort(
        np.asarray(
            scores,
            dtype=float,
        )
    )

    n = len(scores)

    # Standard split-conformal finite-sample index.
    k = int(
        np.ceil(
            (n + 1)
            * (1.0 - alpha)
        )
    )

    k = min(
        max(
            k,
            1,
        ),
        n,
    )

    return float(
        scores[k - 1]
    )


def make_prediction_set(
    p,
    q,
):

    include_zero = (
        1.0 - p
        >= 1.0 - q
    )

    include_one = (
        p
        >= 1.0 - q
    )

    if include_zero and include_one:
        return "0,1"

    if include_one:
        return "1"

    if include_zero:
        return "0"

    return "EMPTY"


def prediction_set_size(
    value,
):

    if value == "EMPTY":
        return 0

    if value == "0,1":
        return 2

    return 1


def prediction_sets(
    probabilities,
    q,
):

    return [
        make_prediction_set(
            p,
            q,
        )
        for p in probabilities
    ]


def conformal_summary(
    y,
    probabilities,
    q,
):

    sets = prediction_sets(
        probabilities,
        q,
    )

    sizes = np.asarray(
        [
            prediction_set_size(s)
            for s in sets
        ],
        dtype=int,
    )

    covered = []

    for truth, s in zip(
        y,
        sets,
    ):

        if s == "0":
            covered.append(
                truth == 0
            )

        elif s == "1":
            covered.append(
                truth == 1
            )

        elif s == "0,1":
            covered.append(
                True
            )

        else:
            covered.append(
                False
            )

    covered = np.asarray(
        covered,
        dtype=bool,
    )

    return {
        "n":
            int(len(y)),

        "coverage":
            float(
                covered.mean()
            ),

        "average_set_size":
            float(
                sizes.mean()
            ),

        "singleton_rate":
            float(
                np.mean(
                    sizes == 1
                )
            ),

        "ambiguous_rate":
            float(
                np.mean(
                    sizes == 2
                )
            ),

        "empty_rate":
            float(
                np.mean(
                    sizes == 0
                )
            ),

        "quantile":
            float(q),
    }


# ============================================================
# SELECTIVE RISK
# ============================================================

def selective_curve(
    y,
    probabilities,
    uncertainty,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    prediction = (
        probabilities >= 0.5
    ).astype(int)

    error = (
        prediction != y
    ).astype(float)

    # Higher uncertainty = more likely to reject.
    # Therefore accept cases from LOWEST uncertainty first.
    order = np.argsort(
        uncertainty
    )

    sorted_error = error[
        order
    ]

    sorted_uncertainty = uncertainty[
        order
    ]

    cumulative_error = np.cumsum(
        sorted_error
    )

    n = len(y)

    coverages = np.arange(
        1,
        n + 1,
    ) / n

    risks = (
        cumulative_error
        / np.arange(
            1,
            n + 1,
        )
    )

    aurc = float(
        np.trapezoid(
            risks,
            coverages,
        )
    )

    selected = {}

    for target in COVERAGE_TARGETS:

        k = max(
            1,
            int(
                round(
                    target * n
                )
            ),
        )

        k = min(
            k,
            n,
        )

        selected[
            f"{target:.2f}"
        ] = {
            "target_coverage":
                target,

            "actual_coverage":
                float(
                    k / n
                ),

            "risk":
                float(
                    risks[k - 1]
                ),

            "accepted":
                int(k),

            "mean_uncertainty":
                float(
                    sorted_uncertainty[
                        :k
                    ].mean()
                ),
        }

    return {
        "aurc":
            aurc,

        "curve": {
            "coverage":
                coverages.tolist(),

            "risk":
                risks.tolist(),
        },

        "risk_at_target_coverage":
            selected,
    }


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 80)
    print(
        "CBIS-DDSM PHASE 8 + 9"
    )
    print(
        "CONFORMAL + SELECTIVE RELIABILITY"
    )
    print("=" * 80)

    base = pd.read_csv(
        BASE_FILE
    )

    vqc = pd.read_csv(
        VQC_FILE
    )

    mc = pd.read_csv(
        MC_FILE
    )

    # --------------------------------------------------------
    # Base file
    # --------------------------------------------------------

    required_base = {
        "split",
        "patient_id",
        "label",
        "probability",
        "model",
    }

    missing = (
        required_base
        - set(base.columns)
    )

    if missing:
        raise RuntimeError(
            f"Base file missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Expected model rows
    # --------------------------------------------------------

    expected_models = {
        "matched_classical_224",
        "vqc_6q_depth2",
    }

    if (
        set(base["model"].unique())
        != expected_models
    ):

        raise RuntimeError(
            "Unexpected base models."
        )

    # --------------------------------------------------------
    # Construct model-specific patient tables
    # --------------------------------------------------------

    tables = {}

    for model in sorted(
        expected_models
    ):

        x = base[
            base["model"] == model
        ].copy()

        calibration = x[
            x["split"]
            == "calibration"
        ].copy()

        test = x[
            x["split"]
            == "internal_test"
        ].copy()

        if len(calibration) != 235:
            raise RuntimeError(
                f"{model}: expected 235 calibration patients, "
                f"found {len(calibration)}"
            )

        if len(test) != 235:
            raise RuntimeError(
                f"{model}: expected 235 test patients, "
                f"found {len(test)}"
            )

        tables[model] = {
            "calibration":
                calibration,
            "test":
                test,
        }

    # --------------------------------------------------------
    # VQC uncertainty
    # --------------------------------------------------------

    vqc_test = vqc.copy()

    if len(vqc_test) != 470:
        raise RuntimeError(
            "VQC uncertainty file should contain "
            "235 VQC + 235 MC patients."
        )

    vqc_test = vqc_test[
        vqc_test["model"]
        == "vqc_6q_depth2"
    ].copy()

    if len(vqc_test) != 235:
        raise RuntimeError(
            "Expected 235 VQC uncertainty patients."
        )

    # --------------------------------------------------------
    # MC-Dropout uncertainty
    # --------------------------------------------------------

    # The MC-Dropout uncertainty file is record-level.
    # Aggregate it to patient level using mean prediction
    # and mean uncertainty, matching the patient-primary
    # evaluation policy used elsewhere.
    #
    # Duplicate patient IDs are expected at record level.
    # --------------------------------------------------------

    required_mc = {
        "patient_id",
        "label",
        "mean_probability",
        "predictive_entropy",
        "predictive_variance",
        "expected_entropy",
        "mutual_information",
    }

    missing = (
        required_mc
        - set(mc.columns)
    )

    if missing:
        raise RuntimeError(
            f"MC file missing columns: {sorted(missing)}"
        )

    mc_patient = (
        mc.groupby(
            "patient_id",
            sort=True,
        )
        .agg(
            label=(
                "label",
                "first",
            ),
            mean_probability=(
                "mean_probability",
                "mean",
            ),
            predictive_entropy=(
                "predictive_entropy",
                "mean",
            ),
            predictive_variance=(
                "predictive_variance",
                "mean",
            ),
            expected_entropy=(
                "expected_entropy",
                "mean",
            ),
            mutual_information=(
                "mutual_information",
                "mean",
            ),
        )
        .reset_index()
    )

    if len(mc_patient) != 235:
        raise RuntimeError(
            f"Expected 235 MC patients, found {len(mc_patient)}"
        )

    # MC prediction rows correspond to internal test.
    # The file was built from internal-test latents only.
    # Verify labels against the canonical test cohort.
    canonical_mc_test = tables[
        "matched_classical_224"
    ][
        "test"
    ][
        [
            "patient_id",
            "label",
        ]
    ].sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    mc_alignment = mc_patient[
        [
            "patient_id",
            "label",
        ]
    ].sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    if not canonical_mc_test.equals(
        mc_alignment
    ):

        raise RuntimeError(
            "MC-Dropout patient cohort does not match canonical internal test."
        )

    # --------------------------------------------------------
    # Align VQC uncertainty to canonical VQC test
    # --------------------------------------------------------

    canonical_vqc_test = tables[
        "vqc_6q_depth2"
    ][
        "test"
    ][
        [
            "patient_id",
            "label",
        ]
    ].sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    vqc_alignment = vqc_test[
        [
            "patient_id",
            "label",
        ]
    ].sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    if not canonical_vqc_test.equals(
        vqc_alignment
    ):

        raise RuntimeError(
            "VQC uncertainty patient cohort does not match canonical internal test."
        )

    # --------------------------------------------------------
    # Final result containers
    # --------------------------------------------------------

    conformal_results = {}
    selective_results = {}
    output_frames = []

    # ========================================================
    # CLASSICAL CONFORMAL
    # ========================================================

    for model in [
        "matched_classical_224",
        "vqc_6q_depth2",
    ]:

        cal = tables[
            model
        ][
            "calibration"
        ]

        test = tables[
            model
        ][
            "test"
        ]

        y_cal = cal[
            "label"
        ].to_numpy(
            dtype=int
        )

        p_cal = cal[
            "probability"
        ].to_numpy(
            dtype=float
        )

        y_test = test[
            "label"
        ].to_numpy(
            dtype=int
        )

        p_test = test[
            "probability"
        ].to_numpy(
            dtype=float
        )

        scores = conformity_scores(
            y_cal,
            p_cal,
        )

        conformal_results[
            model
        ] = {}

        for alpha in ALPHAS:

            q = conformal_quantile(
                scores,
                alpha,
            )

            summary = conformal_summary(
                y_test,
                p_test,
                q,
            )

            summary[
                "alpha"
            ] = alpha

            summary[
                "target_coverage"
            ] = 1.0 - alpha

            conformal_results[
                model
            ][
                str(alpha)
            ] = summary

            sets = prediction_sets(
                p_test,
                q,
            )

            for pid, truth, probability, s in zip(
                test[
                    "patient_id"
                ],
                y_test,
                p_test,
                sets,
            ):

                output_frames.append({
                    "model":
                        model,

                    "alpha":
                        alpha,

                    "patient_id":
                        pid,

                    "label":
                        int(truth),

                    "probability":
                        float(probability),

                    "prediction_set":
                        s,

                    "set_size":
                        prediction_set_size(s),

                    "covered":
                        (
                            (
                                s == "0"
                                and truth == 0
                            )
                            or
                            (
                                s == "1"
                                and truth == 1
                            )
                            or
                            s == "0,1"
                        ),
                })

    # ========================================================
    # VQC SELECTIVE
    # ========================================================

    vqc_test_base = tables[
        "vqc_6q_depth2"
    ][
        "test"
    ][
        [
            "patient_id",
            "label",
            "probability",
        ]
    ].copy()

    vqc_u = vqc_test[
        [
            "patient_id",
            "predictive_entropy",
            "parameter_ensemble_predictive_entropy",
            "epistemic_information_gain",
            "parameter_ensemble_probability_variance",
            "shots_100_probability_variance",
            "shots_500_probability_variance",
            "shots_1000_probability_variance",
        ]
    ].copy()

    vqc_selective = vqc_test_base.merge(
        vqc_u,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(vqc_selective) != 235:
        raise RuntimeError(
            "VQC selective table should contain 235 patients."
        )

    vqc_selective_results = {}

    vqc_uncertainties = {
        "predictive_entropy":
            "predictive_entropy",

        "parameter_predictive_entropy":
            "parameter_ensemble_predictive_entropy",

        "epistemic_information_gain":
            "epistemic_information_gain",

        "parameter_variance":
            "parameter_ensemble_probability_variance",

        "shot_variance_100":
            "shots_100_probability_variance",

        "shot_variance_500":
            "shots_500_probability_variance",

        "shot_variance_1000":
            "shots_1000_probability_variance",
    }

    for name, column in (
        vqc_uncertainties.items()
    ):

        vqc_selective_results[name] = (
            selective_curve(
                vqc_selective[
                    "label"
                ].to_numpy(
                    dtype=int
                ),
                vqc_selective[
                    "probability"
                ].to_numpy(
                    dtype=float
                ),
                vqc_selective[
                    column
                ].to_numpy(
                    dtype=float
                ),
            )
        )

    # ========================================================
    # MC SELECTIVE
    # ========================================================

    mc_selective_results = {}

    # Match canonical classical test probability.
    classical_test = tables[
        "matched_classical_224"
    ][
        "test"
    ][
        [
            "patient_id",
            "label",
            "probability",
        ]
    ].copy()

    mc_selective = classical_test.merge(
        mc_patient,
        on=[
            "patient_id",
            "label",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(mc_selective) != 235:
        raise RuntimeError(
            "MC selective table should contain 235 patients."
        )

    mc_uncertainties = {
        "predictive_entropy":
            "predictive_entropy",

        "expected_entropy":
            "expected_entropy",

        "mutual_information":
            "mutual_information",

        "predictive_variance":
            "predictive_variance",
    }

    for name, column in (
        mc_uncertainties.items()
    ):

        mc_selective_results[name] = (
            selective_curve(
                mc_selective[
                    "label"
                ].to_numpy(
                    dtype=int
                ),
                mc_selective[
                    "probability"
                ].to_numpy(
                    dtype=float
                ),
                mc_selective[
                    column
                ].to_numpy(
                    dtype=float
                ),
            )
        )

    selective_results = {
        "vqc":
            vqc_selective_results,

        "mc_dropout":
            mc_selective_results,
    }

    # ========================================================
    # PRIMARY COMMON ENTROPY COMPARISON
    # ========================================================

    common_vqc = selective_curve(
        vqc_selective[
            "label"
        ].to_numpy(
            dtype=int
        ),
        vqc_selective[
            "probability"
        ].to_numpy(
            dtype=float
        ),
        vqc_selective[
            "predictive_entropy"
        ].to_numpy(
            dtype=float
        ),
    )

    common_mc = selective_curve(
        mc_selective[
            "label"
        ].to_numpy(
            dtype=int
        ),
        mc_selective[
            "probability"
        ].to_numpy(
            dtype=float
        ),
        mc_selective[
            "predictive_entropy"
        ].to_numpy(
            dtype=float
        ),
    )

    # ========================================================
    # SAVE CONFORMAL PREDICTIONS
    # ========================================================

    conformal_file = (
        OUT
        / "FINAL_CONFORMAL_PREDICTIONS.csv"
    )

    pd.DataFrame(
        output_frames
    ).to_csv(
        conformal_file,
        index=False,
    )

    # ========================================================
    # SAVE SELECTIVE PATIENT TABLE
    # ========================================================

    selective_file = (
        OUT
        / "FINAL_SELECTIVE_PATIENT_DATA.csv"
    )

    vqc_export = vqc_selective.copy()

    vqc_export[
        "model"
    ] = "vqc_6q_depth2"

    mc_export = mc_selective.copy()

    mc_export[
        "model"
    ] = "matched_mc_dropout_224"

    combined_selective = pd.concat(
        [
            vqc_export,
            mc_export,
        ],
        ignore_index=True,
    )

    combined_selective.to_csv(
        selective_file,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    results = {
        "experiment":
            "CBIS-DDSM final Phase 8-9 conformal and selective reliability",

        "calibration_patients":
            235,

        "internal_test_patients":
            235,

        "models": [
            "matched_classical_224",
            "vqc_6q_depth2",
        ],

        "conformal":
            conformal_results,

        "selective_prediction": {
            "primary_common_measure":
                "predictive entropy",

            "vqc_predictive_entropy":
                common_vqc,

            "mc_dropout_predictive_entropy":
                common_mc,

            "all_vqc_uncertainties":
                vqc_selective_results,

            "all_mc_uncertainties":
                mc_selective_results,
        },

        "interpretation": {
            "conformal":
                "split-conformal prediction fitted on the calibration partition and evaluated on the untouched internal-test partition",

            "selective":
                "cases are accepted in increasing order of uncertainty; risk is empirical error rate among accepted cases",

            "uncertainty_comparison":
                "predictive entropy is the common direct comparison; model-specific uncertainty quantities remain separately reported",

            "cmmd":
                "not used for model selection or threshold tuning",
        },

        "files": {
            "conformal":
                str(conformal_file),

            "selective":
                str(selective_file),
        },

        "status":
            "PHASE8_9_COMPLETE",
    }

    results_file = (
        OUT
        / "PHASE8_9_FINAL_RESULTS.json"
    )

    results_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CONFORMAL SUMMARY"
    )
    print("=" * 80)

    for model in conformal_results:

        print()
        print(model)

        for alpha in ALPHAS:

            r = conformal_results[
                model
            ][
                str(alpha)
            ]

            print(
                f"alpha={alpha:.2f} "
                f"target={r['target_coverage']:.2f} "
                f"coverage={r['coverage']:.6f} "
                f"avg_set_size={r['average_set_size']:.6f}"
            )

    print()
    print("=" * 80)
    print(
        "PRIMARY SELECTIVE COMPARISON"
    )
    print("=" * 80)

    print()
    print(
        "VQC predictive-entropy AURC:",
        common_vqc["aurc"],
    )

    print(
        "MC-Dropout predictive-entropy AURC:",
        common_mc["aurc"],
    )

    print()

    print(
        "VQC predictive-entropy "
        "error-detection AUROC:",
        safe_auc(
            (
                (
                    vqc_selective[
                        "probability"
                    ].to_numpy()
                    >= 0.5
                ).astype(int)
                != vqc_selective[
                    "label"
                ].to_numpy()
            ).astype(int),
            vqc_selective[
                "predictive_entropy"
            ].to_numpy(),
        ),
    )

    print(
        "MC-Dropout predictive-entropy "
        "error-detection AUROC:",
        safe_auc(
            (
                (
                    mc_selective[
                        "probability"
                    ].to_numpy()
                    >= 0.5
                ).astype(int)
                != mc_selective[
                    "label"
                ].to_numpy()
            ).astype(int),
            mc_selective[
                "predictive_entropy"
            ].to_numpy(),
        ),
    )

    print()
    print(
        "FINAL CONFORMAL:",
        conformal_file,
    )

    print(
        "FINAL SELECTIVE:",
        selective_file,
    )

    print(
        "FINAL RESULTS:",
        results_file,
    )

    print()
    print(
        "STATUS: PHASE8_9_COMPLETE"
    )


if __name__ == "__main__":
    main()