from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

INPUT = (
    ROOT
    / "experiments"
    / "cbis_reliability_analysis"
    / "PATIENT_LEVEL_PREDICTIONS_MEAN.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_temperature_calibration"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EPS = 1e-7


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def probabilities_to_logits(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )
    return np.log(
        p / (1.0 - p)
    )


def temperature_scale(
    probabilities,
    temperature,
):

    logits = probabilities_to_logits(
        probabilities
    )

    return sigmoid(
        logits / temperature
    )


def fit_temperature(
    y,
    probabilities,
):

    def objective(log_temperature):

        temperature = np.exp(
            log_temperature
        )

        calibrated = temperature_scale(
            probabilities,
            temperature,
        )

        return log_loss(
            y,
            np.clip(
                calibrated,
                EPS,
                1.0 - EPS,
            ),
            labels=[0, 1],
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(0.05),
            np.log(20.0),
        ),
        method="bounded",
        options={
            "xatol": 1e-8,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Temperature optimization failed."
        )

    return float(
        np.exp(
            result.x
        )
    )


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

        predictions = (
            p[mask] >= 0.5
        ).astype(int)

        accuracy = float(
            np.mean(
                predictions
                == y[mask]
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


def summarize(
    y,
    p,
):

    return {
        "n":
            int(len(y)),

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
                        EPS,
                        1.0 - EPS,
                    ),
                    labels=[0, 1],
                )
            ),

        "ece_10bin":
            ece(
                y,
                p,
            ),
    }


def main():

    print("=" * 80)
    print("CBIS-DDSM TEMPERATURE CALIBRATION")
    print("=" * 80)

    df = pd.read_csv(
        INPUT
    )

    required = {
        "split",
        "patient_id",
        "label",
        "model",
        "probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        "Total patient-model rows:",
        len(df),
    )

    calibration = df[
        df["split"] == "calibration"
    ].copy()

    test = df[
        df["split"] == "internal_test"
    ].copy()

    print(
        "Calibration rows:",
        len(calibration),
    )

    print(
        "Internal-test rows:",
        len(test),
    )

    expected_models = {
        "matched_classical_224",
        "vqc_6q_depth2",
    }

    actual_models = set(
        df["model"].unique()
    )

    if actual_models != expected_models:
        raise RuntimeError(
            f"Unexpected models: {actual_models}"
        )

    results = {}
    calibrated_rows = []

    for model in sorted(
        expected_models
    ):

        cal = calibration[
            calibration["model"]
            == model
        ].copy()

        tst = test[
            test["model"]
            == model
        ].copy()

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

        y_test = tst[
            "label"
        ].to_numpy(
            dtype=int
        )

        p_test = tst[
            "probability"
        ].to_numpy(
            dtype=float
        )

        temperature = fit_temperature(
            y_cal,
            p_cal,
        )

        p_calibrated_cal = temperature_scale(
            p_cal,
            temperature,
        )

        p_calibrated_test = temperature_scale(
            p_test,
            temperature,
        )

        cal_before = summarize(
            y_cal,
            p_cal,
        )

        cal_after = summarize(
            y_cal,
            p_calibrated_cal,
        )

        test_before = summarize(
            y_test,
            p_test,
        )

        test_after = summarize(
            y_test,
            p_calibrated_test,
        )

        results[model] = {
            "temperature":
                temperature,

            "calibration_before":
                cal_before,

            "calibration_after":
                cal_after,

            "internal_test_before":
                test_before,

            "internal_test_after":
                test_after,
        }

        tst = tst.copy()

        tst[
            "probability_uncalibrated"
        ] = p_test

        tst[
            "probability_calibrated"
        ] = p_calibrated_test

        cal = cal.copy()

        cal[
            "probability_uncalibrated"
        ] = p_cal

        cal[
            "probability_calibrated"
        ] = p_calibrated_cal

        calibrated_rows.append(
            cal
        )

        calibrated_rows.append(
            tst
        )

    calibrated = pd.concat(
        calibrated_rows,
        ignore_index=True,
    )

    pred_file = (
        OUT
        / "TEMPERATURE_CALIBRATED_PATIENT_PREDICTIONS.csv"
    )

    calibrated.to_csv(
        pred_file,
        index=False,
    )

    results_file = (
        OUT
        / "TEMPERATURE_CALIBRATION_RESULTS.json"
    )

    results_file.write_text(
        json.dumps(
            {
                "experiment":
                    "patient-level post-hoc temperature scaling",

                "calibration_patients":
                    235,

                "internal_test_patients":
                    235,

                "primary_evaluation":
                    "internal_test_after_calibration",

                "models":
                    results,

                "status":
                    "COMPLETE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)

    for model in sorted(
        results
    ):

        r = results[model]

        print()
        print(model)

        print(
            "  Temperature:",
            r["temperature"],
        )

        print(
            "  Test NLL before:",
            r[
                "internal_test_before"
            ][
                "nll"
            ],
        )

        print(
            "  Test NLL after:",
            r[
                "internal_test_after"
            ][
                "nll"
            ],
        )

        print(
            "  Test Brier before:",
            r[
                "internal_test_before"
            ][
                "brier"
            ],
        )

        print(
            "  Test Brier after:",
            r[
                "internal_test_after"
            ][
                "brier"
            ],
        )

        print(
            "  Test ECE before:",
            r[
                "internal_test_before"
            ][
                "ece_10bin"
            ],
        )

        print(
            "  Test ECE after:",
            r[
                "internal_test_after"
            ][
                "ece_10bin"
            ],
        )

    print()
    print(
        "Predictions:",
        pred_file,
    )

    print(
        "Results:",
        results_file,
    )

    print()
    print(
        "STATUS: TEMPERATURE CALIBRATION COMPLETE"
    )


if __name__ == "__main__":
    main()