from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    / "cbis_conformal_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ALPHAS = (
    0.10,
    0.05,
)


def split_conformity_score(
    y,
    p,
):
    """
    Binary conformal score:
        s = 1 - probability assigned to the true class.
    """

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
    """
    Finite-sample split-conformal quantile.

    k = ceil((n + 1) * (1-alpha))
    q = k-th smallest score, capped at n.
    """

    scores = np.sort(
        np.asarray(
            scores,
            dtype=float,
        )
    )

    n = len(scores)

    k = int(
        np.ceil(
            (n + 1)
            * (1.0 - alpha)
        )
    )

    k = min(
        max(k, 1),
        n,
    )

    return float(
        scores[k - 1]
    )


def prediction_set(
    p,
    q,
):
    """
    Return binary conformal prediction sets.

    Class 0 included when 1-p >= 1-q.
    Class 1 included when p >= 1-q.
    """

    sets = []

    for probability in p:

        include_0 = (
            1.0 - probability
            >= 1.0 - q
        )

        include_1 = (
            probability
            >= 1.0 - q
        )

        if include_0 and include_1:
            label = "0,1"
        elif include_1:
            label = "1"
        elif include_0:
            label = "0"
        else:
            label = "EMPTY"

        sets.append(
            label
        )

    return sets


def evaluate_sets(
    y,
    p,
    q,
):

    sets = prediction_set(
        p,
        q,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    sizes = []

    covered = []

    singleton = []

    ambiguous = []

    empty = []

    for truth, label_set in zip(
        y,
        sets,
    ):

        if label_set == "0":
            size = 1
            contains_truth = (
                truth == 0
            )
        elif label_set == "1":
            size = 1
            contains_truth = (
                truth == 1
            )
        elif label_set == "0,1":
            size = 2
            contains_truth = True
        else:
            size = 0
            contains_truth = False

        sizes.append(
            size
        )

        covered.append(
            contains_truth
        )

        singleton.append(
            size == 1
        )

        ambiguous.append(
            size == 2
        )

        empty.append(
            size == 0
        )

    return {
        "n": int(len(y)),
        "coverage": float(
            np.mean(covered)
        ),
        "average_set_size": float(
            np.mean(sizes)
        ),
        "singleton_rate": float(
            np.mean(singleton)
        ),
        "ambiguous_rate": float(
            np.mean(ambiguous)
        ),
        "empty_rate": float(
            np.mean(empty)
        ),
        "sets": sets,
    }


def main():

    print("=" * 80)
    print(
        "CBIS-DDSM SPLIT-CONFORMAL ANALYSIS"
    )
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
            f"Missing columns: {sorted(missing)}"
        )

    calibration = df[
        df["split"]
        == "calibration"
    ].copy()

    test = df[
        df["split"]
        == "internal_test"
    ].copy()

    print(
        "Calibration patient-model rows:",
        len(calibration),
    )

    print(
        "Internal-test patient-model rows:",
        len(test),
    )

    models = sorted(
        df["model"].unique()
    )

    expected = {
        "matched_classical_224",
        "vqc_6q_depth2",
    }

    if set(models) != expected:
        raise RuntimeError(
            f"Unexpected models: {models}"
        )

    results = {}

    output_rows = []

    for model in models:

        cal = calibration[
            calibration["model"]
            == model
        ].copy()

        tst = test[
            test["model"]
            == model
        ].copy()

        if len(cal) != 235:
            raise RuntimeError(
                f"{model}: expected 235 calibration patients, "
                f"found {len(cal)}"
            )

        if len(tst) != 235:
            raise RuntimeError(
                f"{model}: expected 235 internal-test patients, "
                f"found {len(tst)}"
            )

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

        scores = split_conformity_score(
            y_cal,
            p_cal,
        )

        results[model] = {
            "calibration": {
                "n":
                    int(len(scores)),
                "mean_score":
                    float(scores.mean()),
                "median_score":
                    float(np.median(scores)),
            },
            "internal_test": {},
        }

        for alpha in ALPHAS:

            q = conformal_quantile(
                scores,
                alpha,
            )

            evaluation = evaluate_sets(
                y_test,
                p_test,
                q,
            )

            # Remove full set list from summary JSON;
            # save it separately below.
            summary = {
                key: value
                for key, value
                in evaluation.items()
                if key != "sets"
            }

            summary[
                "alpha"
            ] = alpha

            summary[
                "target_coverage"
            ] = 1.0 - alpha

            summary[
                "quantile"
            ] = q

            results[model][
                "internal_test"
            ][
                str(alpha)
            ] = summary

            for pid, truth, prob, label_set in zip(
                tst["patient_id"],
                y_test,
                p_test,
                evaluation["sets"],
            ):

                output_rows.append({
                    "model":
                        model,
                    "alpha":
                        alpha,
                    "patient_id":
                        pid,
                    "label":
                        int(truth),
                    "probability":
                        float(prob),
                    "conformal_quantile":
                        float(q),
                    "prediction_set":
                        label_set,
                    "set_size":
                        0
                        if label_set == "EMPTY"
                        else len(
                            label_set.split(",")
                        ),
                    "covered":
                        (
                            (
                                label_set == "0"
                                and truth == 0
                            )
                            or
                            (
                                label_set == "1"
                                and truth == 1
                            )
                            or
                            label_set == "0,1"
                        ),
                })

    output_file = (
        OUT
        / "CONFORMAL_PATIENT_PREDICTIONS.csv"
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        output_file,
        index=False,
    )

    results_file = (
        OUT
        / "CONFORMAL_RESULTS.json"
    )

    results_file.write_text(
        json.dumps(
            {
                "experiment":
                    "patient-level split conformal prediction",

                "calibration_patients":
                    235,

                "internal_test_patients":
                    235,

                "alphas":
                    list(ALPHAS),

                "results":
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
    print(
        "CONFORMAL PREDICTION SUMMARY"
    )
    print("=" * 80)

    for model in models:

        print()
        print(model)

        for alpha in ALPHAS:

            r = results[
                model
            ][
                "internal_test"
            ][
                str(alpha)
            ]

            print(
                f"  alpha={alpha:.2f}"
            )

            print(
                "    target coverage:",
                r[
                    "target_coverage"
                ],
            )

            print(
                "    conformal quantile:",
                r[
                    "quantile"
                ],
            )

            print(
                "    observed coverage:",
                r[
                    "coverage"
                ],
            )

            print(
                "    average set size:",
                r[
                    "average_set_size"
                ],
            )

            print(
                "    singleton rate:",
                r[
                    "singleton_rate"
                ],
            )

            print(
                "    ambiguous rate:",
                r[
                    "ambiguous_rate"
                ],
            )

            print(
                "    empty rate:",
                r[
                    "empty_rate"
                ],
            )

    print()
    print(
        "Prediction sets:",
        output_file,
    )

    print(
        "Results:",
        results_file,
    )

    print()
    print(
        "STATUS: CONFORMAL ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()