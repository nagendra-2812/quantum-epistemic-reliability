from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pennylane as qml

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)


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

RUN36 = (
    ROOT
    / "experiments"
    / "STEP36_CALIBRATION"
)

LATENT_FILE = (
    RUN34B
    / "latent"
    / "SHARED_6D_LATENTS.pt"
)

MLP_CHECKPOINT = (
    RUN34B
    / "checkpoints"
    / "MATCHED_MLP_25PARAM_BEST.pt"
)

VQC_CHECKPOINT = (
    RUN34B
    / "checkpoints"
    / "VQC_6Q_DEPTH2_24PARAM_BEST.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP37_CONFORMAL"
)

TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_data"
METRIC_DIR = OUT / "metrics"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    TABLE_DIR,
    SOURCE_DIR,
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


def sigmoid(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return 1.0 / (
        1.0
        +
        np.exp(
            -np.clip(
                x,
                -50,
                50,
            )
        )
    )


def logit(p):

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p
        /
        (
            1.0 - p
        )
    )


def binary_entropy(p):

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
        * np.log(
            1.0 - p
        )
    )


# ------------------------------------------------------------
# Split-conformal binary classification
#
# Nonconformity:
#   score = 1 - p_true
#
# Finite-sample quantile:
#   k = ceil((n + 1)*(1-alpha))
#   q = k-th order statistic with finite-sample clipping
#
# Prediction set:
#   include class c when 1-p(c) <= q
# ------------------------------------------------------------

def conformal_threshold(
    y,
    p,
    alpha,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    p_true = np.where(
        y == 1,
        p,
        1.0 - p,
    )

    scores = 1.0 - p_true

    n = len(
        scores
    )

    k = int(
        np.ceil(
            (
                n + 1
            )
            *
            (
                1.0 - alpha
            )
        )
    )

    k = min(
        max(
            k,
            1,
        ),
        n,
    )

    sorted_scores = np.sort(
        scores
    )

    q = float(
        sorted_scores[
            k - 1
        ]
    )

    return (
        q,
        scores,
    )


def evaluate_conformal(
    y,
    p,
    q,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    p0 = 1.0 - p
    p1 = p

    include0 = (
        1.0 - p0
    ) <= (
        q + 1e-12
    )

    include1 = (
        1.0 - p1
    ) <= (
        q + 1e-12
    )

    # Actual coverage.
    covered = np.where(
        y == 0,
        include0,
        include1,
    )

    # Set size.
    set_size = (
        include0.astype(int)
        +
        include1.astype(int)
    )

    # Point prediction only when singleton.
    singleton = (
        set_size
        == 1
    )

    point_prediction = np.full(
        len(y),
        -1,
        dtype=int,
    )

    point_prediction[
        include0 & ~include1
    ] = 0

    point_prediction[
        include1 & ~include0
    ] = 1

    singleton_accuracy = (
        float(
            accuracy_score(
                y[singleton],
                point_prediction[
                    singleton
                ],
            )
        )
        if singleton.any()
        else float("nan")
    )

    return {

        "coverage":
            float(
                np.mean(
                    covered
                )
            ),

        "mean_set_size":
            float(
                np.mean(
                    set_size
                )
            ),

        "singleton_rate":
            float(
                np.mean(
                    singleton
                )
            ),

        "singleton_accuracy":
            singleton_accuracy,

        "empty_set_rate":
            float(
                np.mean(
                    set_size == 0
                )
            ),

        "doubleton_rate":
            float(
                np.mean(
                    set_size == 2
                )
            ),

    }, {

        "include_benign":
            include0,

        "include_malignant":
            include1,

        "covered":
            covered,

        "set_size":
            set_size,

        "point_prediction":
            point_prediction,

    }


# ============================================================
# LOAD LATENT
# ============================================================

print()
print("=" * 90)
print("STEP 37 - CONFORMAL PREDICTION")
print("=" * 90)

for path in [
    LATENT_FILE,
    MLP_CHECKPOINT,
    VQC_CHECKPOINT,
]:

    if not path.is_file():

        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


data = torch.load(
    LATENT_FILE,
    map_location="cpu",
    weights_only=False,
)

cal_z = (
    data[
        "calibration_z"
    ]
    .float()
    .cpu()
)

cal_y = (
    data[
        "calibration_y"
    ]
    .float()
    .cpu()
)

test_z = (
    data[
        "internal_test_z"
    ]
    .float()
    .cpu()
)

test_y = (
    data[
        "internal_test_y"
    ]
    .float()
    .cpu()
)

Y_CAL = (
    cal_y.numpy()
    .astype(int)
)

Y_TEST = (
    test_y.numpy()
    .astype(int)
)


print()
print(
    "Calibration:",
    tuple(
        cal_z.shape
    ),
)

print(
    "Internal test:",
    tuple(
        test_z.shape
    ),
)


# ============================================================
# MLP
# ============================================================

class MatchedMLP(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                6,
                3,
            ),

            nn.Tanh(),

            nn.Linear(
                3,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        ).squeeze(
            -1
        )


mlp = MatchedMLP()

mlp.load_state_dict(
    torch.load(
        MLP_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
)

mlp.eval()

with torch.no_grad():

    p_cal_mlp = (
        torch.sigmoid(
            mlp(
                cal_z
            )
        )
        .numpy()
    )

    p_test_mlp = (
        torch.sigmoid(
            mlp(
                test_z
            )
        )
        .numpy()
    )


# ============================================================
# VQC
# ============================================================

vqc_state = torch.load(
    VQC_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

theta = (
    vqc_state[
        "theta"
    ]
    .detach()
    .cpu()
    .numpy()
)


dev = qml.device(
    "lightning.qubit",
    wires=6,
)


@qml.qnode(
    dev
)
def circuit(
    x
):

    for q in range(
        6
    ):

        qml.RY(
            x[q],
            wires=q,
        )

        qml.RZ(
            x[q],
            wires=q,
        )

    for layer in range(
        2
    ):

        for q in range(
            6
        ):

            qml.RY(
                theta[
                    layer,
                    q,
                    0,
                ],
                wires=q,
            )

            qml.RZ(
                theta[
                    layer,
                    q,
                    1,
                ],
                wires=q,
            )

        for q in range(
            6
        ):

            qml.CNOT(
                wires=[
                    q,
                    (
                        q + 1
                    ) % 6,
                ]
            )

    return qml.expval(
        qml.PauliZ(0)
    )


def vqc_predict(
    z
):

    values = []

    for row in z.numpy():

        expectation = float(
            circuit(
                row
            )
        )

        probability = (
            expectation
            + 1.0
        ) / 2.0

        values.append(
            float(
                np.clip(
                    probability,
                    EPS,
                    1.0 - EPS,
                )
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


print()
print(
    "Generating frozen VQC calibration/test probabilities..."
)

p_cal_vqc = vqc_predict(
    cal_z
)

p_test_vqc = vqc_predict(
    test_z
)


# ============================================================
# CONFORMAL
# ============================================================

models = {

    "MLP":
        (
            Y_CAL,
            p_cal_mlp,
            Y_TEST,
            p_test_mlp,
        ),

    "VQC":
        (
            Y_CAL,
            p_cal_vqc,
            Y_TEST,
            p_test_vqc,
        ),
}


alphas = [
    0.10,
    0.05,
]

result_rows = []
source_rows = []


for model_name, (
    y_cal,
    p_cal,
    y_test,
    p_test,
) in models.items():

    print()
    print(
        "-" * 80
    )

    print(
        model_name
    )

    for alpha in alphas:

        q, cal_scores = (
            conformal_threshold(
                y_cal,
                p_cal,
                alpha,
            )
        )

        test_summary, test_sets = (
            evaluate_conformal(
                y_test,
                p_test,
                q,
            )
        )

        cal_summary, _ = (
            evaluate_conformal(
                y_cal,
                p_cal,
                q,
            )
        )

        result_rows.append({

            "model":
                model_name,

            "alpha":
                alpha,

            "target_coverage":
                1.0 - alpha,

            "quantile_threshold":
                q,

            "calibration_coverage":
                cal_summary[
                    "coverage"
                ],

            "test_coverage":
                test_summary[
                    "coverage"
                ],

            "mean_test_set_size":
                test_summary[
                    "mean_set_size"
                ],

            "singleton_rate":
                test_summary[
                    "singleton_rate"
                ],

            "singleton_accuracy":
                test_summary[
                    "singleton_accuracy"
                ],

            "empty_set_rate":
                test_summary[
                    "empty_set_rate"
                ],

            "doubleton_rate":
                test_summary[
                    "doubleton_rate"
                ],

        })

        print()
        print(
            f"alpha={alpha}"
        )

        print(
            "  threshold:",
            q
        )

        print(
            "  calibration coverage:",
            cal_summary[
                "coverage"
            ]
        )

        print(
            "  test coverage:",
            test_summary[
                "coverage"
            ]
        )

        print(
            "  mean test set size:",
            test_summary[
                "mean_set_size"
            ]
        )

        print(
            "  singleton rate:",
            test_summary[
                "singleton_rate"
            ]
        )


        # ----------------------------------------------------
        # Source records
        # ----------------------------------------------------

        for i in range(
            len(y_test)
        ):

            source_rows.append({

                "model":
                    model_name,

                "alpha":
                    alpha,

                "record_index":
                    i,

                "true_label":
                    int(
                        y_test[i]
                    ),

                "probability_malignant":
                    float(
                        p_test[i]
                    ),

                "probability_benign":
                    float(
                        1.0
                        -
                        p_test[i]
                    ),

                "include_benign":
                    bool(
                        test_sets[
                            "include_benign"
                        ][i]
                    ),

                "include_malignant":
                    bool(
                        test_sets[
                            "include_malignant"
                        ][i]
                    ),

                "covered":
                    bool(
                        test_sets[
                            "covered"
                        ][i]
                    ),

                "set_size":
                    int(
                        test_sets[
                            "set_size"
                        ][i]
                    ),

                "point_prediction":
                    int(
                        test_sets[
                            "point_prediction"
                        ][i]
                    ),

                "conformity_threshold":
                    float(
                        q
                    ),
            })


results_df = pd.DataFrame(
    result_rows
)

source_df = pd.DataFrame(
    source_rows
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    TABLE_DIR
    / "TABLE_37_CONFORMAL_RESULTS.csv",
    index=False,
)

source_df.to_csv(
    SOURCE_DIR
    / "CONFORMAL_PREDICTION_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP37_COMPLETE",

    "method":
        "split_conformal_binary_classification",

    "calibration_split":
        "calibration",

    "evaluation_split":
        "internal_test",

    "alphas":
        alphas,

    "models":
        result_rows,

    "nonconformity_score":
        "1 - probability_of_true_class",

    "temperature_scaling_not_used":
        True,

    "test_leakage":
        False,

    "note":
        (
            "Conformal thresholds were fitted exclusively "
            "on the frozen calibration split and evaluated "
            "on the internal test split."
        ),
}


(
    METRIC_DIR
    / "STEP37_FINAL_RESULTS.json"
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

    "method":
        "split_conformal_binary_classification",

    "alphas":
        alphas,

    "nonconformity":
        "1-p_true",

    "fit_split":
        "calibration",

    "evaluation_split":
        "internal_test",

    "mlp_parameters":
        25,

    "vqc_parameters":
        24,

    "vqc_qubits":
        6,

    "vqc_depth":
        2,

    "test_leakage":
        False,

    "seed":
        SEED,
}


(
    CONFIG_DIR
    / "STEP37_CONFIGURATION.json"
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

    "torch":
        torch.__version__,

    "pennylane":
        qml.__version__,
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
print("=" * 90)
print("STEP 37 COMPLETE")
print("=" * 90)

print()
print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    "Results:",
    METRIC_DIR
    / "STEP37_FINAL_RESULTS.json",
)

print(
    "Table:",
    TABLE_DIR
    / "TABLE_37_CONFORMAL_RESULTS.csv",
)

print(
    "Source:",
    SOURCE_DIR
    / "CONFORMAL_PREDICTION_SOURCE_DATA.csv",
)

print()
print(
    "STATUS: STEP37_COMPLETE"
)