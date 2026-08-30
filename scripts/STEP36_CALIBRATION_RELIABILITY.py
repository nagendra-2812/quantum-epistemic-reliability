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

from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

import pennylane as qml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 2026
EPS = 1e-8

ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

RUN34B = (
    ROOT
    / "experiments"
    / "STEP34B_FINAL_MATCHED_MLP_VQC"
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
    / "STEP36_CALIBRATION"
)

TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_data"
FIG_DIR = OUT / "figures"
METRIC_DIR = OUT / "metrics"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    TABLE_DIR,
    SOURCE_DIR,
    FIG_DIR,
    METRIC_DIR,
    CONFIG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda:
                f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

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


def brier(y, p):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return float(
        np.mean(
            (
                p - y
            ) ** 2
        )
    )


def nll(y, p):

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

    rows = []
    total = len(y)
    value = 0.0

    for i in range(
        bins
    ):

        low = edges[i]
        high = edges[i + 1]

        if i == bins - 1:

            mask = (
                (p >= low)
                &
                (p <= high)
            )

        else:

            mask = (
                (p >= low)
                &
                (p < high)
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

        gap = abs(
            accuracy
            -
            confidence
        )

        value += (
            count / total
        ) * gap

        rows.append(
            {
                "bin":
                    i + 1,

                "lower":
                    low,

                "upper":
                    high,

                "count":
                    count,

                "accuracy":
                    accuracy,

                "confidence":
                    confidence,

                "absolute_gap":
                    gap,
            }
        )

    return (
        float(value),
        pd.DataFrame(rows),
    )


def calibration_summary(
    y,
    p,
):

    ece_value, bins = ece(
        y,
        p,
        10,
    )

    try:

        auc = float(
            roc_auc_score(
                y,
                p,
            )
        )

    except Exception:

        auc = float("nan")

    return (
        {
            "n":
                int(
                    len(y)
                ),

            "roc_auc":
                auc,

            "brier":
                brier(
                    y,
                    p,
                ),

            "nll":
                nll(
                    y,
                    p,
                ),

            "ece_10bin":
                ece_value,
        },
        bins,
    )


def fit_temperature(
    y,
    p,
):

    logits = logit(
        p
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    def objective(
        log_temperature
    ):

        temperature = np.exp(
            log_temperature
        )

        calibrated = sigmoid(
            logits
            /
            temperature
        )

        return nll(
            y,
            calibrated,
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(0.05),
            np.log(20.0),
        ),
        method="bounded",
    )

    temperature = float(
        np.exp(
            result.x
        )
    )

    return temperature


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 90)
print("STEP 36 - CALIBRATION / RELIABILITY ANALYSIS")
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


# ============================================================
# LOAD SHARED LATENT
# ============================================================

data = torch.load(
    LATENT_FILE,
    map_location="cpu",
    weights_only=False,
)

train_z = (
    data[
        "train_z"
    ]
    .float()
    .cpu()
)

train_y = (
    data[
        "train_y"
    ]
    .float()
    .cpu()
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

print()
print(
    "Train latent:",
    tuple(
        train_z.shape
    ),
)

print(
    "Calibration latent:",
    tuple(
        cal_z.shape
    ),
)

print(
    "Internal-test latent:",
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

    cal_mlp = (
        torch.sigmoid(
            mlp(
                cal_z
            )
        )
        .numpy()
    )

    test_mlp = (
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

if theta.shape != (
    2,
    6,
    2,
):

    raise RuntimeError(
        f"Unexpected VQC parameter shape: {theta.shape}"
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
    "Generating calibration/test predictions..."
)

cal_vqc = vqc_predict(
    cal_z
)

test_vqc = vqc_predict(
    test_z
)


# ============================================================
# CALIBRATE
# ============================================================

model_data = {

    "MLP":
        {
            "cal_y":
                cal_y.numpy().astype(int),

            "cal_p":
                cal_mlp,

            "test_y":
                test_y.numpy().astype(int),

            "test_p":
                test_mlp,
        },

    "VQC":
        {
            "cal_y":
                cal_y.numpy().astype(int),

            "cal_p":
                cal_vqc,

            "test_y":
                test_y.numpy().astype(int),

            "test_p":
                test_vqc,
        },
}


results = []
bin_sources = []


for name, item in model_data.items():

    cal_y_np = item[
        "cal_y"
    ]

    cal_p_np = item[
        "cal_p"
    ]

    test_y_np = item[
        "test_y"
    ]

    test_p_np = item[
        "test_p"
    ]

    # --------------------------------------------------------
    # Before
    # --------------------------------------------------------

    cal_before, _ = calibration_summary(
        cal_y_np,
        cal_p_np,
    )

    test_before, test_bins_before = calibration_summary(
        test_y_np,
        test_p_np,
    )

    # --------------------------------------------------------
    # Temperature fitted ONLY on calibration
    # --------------------------------------------------------

    temperature = fit_temperature(
        cal_y_np,
        cal_p_np,
    )

    cal_after_p = sigmoid(
        logit(
            cal_p_np
        )
        /
        temperature
    )

    test_after_p = sigmoid(
        logit(
            test_p_np
        )
        /
        temperature
    )

    # --------------------------------------------------------
    # After
    # --------------------------------------------------------

    cal_after, _ = calibration_summary(
        cal_y_np,
        cal_after_p,
    )

    test_after, test_bins_after = calibration_summary(
        test_y_np,
        test_after_p,
    )

    results.append(
        {

            "model":
                name,

            "temperature":
                temperature,

            "calibration_ece_before":
                cal_before[
                    "ece_10bin"
                ],

            "calibration_ece_after":
                cal_after[
                    "ece_10bin"
                ],

            "calibration_brier_before":
                cal_before[
                    "brier"
                ],

            "calibration_brier_after":
                cal_after[
                    "brier"
                ],

            "calibration_nll_before":
                cal_before[
                    "nll"
                ],

            "calibration_nll_after":
                cal_after[
                    "nll"
                ],

            "test_ece_before":
                test_before[
                    "ece_10bin"
                ],

            "test_ece_after":
                test_after[
                    "ece_10bin"
                ],

            "test_brier_before":
                test_before[
                    "brier"
                ],

            "test_brier_after":
                test_after[
                    "brier"
                ],

            "test_nll_before":
                test_before[
                    "nll"
                ],

            "test_nll_after":
                test_after[
                    "nll"
                ],

            "test_roc_auc_before":
                test_before[
                    "roc_auc"
                ],

            "test_roc_auc_after":
                test_after[
                    "roc_auc"
                ],
        }
    )

    # --------------------------------------------------------
    # Source data
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "label":
                test_y_np,

            "probability_before":
                test_p_np,

            "probability_after":
                test_after_p,
        }
    ).to_csv(
        SOURCE_DIR
        /
        f"{name}_TEST_CALIBRATION_SOURCE.csv",
        index=False,
    )

    for stage, bins in [
        (
            "before",
            test_bins_before,
        ),
        (
            "after",
            test_bins_after,
        ),
    ]:

        if len(bins):

            tmp = bins.copy()

            tmp[
                "model"
            ] = name

            tmp[
                "stage"
            ] = stage

            bin_sources.append(
                tmp
            )


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    TABLE_DIR
    / "TABLE_36_CALIBRATION_RESULTS.csv",
    index=False,
)


bins_df = (
    pd.concat(
        bin_sources,
        ignore_index=True,
    )
    if bin_sources
    else
    pd.DataFrame()
)

bins_df.to_csv(
    SOURCE_DIR
    / "RELIABILITY_DIAGRAM_SOURCE.csv",
    index=False,
)


# ============================================================
# RELIABILITY FIGURE
# ============================================================

fig = plt.figure(
    figsize=(6.5, 5.2)
)

for name, stage, label in [

    (
        "MLP",
        "before",
        "MLP before",
    ),

    (
        "MLP",
        "after",
        "MLP after",
    ),

    (
        "VQC",
        "before",
        "VQC before",
    ),

    (
        "VQC",
        "after",
        "VQC after",
    ),
]:

    if len(bins_df) == 0:
        continue

    subset = bins_df[
        (
            bins_df[
                "model"
            ]
            == name
        )
        &
        (
            bins_df[
                "stage"
            ]
            == stage
        )
    ]

    if len(subset) == 0:
        continue

    plt.plot(
        subset[
            "confidence"
        ],
        subset[
            "accuracy"
        ],
        marker="o",
        linewidth=1.8,
        label=label,
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel(
    "Mean predicted confidence"
)

plt.ylabel(
    "Observed accuracy"
)

plt.title(
    "CBIS-DDSM reliability diagram"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_36_RELIABILITY_DIAGRAM.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_36_RELIABILITY_DIAGRAM.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_36_RELIABILITY_DIAGRAM_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP36_COMPLETE",

    "method":
        "temperature_scaling",

    "fit_split":
        "calibration",

    "evaluation_split":
        "internal_test",

    "test_leakage":
        False,

    "results":
        results,

    "source_latent":
        str(LATENT_FILE),

    "source_mlp_checkpoint":
        str(MLP_CHECKPOINT),

    "source_vqc_checkpoint":
        str(VQC_CHECKPOINT),

    "note":
        (
            "Calibration parameters were fitted exclusively "
            "on the frozen calibration split and applied "
            "unchanged to the internal test split."
        ),
}


(
    METRIC_DIR
    / "STEP36_FINAL_RESULTS.json"
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

    "method":
        "temperature_scaling",

    "ece_bins":
        10,

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

    "source_latent":
        str(LATENT_FILE),

    "source_mlp_checkpoint":
        str(MLP_CHECKPOINT),

    "source_vqc_checkpoint":
        str(VQC_CHECKPOINT),
}


(
    CONFIG_DIR
    / "STEP36_CONFIGURATION.json"
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
# SHA256 INVENTORY
# ============================================================

hash_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
        continue

    hash_rows.append(
        {

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
        }
    )


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
print("STEP 36 COMPLETE")
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
    / "STEP36_FINAL_RESULTS.json",
)

print(
    "Table:",
    TABLE_DIR
    / "TABLE_36_CALIBRATION_RESULTS.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print()
print(
    "STATUS: STEP36_COMPLETE"
)