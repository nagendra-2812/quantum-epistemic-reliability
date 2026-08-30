from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

import pennylane as qml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEED = 2026

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

RUN38 = (
    ROOT
    / "experiments"
    / "STEP38_SELECTIVE_PREDICTION"
)

LATENT_FILE = (
    RUN34B
    / "latent"
    / "SHARED_6D_LATENTS.pt"
)

VQC_CHECKPOINT = (
    RUN34B
    / "checkpoints"
    / "VQC_6Q_DEPTH2_24PARAM_BEST.pt"
)

MLP_CHECKPOINT = (
    RUN34B
    / "checkpoints"
    / "MATCHED_MLP_25PARAM_BEST.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP40_QUANTUM_NOISE_ROBUSTNESS"
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


N_QUBITS = 6
VQC_DEPTH = 2
SHOTS = 1024

NOISE_LEVELS = [
    0.001,
    0.010,
    0.050,
]

NOISE_MODELS = [
    "depolarizing",
    "bit_flip",
    "phase_flip",
]

EPS = 1e-8


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

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


def entropy(p):

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
            (p - y) ** 2
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

    value = 0.0
    n = len(y)

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

        value += (
            count / n
        ) * abs(
            accuracy
            -
            confidence
        )

    return float(
        value
    )


def metrics(
    y,
    p,
):

    pred = (
        p >= 0.5
    ).astype(int)

    try:
        auc = float(
            roc_auc_score(
                y,
                p,
            )
        )
    except Exception:
        auc = float("nan")

    try:
        auprc = float(
            average_precision_score(
                y,
                p,
            )
        )
    except Exception:
        auprc = float("nan")

    return {

        "roc_auc":
            auc,

        "auprc":
            auprc,

        "accuracy":
            float(
                np.mean(
                    pred == y
                )
            ),

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
            ece(
                y,
                p,
                10,
            ),

        "mean_entropy":
            float(
                entropy(
                    p
                ).mean()
            ),

        "mean_probability":
            float(
                p.mean()
            ),
    }


# ============================================================
# INPUT CHECK
# ============================================================

print()
print("=" * 100)
print("STEP 40 - QUANTUM NOISE ROBUSTNESS")
print("=" * 100)

for path in [
    LATENT_FILE,
    VQC_CHECKPOINT,
    MLP_CHECKPOINT,
]:

    if not path.is_file():

        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


# ============================================================
# LOAD FROZEN LATENT
# ============================================================

data = torch.load(
    LATENT_FILE,
    map_location="cpu",
    weights_only=False,
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

X = test_z.numpy()
Y = test_y.numpy().astype(int)

print()
print(
    "Internal-test latent:",
    X.shape,
)

print(
    "Internal-test records:",
    len(Y),
)


# ============================================================
# LOAD VQC PARAMETERS
# ============================================================

state = torch.load(
    VQC_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

if "theta" not in state:

    raise RuntimeError(
        "VQC checkpoint does not contain theta."
    )

theta = (
    state[
        "theta"
    ]
    .detach()
    .cpu()
    .numpy()
)

if theta.shape != (
    VQC_DEPTH,
    N_QUBITS,
    2,
):

    raise RuntimeError(
        f"Unexpected theta shape: {theta.shape}"
    )


print()
print(
    "VQC parameters:",
    int(theta.size),
)


# ============================================================
# FROZEN VQC CIRCUIT
# ============================================================

def predict_vqc(
    inputs,
    noise_model=None,
    noise_level=0.0,
):

    if noise_model is None:

        dev = qml.device(
            "default.qubit",
            wires=N_QUBITS,
            shots=SHOTS,
            seed=SEED,
        )

    else:

        dev = qml.device(
            "default.mixed",
            wires=N_QUBITS,
            shots=SHOTS,
            seed=SEED,
        )


    @qml.qnode(
        dev
    )
    def circuit(
        x
    ):

        # ----------------------------------------------------
        # Input encoding
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qml.RY(
                x[q],
                wires=q,
            )

            qml.RZ(
                x[q],
                wires=q,
            )

            # Noise after encoding gates.
            if (
                noise_model
                is not None
            ):

                if noise_model == "depolarizing":

                    qml.DepolarizingChannel(
                        noise_level,
                        wires=q,
                    )

                elif noise_model == "bit_flip":

                    qml.BitFlip(
                        noise_level,
                        wires=q,
                    )

                elif noise_model == "phase_flip":

                    qml.PhaseFlip(
                        noise_level,
                        wires=q,
                    )

                else:

                    raise ValueError(
                        f"Unknown noise model: {noise_model}"
                    )


        # ----------------------------------------------------
        # Frozen trainable layers
        # ----------------------------------------------------

        for layer in range(
            VQC_DEPTH
        ):

            for q in range(
                N_QUBITS
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

                if (
                    noise_model
                    is not None
                ):

                    if noise_model == "depolarizing":

                        qml.DepolarizingChannel(
                            noise_level,
                            wires=q,
                        )

                    elif noise_model == "bit_flip":

                        qml.BitFlip(
                            noise_level,
                            wires=q,
                        )

                    elif noise_model == "phase_flip":

                        qml.PhaseFlip(
                            noise_level,
                            wires=q,
                        )


            for q in range(
                N_QUBITS
            ):

                qml.CNOT(
                    wires=[
                        q,
                        (
                            q + 1
                        )
                        % N_QUBITS,
                    ]
                )

        return qml.expval(
            qml.PauliZ(0)
        )


    outputs = []

    for i, row in enumerate(
        inputs,
        1,
    ):

        if i % 100 == 0:

            print(
                f"  evaluated {i}/{len(inputs)}",
                flush=True,
            )

        z_value = float(
            circuit(
                row
            )
        )

        p = (
            z_value
            + 1.0
        ) / 2.0

        outputs.append(
            float(
                np.clip(
                    p,
                    EPS,
                    1.0 - EPS,
                )
            )
        )

    return np.asarray(
        outputs,
        dtype=float,
    )


# ============================================================
# NOISE CONDITIONS
# ============================================================

conditions = [
    {
        "noise_model":
            "none",

        "noise_level":
            0.0,
    }
]

for model in NOISE_MODELS:

    for level in NOISE_LEVELS:

        conditions.append(
            {
                "noise_model":
                    model,

                "noise_level":
                    level,
            }
        )


# ============================================================
# RUN
# ============================================================

result_rows = []
source_rows = []

reference_probability = None


for condition in conditions:

    noise_model = condition[
        "noise_model"
    ]

    noise_level = condition[
        "noise_level"
    ]

    print()
    print(
        "-" * 90
    )

    print(
        "Noise model:",
        noise_model,
    )

    print(
        "Noise level:",
        noise_level,
    )

    p = predict_vqc(
        X,
        None
        if noise_model == "none"
        else noise_model,
        noise_level,
    )

    row_metrics = metrics(
        Y,
        p,
    )

    if noise_model == "none":

        reference_probability = (
            p.copy()
        )

        reference_metrics = (
            row_metrics.copy()
        )

    auc_delta = (
        row_metrics[
            "roc_auc"
        ]
        -
        reference_metrics[
            "roc_auc"
        ]
    )

    auprc_delta = (
        row_metrics[
            "auprc"
        ]
        -
        reference_metrics[
            "auprc"
        ]
    )

    brier_delta = (
        row_metrics[
            "brier"
        ]
        -
        reference_metrics[
            "brier"
        ]
    )

    nll_delta = (
        row_metrics[
            "nll"
        ]
        -
        reference_metrics[
            "nll"
        ]
    )

    ece_delta = (
        row_metrics[
            "ece_10bin"
        ]
        -
        reference_metrics[
            "ece_10bin"
        ]
    )

    result_rows.append({

        "model":
            "VQC",

        "noise_model":
            noise_model,

        "noise_level":
            float(
                noise_level
            ),

        **row_metrics,

        "roc_auc_delta_vs_noiseless":
            float(
                auc_delta
            ),

        "auprc_delta_vs_noiseless":
            float(
                auprc_delta
            ),

        "brier_delta_vs_noiseless":
            float(
                brier_delta
            ),

        "nll_delta_vs_noiseless":
            float(
                nll_delta
            ),

        "ece_delta_vs_noiseless":
            float(
                ece_delta
            ),
    })


    for i in range(
        len(Y)
    ):

        source_rows.append({

            "record_index":
                i,

            "label":
                int(
                    Y[i]
                ),

            "noise_model":
                noise_model,

            "noise_level":
                float(
                    noise_level
                ),

            "probability":
                float(
                    p[i]
                ),

            "entropy":
                float(
                    entropy(
                        [
                            p[i]
                        ]
                    )[0]
                ),
        })


results_df = pd.DataFrame(
    result_rows
)

source_df = pd.DataFrame(
    source_rows
)


# ============================================================
# SAVE TABLES
# ============================================================

results_df.to_csv(
    TABLE_DIR
    / "TABLE_40_QUANTUM_NOISE_ROBUSTNESS.csv",
    index=False,
)

source_df.to_csv(
    SOURCE_DIR
    / "QUANTUM_NOISE_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# FIGURE 1 - AUC DEGRADATION
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for noise_model in NOISE_MODELS:

    subset = results_df[
        results_df[
            "noise_model"
        ]
        ==
        noise_model
    ]

    plt.plot(
        subset[
            "noise_level"
        ],
        subset[
            "roc_auc"
        ],
        marker="o",
        linewidth=1.8,
        label=noise_model,
    )

plt.axhline(
    reference_metrics[
        "roc_auc"
    ],
    linestyle="--",
    linewidth=1.0,
    label="noiseless reference",
)

plt.xlabel(
    "Noise probability"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "VQC robustness to quantum noise"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ROBUSTNESS.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ROBUSTNESS.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ROBUSTNESS_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 - ENTROPY
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for noise_model in NOISE_MODELS:

    subset = results_df[
        results_df[
            "noise_model"
        ]
        ==
        noise_model
    ]

    plt.plot(
        subset[
            "noise_level"
        ],
        subset[
            "mean_entropy"
        ],
        marker="o",
        linewidth=1.8,
        label=noise_model,
    )

plt.axhline(
    reference_metrics[
        "mean_entropy"
    ],
    linestyle="--",
    linewidth=1.0,
    label="noiseless reference",
)

plt.xlabel(
    "Noise probability"
)

plt.ylabel(
    "Mean predictive entropy"
)

plt.title(
    "VQC uncertainty under quantum noise"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ENTROPY.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ENTROPY.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_ENTROPY_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3 - BRIER / NLL
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for noise_model in NOISE_MODELS:

    subset = results_df[
        results_df[
            "noise_model"
        ]
        ==
        noise_model
    ]

    plt.plot(
        subset[
            "noise_level"
        ],
        subset[
            "brier"
        ],
        marker="o",
        linewidth=1.8,
        label=noise_model,
    )

plt.xlabel(
    "Noise probability"
)

plt.ylabel(
    "Brier score"
)

plt.title(
    "VQC probabilistic degradation under quantum noise"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_BRIER.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_BRIER.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_40_VQC_NOISE_BRIER_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP40_COMPLETE",

    "dataset":
        "CBIS-DDSM",

    "test_records":
        int(
            len(Y)
        ),

    "vqc_qubits":
        N_QUBITS,

    "vqc_depth":
        VQC_DEPTH,

    "vqc_parameters":
        int(
            theta.size
        ),

    "shots":
        SHOTS,

    "noise_models":
        NOISE_MODELS,

    "noise_levels":
        NOISE_LEVELS,

    "reference":
        reference_metrics,

    "results":
        result_rows,

    "interpretation":
        (
            "Inference-only quantum-noise robustness test "
            "using the frozen 34B VQC parameters. No model "
            "retraining or parameter adaptation was performed."
        ),

    "artifacts":
        {
            "table":
                str(
                    TABLE_DIR
                    / "TABLE_40_QUANTUM_NOISE_ROBUSTNESS.csv"
                ),

            "source":
                str(
                    SOURCE_DIR
                    / "QUANTUM_NOISE_SOURCE_DATA.csv"
                ),

            "figures":
                str(
                    FIG_DIR
                ),
        },
}


(
    METRIC_DIR
    / "STEP40_FINAL_RESULTS.json"
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

    "shots":
        SHOTS,

    "vqc_qubits":
        N_QUBITS,

    "vqc_depth":
        VQC_DEPTH,

    "vqc_parameters":
        int(
            theta.size
        ),

    "noise_models":
        NOISE_MODELS,

    "noise_levels":
        NOISE_LEVELS,

    "inference_only":
        True,

    "retraining":
        False,

    "source_latent":
        str(
            LATENT_FILE
        ),

    "source_vqc_checkpoint":
        str(
            VQC_CHECKPOINT
        ),
}


(
    CONFIG_DIR
    / "STEP40_CONFIGURATION.json"
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
    "STEP 40 COMPLETE"
)
print("=" * 100)

print()
print(
    results_df[
        [
            "noise_model",
            "noise_level",
            "roc_auc",
            "auprc",
            "brier",
            "nll",
            "ece_10bin",
            "mean_entropy",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Results:",
    METRIC_DIR
    / "STEP40_FINAL_RESULTS.json",
)

print(
    "Table:",
    TABLE_DIR
    / "TABLE_40_QUANTUM_NOISE_ROBUSTNESS.csv",
)

print(
    "Source:",
    SOURCE_DIR
    / "QUANTUM_NOISE_SOURCE_DATA.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print()
print(
    "STATUS: STEP40_COMPLETE"
)