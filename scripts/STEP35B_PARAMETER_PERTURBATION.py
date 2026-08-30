from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

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
    / "STEP35B_PARAMETER_PERTURBATION"
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

# Relative Gaussian perturbation scales.
SIGMAS = [
    0.01,
    0.05,
    0.10,
]

REPEATS = 20

EPS = 1e-8


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def binary_entropy(p):

    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return (
        -p * np.log(p)
        -
        (1.0 - p)
        * np.log(1.0 - p)
    )


def error_detection_auroc(
    y,
    probability,
    uncertainty,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    errors = (
        prediction != y
    ).astype(int)

    if len(
        np.unique(errors)
    ) < 2:

        return float("nan")

    return float(
        roc_auc_score(
            errors,
            uncertainty,
        )
    )


def risk_coverage(
    y,
    probability,
    uncertainty,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    errors = (
        prediction != y
    ).astype(float)

    order = np.argsort(
        uncertainty
    )

    errors = errors[
        order
    ]

    risks = []
    coverage = []

    cumulative = 0.0
    n = len(errors)

    for k in range(
        1,
        n + 1,
    ):

        cumulative += errors[
            k - 1
        ]

        coverage.append(
            k / n
        )

        risks.append(
            cumulative / k
        )

    coverage = np.asarray(
        coverage,
        dtype=float,
    )

    risks = np.asarray(
        risks,
        dtype=float,
    )

    aurc = float(
        np.trapezoid(
            risks,
            coverage,
        )
    )

    return (
        coverage,
        risks,
        aurc,
    )


def patient_aggregate(
    probability,
    labels,
    patient_ids,
):

    df = pd.DataFrame({

        "patient_id":
            patient_ids,

        "label":
            np.asarray(
                labels
            ).astype(int),

        "probability":
            np.asarray(
                probability
            ).astype(float),

    })

    rows = []

    for patient, group in df.groupby(
        "patient_id"
    ):

        rows.append({

            "patient_id":
                str(patient),

            "label":
                int(
                    group[
                        "label"
                    ].iloc[0]
                ),

            "probability":
                float(
                    group[
                        "probability"
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD LATENT
# ============================================================

print()
print("=" * 90)
print("STEP 35B - PARAMETER PERTURBATION")
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

test_patient_id = [
    str(x)
    for x
    in data[
        "internal_test_patient_id"
    ]
]

X = test_z.numpy()
Y = test_y.numpy().astype(int)


print()
print(
    "Test latent:",
    X.shape,
)

print(
    "Test records:",
    len(Y),
)

print(
    "Test patients:",
    len(
        set(
            test_patient_id
        )
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


mlp_base = MatchedMLP()

mlp_base.load_state_dict(
    torch.load(
        MLP_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
)

mlp_base.eval()


def mlp_predict_with_state(
    state,
):

    model = MatchedMLP()

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    with torch.no_grad():

        p = torch.sigmoid(
            model(
                test_z
            )
        )

    return (
        p.numpy()
    )


mlp_base_state = {
    k:
        v.detach()
        .clone()
    for k, v
    in mlp_base.state_dict().items()
}


# ============================================================
# VQC
# ============================================================

vqc_state = torch.load(
    VQC_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

if "theta" not in vqc_state:

    raise RuntimeError(
        "VQC checkpoint does not contain theta."
    )

base_theta = (
    vqc_state[
        "theta"
    ]
    .detach()
    .cpu()
    .numpy()
)

if base_theta.shape != (
    VQC_DEPTH,
    N_QUBITS,
    2,
):

    raise RuntimeError(
        f"Unexpected VQC theta shape: "
        f"{base_theta.shape}"
    )


def vqc_predict(
    inputs,
    theta,
):

    dev = qml.device(
        "lightning.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(
        dev,
        interface="numpy",
    )
    def circuit(x):

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

    out = []

    for row in inputs:

        z = float(
            circuit(
                row
            )
        )

        p = (
            z + 1.0
        ) / 2.0

        out.append(
            float(
                np.clip(
                    p,
                    EPS,
                    1.0 - EPS,
                )
            )
        )

    return np.asarray(
        out,
        dtype=float,
    )


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

print()
print(
    "Computing baseline predictions..."
)

mlp_base_probability = (
    mlp_predict_with_state(
        mlp_base_state
    )
)

vqc_base_probability = (
    vqc_predict(
        X,
        base_theta,
    )
)


# ============================================================
# PERTURBATION LOOP
# ============================================================

record_rows = []
summary_rows = []

for sigma in SIGMAS:

    print()
    print(
        "-" * 80
    )

    print(
        f"PERTURBATION SCALE: {sigma}"
    )

    mlp_repeated = []
    vqc_repeated = []

    # --------------------------------------------------------
    # Same perturbation count for both models.
    # --------------------------------------------------------

    for repeat in range(
        REPEATS
    ):

        rng = np.random.default_rng(
            SEED
            + int(
                sigma
                * 1000
            )
            + repeat
        )

        # ----------------------------------------------------
        # MLP perturbation
        # ----------------------------------------------------

        mlp_state = {}

        for name, tensor in (
            mlp_base_state.items()
        ):

            scale = float(
                tensor.abs()
                .mean()
                .item()
            )

            noise = torch.tensor(
                rng.normal(
                    0.0,
                    sigma * max(
                        scale,
                        1e-6,
                    ),
                    size=tensor.shape,
                ),
                dtype=tensor.dtype,
            )

            mlp_state[
                name
            ] = (
                tensor
                + noise
            )

        mlp_p = mlp_predict_with_state(
            mlp_state
        )

        mlp_repeated.append(
            mlp_p
        )

        # ----------------------------------------------------
        # VQC perturbation
        # ----------------------------------------------------

        theta_scale = max(
            float(
                np.mean(
                    np.abs(
                        base_theta
                    )
                )
            ),
            1e-6,
        )

        theta_noise = rng.normal(
            0.0,
            sigma * theta_scale,
            size=base_theta.shape,
        )

        perturbed_theta = (
            base_theta
            + theta_noise
        )

        vqc_p = vqc_predict(
            X,
            perturbed_theta,
        )

        vqc_repeated.append(
            vqc_p
        )

        print(
            f"  repeat {repeat + 1}/{REPEATS} complete",
            flush=True,
        )


    mlp_repeated = np.vstack(
        mlp_repeated
    )

    vqc_repeated = np.vstack(
        vqc_repeated
    )


    # ========================================================
    # PER-RECORD VARIABILITY
    # ========================================================

    mlp_mean = (
        mlp_repeated.mean(
            axis=0
        )
    )

    vqc_mean = (
        vqc_repeated.mean(
            axis=0
        )
    )

    mlp_var = (
        mlp_repeated.var(
            axis=0,
            ddof=1,
        )
    )

    vqc_var = (
        vqc_repeated.var(
            axis=0,
            ddof=1,
        )
    )

    mlp_std = np.sqrt(
        mlp_var
    )

    vqc_std = np.sqrt(
        vqc_var
    )

    mlp_entropy = binary_entropy(
        mlp_mean
    )

    vqc_entropy = binary_entropy(
        vqc_mean
    )


    # ========================================================
    # ERROR DETECTION
    # ========================================================

    mlp_auc = error_detection_auroc(
        Y,
        mlp_mean,
        mlp_var,
    )

    vqc_auc = error_detection_auroc(
        Y,
        vqc_mean,
        vqc_var,
    )

    mlp_entropy_auc = error_detection_auroc(
        Y,
        mlp_mean,
        mlp_entropy,
    )

    vqc_entropy_auc = error_detection_auroc(
        Y,
        vqc_mean,
        vqc_entropy,
    )


    # ========================================================
    # AURC
    # ========================================================

    _, _, mlp_aurc = risk_coverage(
        Y,
        mlp_mean,
        mlp_var,
    )

    _, _, vqc_aurc = risk_coverage(
        Y,
        vqc_mean,
        vqc_var,
    )

    _, _, mlp_entropy_aurc = risk_coverage(
        Y,
        mlp_mean,
        mlp_entropy,
    )

    _, _, vqc_entropy_aurc = risk_coverage(
        Y,
        vqc_mean,
        vqc_entropy,
    )


    # ========================================================
    # PATIENT-LEVEL
    # ========================================================

    mlp_patient = patient_aggregate(
        mlp_mean,
        Y,
        test_patient_id,
    )

    vqc_patient = patient_aggregate(
        vqc_mean,
        Y,
        test_patient_id,
    )

    # Aggregate per-record variance to patient mean.
    mlp_var_df = pd.DataFrame(
        {
            "patient_id":
                test_patient_id,
            "variance":
                mlp_var,
        }
    )

    vqc_var_df = pd.DataFrame(
        {
            "patient_id":
                test_patient_id,
            "variance":
                vqc_var,
        }
    )

    mlp_patient_var = (
        mlp_var_df.groupby(
            "patient_id"
        )[
            "variance"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "variance":
                    "variance"
            }
        )
    )

    vqc_patient_var = (
        vqc_var_df.groupby(
            "patient_id"
        )[
            "variance"
        ]
        .mean()
        .reset_index()
    )

    mlp_patient = (
        mlp_patient
        .merge(
            mlp_patient_var,
            on="patient_id",
            validate="one_to_one",
        )
    )

    vqc_patient = (
        vqc_patient
        .merge(
            vqc_patient_var,
            on="patient_id",
            validate="one_to_one",
        )
    )


    mlp_patient_auc = error_detection_auroc(
        mlp_patient[
            "label"
        ],
        mlp_patient[
            "probability"
        ],
        mlp_patient[
            "variance"
        ],
    )

    vqc_patient_auc = error_detection_auroc(
        vqc_patient[
            "label"
        ],
        vqc_patient[
            "probability"
        ],
        vqc_patient[
            "variance"
        ],
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append(
        {

            "model":
                "MLP",

            "perturbation_sigma":
                sigma,

            "repeats":
                REPEATS,

            "mean_prediction":
                float(
                    mlp_mean.mean()
                ),

            "mean_parameter_variance":
                float(
                    mlp_var.mean()
                ),

            "mean_parameter_std":
                float(
                    mlp_std.mean()
                ),

            "mean_predictive_entropy":
                float(
                    mlp_entropy.mean()
                ),

            "error_detection_auroc_variance":
                float(
                    mlp_auc
                ),

            "error_detection_auroc_entropy":
                float(
                    mlp_entropy_auc
                ),

            "aurc_variance":
                float(
                    mlp_aurc
                ),

            "aurc_entropy":
                float(
                    mlp_entropy_aurc
                ),

            "patient_error_detection_auroc_variance":
                float(
                    mlp_patient_auc
                ),
        }
    )

    summary_rows.append(
        {

            "model":
                "VQC",

            "perturbation_sigma":
                sigma,

            "repeats":
                REPEATS,

            "mean_prediction":
                float(
                    vqc_mean.mean()
                ),

            "mean_parameter_variance":
                float(
                    vqc_var.mean()
                ),

            "mean_parameter_std":
                float(
                    vqc_std.mean()
                ),

            "mean_predictive_entropy":
                float(
                    vqc_entropy.mean()
                ),

            "error_detection_auroc_variance":
                float(
                    vqc_auc
                ),

            "error_detection_auroc_entropy":
                float(
                    vqc_entropy_auc
                ),

            "aurc_variance":
                float(
                    vqc_aurc
                ),

            "aurc_entropy":
                float(
                    vqc_entropy_aurc
                ),

            "patient_error_detection_auroc_variance":
                float(
                    vqc_patient_auc
                ),
        }
    )


    # --------------------------------------------------------
    # Source records
    # --------------------------------------------------------

    for i in range(
        len(Y)
    ):

        record_rows.append(
            {

                "model":
                    "MLP",

                "perturbation_sigma":
                    sigma,

                "record_index":
                    i,

                "patient_id":
                    test_patient_id[i],

                "label":
                    int(Y[i]),

                "base_probability":
                    float(
                        mlp_base_probability[i]
                    ),

                "perturbed_mean_probability":
                    float(
                        mlp_mean[i]
                    ),

                "parameter_variance":
                    float(
                        mlp_var[i]
                    ),

                "parameter_std":
                    float(
                        mlp_std[i]
                    ),

                "predictive_entropy":
                    float(
                        mlp_entropy[i]
                    ),
            }
        )

        record_rows.append(
            {

                "model":
                    "VQC",

                "perturbation_sigma":
                    sigma,

                "record_index":
                    i,

                "patient_id":
                    test_patient_id[i],

                "label":
                    int(Y[i]),

                "base_probability":
                    float(
                        vqc_base_probability[i]
                    ),

                "perturbed_mean_probability":
                    float(
                        vqc_mean[i]
                    ),

                "parameter_variance":
                    float(
                        vqc_var[i]
                    ),

                "parameter_std":
                    float(
                        vqc_std[i]
                    ),

                "predictive_entropy":
                    float(
                        vqc_entropy[i]
                    ),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)

record_df = pd.DataFrame(
    record_rows
)


# ============================================================
# SAVE TABLES
# ============================================================

summary_df.to_csv(
    TABLE_DIR
    / "TABLE_35B_PARAMETER_PERTURBATION.csv",
    index=False,
)

record_df.to_csv(
    SOURCE_DIR
    / "PARAMETER_PERTURBATION_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# PLOT — MEAN VARIANCE VS PERTURBATION
# ============================================================

fig = plt.figure(
    figsize=(6.5, 5.2)
)

for model in [
    "MLP",
    "VQC",
]:

    subset = summary_df[
        summary_df[
            "model"
        ]
        == model
    ]

    plt.plot(
        subset[
            "perturbation_sigma"
        ],
        subset[
            "mean_parameter_std"
        ],
        marker="o",
        linewidth=2,
        label=model,
    )

plt.xlabel(
    "Relative parameter perturbation"
)

plt.ylabel(
    "Mean predictive standard deviation"
)

plt.title(
    "Parameter-induced predictive variability"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_35B_PARAMETER_VARIABILITY.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35B_PARAMETER_VARIABILITY.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35B_PARAMETER_VARIABILITY_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# RESULTS
# ============================================================

result = {

    "status":
        "STEP35B_COMPLETE",

    "dataset":
        "CBIS-DDSM",

    "test_records":
        int(len(Y)),

    "test_patients":
        int(
            len(
                set(
                    test_patient_id
                )
            )
        ),

    "perturbation_scales":
        SIGMAS,

    "repeats":
        REPEATS,

    "models":
        {
            "MLP":
                {
                    "parameters":
                        25,
                },

            "VQC":
                {
                    "parameters":
                        24,

                    "qubits":
                        N_QUBITS,

                    "depth":
                        VQC_DEPTH,
                },
        },

    "interpretation":
        (
            "Parameter-induced predictive variability "
            "and sensitivity. This is not interpreted "
            "as a Bayesian posterior without an explicit "
            "posterior or ensemble construction."
        ),

    "baseline":
        {
            "mlp":
                {
                    "mean_entropy":
                        float(
                            binary_entropy(
                                mlp_base_probability
                            ).mean()
                        ),
                },

            "vqc":
                {
                    "mean_entropy":
                        float(
                            binary_entropy(
                                vqc_base_probability
                            ).mean()
                        ),
                },
        },

    "summary":
        summary_rows,
}

(
    METRIC_DIR
    / "STEP35B_FINAL_RESULTS.json"
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

    "perturbation_scales":
        SIGMAS,

    "repeats":
        REPEATS,

    "interpretation":
        "parameter-induced predictive variability",

    "mlp_parameters":
        25,

    "vqc_parameters":
        24,

    "vqc_qubits":
        N_QUBITS,

    "vqc_depth":
        VQC_DEPTH,

    "source_latent":
        str(
            LATENT_FILE
        ),

    "source_mlp_checkpoint":
        str(
            MLP_CHECKPOINT
        ),

    "source_vqc_checkpoint":
        str(
            VQC_CHECKPOINT
        ),

    "inference_only":
        True,
}


(
    CONFIG_DIR
    / "STEP35B_CONFIGURATION.json"
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

    "torch":
        torch.__version__,

    "pennylane":
        qml.__version__,

    "backend":
        "lightning.qubit",

    "status":
        "INFERENCE_ONLY",
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

rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
        continue

    rows.append(
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
    rows
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
print(
    "STEP 35B COMPLETE"
)
print("=" * 90)

print()
print(
    "Results:",
    METRIC_DIR
    / "STEP35B_FINAL_RESULTS.json"
)

print(
    "Table:",
    TABLE_DIR
    / "TABLE_35B_PARAMETER_PERTURBATION.csv"
)

print(
    "Source:",
    SOURCE_DIR
    / "PARAMETER_PERTURBATION_SOURCE_DATA.csv"
)

print(
    "Figures:",
    FIG_DIR
)

print()
print(
    "STATUS: STEP35B_COMPLETE"
)