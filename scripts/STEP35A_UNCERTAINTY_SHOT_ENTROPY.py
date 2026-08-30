from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import random
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

import pennylane as qml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 2026

ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

RUN34B = ROOT / "experiments" / "STEP34B_FINAL_MATCHED_MLP_VQC"

LATENT_FILE = RUN34B / "latent" / "SHARED_6D_LATENTS.pt"

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

OUT = ROOT / "experiments" / "STEP35A_UNCERTAINTY"

PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
SOURCE_DIR = OUT / "source_data"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    PRED_DIR,
    METRIC_DIR,
    SOURCE_DIR,
    FIG_DIR,
    TABLE_DIR,
    CONFIG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

SHOTS_LIST = [256, 1024, 4096]
REPEATS = 5

N_QUBITS = 6
VQC_DEPTH = 2
EPS = 1e-8


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def error_detection_auroc(y, probability, uncertainty):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    predictions = (probability >= 0.5).astype(int)
    errors = (predictions != y).astype(int)

    if len(np.unique(errors)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            errors,
            uncertainty,
        )
    )


def risk_coverage(y, probability, uncertainty):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    predictions = (probability >= 0.5).astype(int)
    errors = (predictions != y).astype(float)

    order = np.argsort(uncertainty)
    ordered_errors = errors[order]

    coverages = []
    risks = []

    cumulative = 0.0
    n = len(ordered_errors)

    for k in range(1, n + 1):
        cumulative += ordered_errors[k - 1]
        coverages.append(k / n)
        risks.append(cumulative / k)

    coverages = np.asarray(coverages, dtype=float)
    risks = np.asarray(risks, dtype=float)

    aurc = float(
        np.trapezoid(
            risks,
            coverages,
        )
    )

    return coverages, risks, aurc


def aggregate_patient(probability, labels, patient_ids):
    temp = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": np.asarray(labels).astype(int),
            "probability": np.asarray(probability).astype(float),
        }
    )

    rows = []

    for patient, group in temp.groupby("patient_id"):
        rows.append(
            {
                "patient_id": str(patient),
                "label": int(group["label"].iloc[0]),
                "probability": float(group["probability"].mean()),
            }
        )

    return pd.DataFrame(rows)


print()
print("=" * 90)
print("STEP 35A - VQC SHOT UNCERTAINTY")
print("=" * 90)


# ------------------------------------------------------------
# CHECK INPUTS
# ------------------------------------------------------------

for path in [
    LATENT_FILE,
    MLP_CHECKPOINT,
    VQC_CHECKPOINT,
]:
    if not path.is_file():
        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


# ------------------------------------------------------------
# LOAD LATENT
# ------------------------------------------------------------

data = torch.load(
    LATENT_FILE,
    map_location="cpu",
    weights_only=False,
)

test_z = (
    data["internal_test_z"]
    .float()
    .cpu()
)

test_y = (
    data["internal_test_y"]
    .float()
    .cpu()
)

test_patient_id = [
    str(x)
    for x in data["internal_test_patient_id"]
]

print()
print(
    "Internal-test latent shape:",
    tuple(test_z.shape),
)

print(
    "Internal-test labels:",
    len(test_y),
)

print(
    "Internal-test unique patients:",
    len(set(test_patient_id)),
)


# ------------------------------------------------------------
# MLP BASELINE
# ------------------------------------------------------------

class MatchedMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(6, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


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
    mlp_probability = (
        torch.sigmoid(
            mlp(test_z)
        )
        .numpy()
    )

mlp_entropy = binary_entropy(
    mlp_probability
)

mlp_error_auroc = error_detection_auroc(
    test_y.numpy(),
    mlp_probability,
    mlp_entropy,
)

mlp_cover, mlp_risk, mlp_aurc = risk_coverage(
    test_y.numpy(),
    mlp_probability,
    mlp_entropy,
)

mlp_patient = aggregate_patient(
    mlp_probability,
    test_y.numpy(),
    test_patient_id,
)

mlp_patient_entropy = binary_entropy(
    mlp_patient["probability"].to_numpy()
)

mlp_patient_error_auroc = error_detection_auroc(
    mlp_patient["label"].to_numpy(),
    mlp_patient["probability"].to_numpy(),
    mlp_patient_entropy,
)

mlp_patient_cover, mlp_patient_risk, mlp_patient_aurc = risk_coverage(
    mlp_patient["label"].to_numpy(),
    mlp_patient["probability"].to_numpy(),
    mlp_patient_entropy,
)

print()
print("MLP BASELINE")
print("Mean entropy:", float(mlp_entropy.mean()))
print("Error-detection AUROC:", mlp_error_auroc)
print("AURC:", mlp_aurc)


# ------------------------------------------------------------
# LOAD VQC PARAMETERS
# ------------------------------------------------------------

vqc_state = torch.load(
    VQC_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

if "theta" not in vqc_state:
    raise RuntimeError(
        "VQC checkpoint does not contain theta."
    )

theta = (
    vqc_state["theta"]
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
print("VQC")
print("Qubits:", N_QUBITS)
print("Depth:", VQC_DEPTH)
print("Trainable parameters:", theta.size)


# ------------------------------------------------------------
# FINITE-SHOT VQC
# ------------------------------------------------------------

def run_vqc(
    inputs,
    theta,
    shots,
    seed,
):

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
        shots=shots,
        seed=seed,
    )

    @qml.qnode(device)
    def circuit(x):

        for q in range(N_QUBITS):

            qml.RY(
                x[q],
                wires=q,
            )

            qml.RZ(
                x[q],
                wires=q,
            )

        for layer in range(VQC_DEPTH):

            for q in range(N_QUBITS):

                qml.RY(
                    theta[layer, q, 0],
                    wires=q,
                )

                qml.RZ(
                    theta[layer, q, 1],
                    wires=q,
                )

            for q in range(N_QUBITS):

                qml.CNOT(
                    wires=[
                        q,
                        (q + 1) % N_QUBITS,
                    ]
                )

        return qml.expval(
            qml.PauliZ(0)
        )

    outputs = []

    for i in range(len(inputs)):

        z_value = float(
            circuit(
                inputs[i]
            )
        )

        probability = (
            z_value + 1.0
        ) / 2.0

        outputs.append(
            float(
                np.clip(
                    probability,
                    EPS,
                    1.0 - EPS,
                )
            )
        )

    return np.asarray(
        outputs,
        dtype=float,
    )


test_inputs = test_z.numpy()

all_results = []
summary = []


for shots in SHOTS_LIST:

    print()
    print("-" * 80)
    print(
        f"SHOT BUDGET: {shots}"
    )

    repeated = []

    for repeat in range(REPEATS):

        seed = (
            SEED
            + shots * 100
            + repeat
        )

        probabilities = run_vqc(
            test_inputs,
            theta,
            shots,
            seed,
        )

        repeated.append(
            probabilities
        )

        print(
            f"  repeat {repeat + 1}/{REPEATS} complete",
            flush=True,
        )

    repeated = np.vstack(
        repeated
    )

    mean_probability = (
        repeated.mean(
            axis=0
        )
    )

    variance = (
        repeated.var(
            axis=0,
            ddof=1,
        )
    )

    std = np.sqrt(
        variance
    )

    entropy = binary_entropy(
        mean_probability
    )

    entropy_auc = error_detection_auroc(
        test_y.numpy(),
        mean_probability,
        entropy,
    )

    variance_auc = error_detection_auroc(
        test_y.numpy(),
        mean_probability,
        variance,
    )

    cover, risk, aurc = risk_coverage(
        test_y.numpy(),
        mean_probability,
        entropy,
    )

    cover_var, risk_var, aurc_var = risk_coverage(
        test_y.numpy(),
        mean_probability,
        variance,
    )

    summary.append(
        {
            "shot_budget": shots,
            "repeats": REPEATS,
            "mean_probability": float(mean_probability.mean()),
            "mean_shot_variance": float(variance.mean()),
            "mean_shot_std": float(std.mean()),
            "mean_predictive_entropy": float(entropy.mean()),
            "entropy_error_detection_auroc": float(entropy_auc),
            "variance_error_detection_auroc": float(variance_auc),
            "entropy_aurc": float(aurc),
            "variance_aurc": float(aurc_var),
        }
    )

    for i in range(len(test_inputs)):

        all_results.append(
            {
                "shot_budget": shots,
                "record_index": i,
                "patient_id": test_patient_id[i],
                "label": int(test_y[i].item()),
                "mean_probability": float(mean_probability[i]),
                "shot_variance": float(variance[i]),
                "shot_std": float(std[i]),
                "predictive_entropy": float(entropy[i]),
            }
        )

    print()
    print(
        "  Mean shot std:",
        float(std.mean()),
    )

    print(
        "  Mean shot variance:",
        float(variance.mean()),
    )

    print(
        "  Mean predictive entropy:",
        float(entropy.mean()),
    )

    print(
        "  Entropy error-detection AUROC:",
        entropy_auc,
    )

    print(
        "  Variance error-detection AUROC:",
        variance_auc,
    )

    print(
        "  Entropy AURC:",
        aurc,
    )

    print(
        "  Variance AURC:",
        aurc_var,
    )


shot_df = pd.DataFrame(
    all_results
)

summary_df = pd.DataFrame(
    summary
)

shot_df.to_csv(
    SOURCE_DIR
    / "VQC_SHOT_LEVEL_UNCERTAINTY_SOURCE.csv",
    index=False,
)

summary_df.to_csv(
    TABLE_DIR
    / "TABLE_35A_VQC_SHOT_UNCERTAINTY.csv",
    index=False,
)


# ------------------------------------------------------------
# 4096-SHOT PATIENT LEVEL
# ------------------------------------------------------------

best = shot_df[
    shot_df["shot_budget"]
    == 4096
].copy()

vqc_patient = (
    best.groupby(
        "patient_id"
    )
    .agg(
        label=("label", "first"),
        probability=("mean_probability", "mean"),
        shot_variance=("shot_variance", "mean"),
        shot_std=("shot_std", "mean"),
        predictive_entropy=("predictive_entropy", "mean"),
    )
    .reset_index()
)

vqc_patient_error_entropy = error_detection_auroc(
    vqc_patient["label"].to_numpy(),
    vqc_patient["probability"].to_numpy(),
    vqc_patient["predictive_entropy"].to_numpy(),
)

vqc_patient_error_variance = error_detection_auroc(
    vqc_patient["label"].to_numpy(),
    vqc_patient["probability"].to_numpy(),
    vqc_patient["shot_variance"].to_numpy(),
)

vqc_patient_cover, vqc_patient_risk, vqc_patient_aurc = risk_coverage(
    vqc_patient["label"].to_numpy(),
    vqc_patient["probability"].to_numpy(),
    vqc_patient["predictive_entropy"].to_numpy(),
)

vqc_patient_cover_var, vqc_patient_risk_var, vqc_patient_aurc_var = risk_coverage(
    vqc_patient["label"].to_numpy(),
    vqc_patient["probability"].to_numpy(),
    vqc_patient["shot_variance"].to_numpy(),
)

vqc_patient.to_csv(
    PRED_DIR
    / "VQC_PATIENT_LEVEL_4096SHOT_UNCERTAINTY.csv",
    index=False,
)


# ------------------------------------------------------------
# PATIENT COMPARISON
# ------------------------------------------------------------

comparison = (
    vqc_patient[
        [
            "patient_id",
            "label",
            "probability",
            "shot_variance",
            "shot_std",
            "predictive_entropy",
        ]
    ]
    .rename(
        columns={
            "probability": "vqc_probability",
            "shot_variance": "vqc_shot_variance",
            "shot_std": "vqc_shot_std",
            "predictive_entropy": "vqc_predictive_entropy",
        }
    )
    .merge(
        mlp_patient[
            [
                "patient_id",
                "probability",
            ]
        ].rename(
            columns={
                "probability": "mlp_probability",
            }
        ),
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
)

comparison[
    "mlp_predictive_entropy"
] = binary_entropy(
    comparison[
        "mlp_probability"
    ].to_numpy()
)

comparison.to_csv(
    PRED_DIR
    / "PATIENT_LEVEL_UNCERTAINTY_COMPARISON.csv",
    index=False,
)

comparison.to_csv(
    SOURCE_DIR
    / "PATIENT_LEVEL_UNCERTAINTY_SOURCE_DATA.csv",
    index=False,
)


# ------------------------------------------------------------
# FIGURE 1
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(6.5, 5.2)
)

plt.scatter(
    best["predictive_entropy"],
    best["shot_variance"],
    s=14,
    alpha=0.65,
)

plt.xlabel(
    "Predictive entropy"
)

plt.ylabel(
    "VQC shot variance"
)

plt.title(
    "VQC entropy versus shot variance"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_ENTROPY_VS_SHOT_VARIANCE.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_ENTROPY_VS_SHOT_VARIANCE.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_ENTROPY_VS_SHOT_VARIANCE_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 2
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(6.5, 5.2)
)

plt.plot(
    mlp_patient_cover,
    mlp_patient_risk,
    linewidth=2,
    label="Matched MLP entropy",
)

plt.plot(
    vqc_patient_cover,
    vqc_patient_risk,
    linewidth=2,
    label="VQC entropy",
)

plt.xlabel(
    "Coverage"
)

plt.ylabel(
    "Risk"
)

plt.title(
    "Patient-level risk-coverage"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_RISK_COVERAGE.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_RISK_COVERAGE.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_RISK_COVERAGE_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 3
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(6.5, 5.2)
)

plt.plot(
    summary_df["shot_budget"],
    summary_df["mean_shot_std"],
    marker="o",
    linewidth=2,
)

plt.xlabel(
    "Measurement shots"
)

plt.ylabel(
    "Mean shot standard deviation"
)

plt.title(
    "VQC measurement variability"
)

plt.xscale(
    "log",
    base=2,
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_SHOT_VARIABILITY.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_SHOT_VARIABILITY.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_35A_SHOT_VARIABILITY_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

result = {

    "status":
        "STEP35A_COMPLETE",

    "dataset":
        "CBIS-DDSM",

    "test_records":
        int(len(test_z)),

    "test_patients":
        int(len(set(test_patient_id))),

    "vqc_qubits":
        N_QUBITS,

    "vqc_depth":
        VQC_DEPTH,

    "vqc_parameters":
        int(theta.size),

    "shot_budgets":
        SHOTS_LIST,

    "repeats":
        REPEATS,

    "mlp":
        {
            "mean_entropy":
                float(mlp_entropy.mean()),

            "error_detection_auroc":
                float(mlp_error_auroc),

            "aurc":
                float(mlp_aurc),

            "patient_error_detection_auroc":
                float(mlp_patient_error_auroc),

            "patient_aurc":
                float(mlp_patient_aurc),
        },

    "vqc_4096":
        {
            "mean_entropy":
                float(
                    best["predictive_entropy"].mean()
                ),

            "mean_shot_variance":
                float(
                    best["shot_variance"].mean()
                ),

            "mean_shot_std":
                float(
                    best["shot_std"].mean()
                ),

            "patient_error_detection_auroc_entropy":
                float(vqc_patient_error_entropy),

            "patient_error_detection_auroc_variance":
                float(vqc_patient_error_variance),

            "patient_aurc_entropy":
                float(vqc_patient_aurc),

            "patient_aurc_variance":
                float(vqc_patient_aurc_var),
        },

    "artifacts":
        {
            "source":
                str(
                    SOURCE_DIR
                    / "VQC_SHOT_LEVEL_UNCERTAINTY_SOURCE.csv"
                ),

            "table":
                str(
                    TABLE_DIR
                    / "TABLE_35A_VQC_SHOT_UNCERTAINTY.csv"
                ),

            "patient_comparison":
                str(
                    PRED_DIR
                    / "PATIENT_LEVEL_UNCERTAINTY_COMPARISON.csv"
                ),

            "figures":
                str(FIG_DIR),
        },
}


(
    METRIC_DIR
    / "STEP35A_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


config = {

    "seed":
        SEED,

    "shot_budgets":
        SHOTS_LIST,

    "repeats":
        REPEATS,

    "qubits":
        N_QUBITS,

    "depth":
        VQC_DEPTH,

    "parameters":
        int(theta.size),

    "backend":
        "default.qubit",

    "inference_only":
        True,

    "source_latent":
        str(LATENT_FILE),

    "source_vqc_checkpoint":
        str(VQC_CHECKPOINT),

    "status":
        "FROZEN_INFERENCE_ONLY",
}


(
    CONFIG_DIR
    / "STEP35A_CONFIGURATION.json"
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

    "cuda_available":
        bool(
            torch.cuda.is_available()
        ),

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


# ------------------------------------------------------------
# SHA256
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print()
print("=" * 90)
print("STEP 35A COMPLETE")
print("=" * 90)

print()
print(
    "MLP error-detection AUROC:",
    mlp_error_auroc,
)

print(
    "MLP AURC:",
    mlp_aurc,
)

print()
print(
    "VQC 4096 mean entropy:",
    float(
        best["predictive_entropy"].mean()
    ),
)

print(
    "VQC 4096 mean shot variance:",
    float(
        best["shot_variance"].mean()
    ),
)

print(
    "VQC patient entropy error AUROC:",
    vqc_patient_error_entropy,
)

print(
    "VQC patient entropy AURC:",
    vqc_patient_aurc,
)

print()
print(
    "Results:",
    METRIC_DIR
    / "STEP35A_FINAL_RESULTS.json",
)

print(
    "STATUS: STEP35A_COMPLETE"
)