from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.decomposition import PCA
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

PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

THERMO_DIR = (
    ROOT
    / "experiments"
    / "phase12_thermography_dmr_ir"
)

MASTER_MANIFEST = (
    THERMO_DIR
    / "THERMOGRAPHY_DMRIR_MASTER_MANIFEST.csv"
)

SOURCE_AUDIT = (
    THERMO_DIR
    / "THERMOGRAPHY_DMRIR_SOURCE_AUDIT.json"
)

RUN34A = (
    ROOT
    / "experiments"
    / "STEP34A_V2_FINAL_ASUS_PUBLICATION"
)

RUN34B = (
    ROOT
    / "experiments"
    / "STEP34B_FINAL_MATCHED_MLP_VQC"
)

RESNET_CHECKPOINT = (
    RUN34A
    / "checkpoints"
    / "BEST_RESNET50_STATE_DICT.pt"
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

TRAIN_FEATURE_FILE = (
    RUN34B
    / "features"
    / "TRAIN_RESNET50_2048_FEATURES.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP41C_THERMOGRAPHY_EXTERNAL_INFERENCE"
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

IMAGE_SIZE = 512
RESIZE_SIZE = 576
BATCH_SIZE = 16

N_QUBITS = 6
VQC_DEPTH = 2

EPS = 1e-8


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HASH
# ============================================================

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


# ============================================================
# METRICS
# ============================================================

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
        (1.0 - p)
        * np.log(
            1.0 - p
        )
    )


def brier(y, p):

    return float(
        np.mean(
            (
                np.asarray(p)
                -
                np.asarray(y)
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
            (1.0 - y)
            * np.log(
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

    value = 0.0
    total = len(y)

    for i in range(bins):

        lo = edges[i]
        hi = edges[i + 1]

        if i == bins - 1:

            mask = (
                (p >= lo)
                &
                (p <= hi)
            )

        else:

            mask = (
                (p >= lo)
                &
                (p < hi)
            )

        n = int(
            mask.sum()
        )

        if n == 0:
            continue

        accuracy = float(
            y[mask].mean()
        )

        confidence = float(
            p[mask].mean()
        )

        value += (
            n / total
        ) * abs(
            accuracy
            -
            confidence
        )

    return float(value)


def safe_auc(y, p):

    try:

        return float(
            roc_auc_score(
                y,
                p,
            )
        )

    except Exception:

        return float("nan")


def safe_auprc(y, p):

    try:

        return float(
            average_precision_score(
                y,
                p,
            )
        )

    except Exception:

        return float("nan")


def metrics(y, p):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    return {

        "n":
            int(len(y)),

        "positive_count":
            int(y.sum()),

        "negative_count":
            int(len(y) - y.sum()),

        "roc_auc":
            safe_auc(
                y,
                p,
            ),

        "auprc":
            safe_auprc(
                y,
                p,
            ),

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

        "ece":
            ece(
                y,
                p,
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


def error_detection_auroc(
    y,
    p,
    u,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    u = np.asarray(
        u,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    error = (
        pred != y
    ).astype(int)

    if len(
        np.unique(error)
    ) < 2:

        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            error,
            u,
        )
    )


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 100)
print(
    "STEP 41C - THERMOGRAPHY EXTERNAL INFERENCE"
)
print("=" * 100)

for path in [
    MASTER_MANIFEST,
    SOURCE_AUDIT,
    RESNET_CHECKPOINT,
    MLP_CHECKPOINT,
    VQC_CHECKPOINT,
    TRAIN_FEATURE_FILE,
]:

    if not path.is_file():

        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


# ============================================================
# LOAD MASTER MANIFEST
# ============================================================

manifest = pd.read_csv(
    MASTER_MANIFEST
)

required = [
    "image_path",
    "source",
    "top_level_class",
    "label",
    "label_name",
    "subject_id",
    "relative_path",
    "width",
    "height",
    "image_mode",
    "sha256",
]

missing = [
    x for x in required
    if x not in manifest.columns
]

if missing:

    raise RuntimeError(
        f"Thermography manifest missing columns: {missing}"
    )


if len(manifest) != 2751:

    raise RuntimeError(
        f"Expected 2751 thermography records; "
        f"found {len(manifest)}."
    )


print()
print(
    "Total records:",
    len(manifest),
)

print(
    "Sources:",
    manifest[
        "source"
    ].value_counts()
    .to_dict(),
)

print(
    "Labels:",
    manifest[
        "label_name"
    ].value_counts()
    .to_dict(),
)


# ============================================================
# SOURCE CONTRACT
# ============================================================

expected_source_counts = {

    "DMR_IR":
        2394,

    "THERMOGRAPHY_BENIGN_MALIGNANT":
        357,
}

actual_source_counts = (
    manifest[
        "source"
    ]
    .value_counts()
    .to_dict()
)

if actual_source_counts != expected_source_counts:

    raise RuntimeError(
        "Thermography source counts do not match "
        f"the frozen source contract: {actual_source_counts}"
    )


dmr = manifest[
    manifest[
        "source"
    ]
    ==
    "DMR_IR"
].copy()

mendeley = manifest[
    manifest[
        "source"
    ]
    ==
    "THERMOGRAPHY_BENIGN_MALIGNANT"
].copy()


# ============================================================
# LABEL CONTRACT
# ============================================================

dmr_labels = set(
    dmr[
        "label_name"
    ].astype(str)
)

mendeley_labels = set(
    mendeley[
        "label_name"
    ].astype(str)
)

if dmr_labels != {
    "Healthy",
    "Sick",
}:

    raise RuntimeError(
        f"Unexpected DMR-IR labels: {dmr_labels}"
    )

if mendeley_labels != {
    "Benign",
    "Malignant",
}:

    raise RuntimeError(
        f"Unexpected Mendeley labels: {mendeley_labels}"
    )


# ============================================================
# PATH VALIDATION
# ============================================================

resolved = []

for raw in manifest[
    "image_path"
].astype(str):

    p = Path(
        raw
    )

    if p.is_file():

        resolved.append(
            p.resolve()
        )

        continue

    # Try dataset root relative path.
    candidate = (
        ROOT
        / raw
    )

    if candidate.is_file():

        resolved.append(
            candidate.resolve()
        )

        continue

    # Try thermography directory relative path.
    candidate = (
        ROOT
        / "Breast Thermography"
        / raw
    )

    if candidate.is_file():

        resolved.append(
            candidate.resolve()
        )

        continue

    resolved.append(
        None
    )


if any(
    x is None
    for x in resolved
):

    count = sum(
        x is None
        for x in resolved
    )

    raise RuntimeError(
        f"Unresolved thermography image paths: {count}"
    )


manifest[
    "resolved_path"
] = [
    str(x)
    for x in resolved
]


# ============================================================
# IMAGE HASH VALIDATION
# ============================================================

print()
print(
    "Validating physical image hashes..."
)

hash_failures = []

for i, row in manifest.iterrows():

    path = Path(
        row[
            "resolved_path"
        ]
    )

    actual = sha256_file(
        path
    )

    expected = str(
        row[
            "sha256"
        ]
    )

    if actual != expected:

        hash_failures.append({

            "row":
                int(i),

            "path":
                str(path),

            "expected_sha256":
                expected,

            "actual_sha256":
                actual,
        })

if hash_failures:

    pd.DataFrame(
        hash_failures
    ).to_csv(
        SOURCE_DIR
        / "HASH_VALIDATION_FAILURES.csv",
        index=False,
    )

    raise RuntimeError(
        f"{len(hash_failures)} thermography image hashes failed."
    )

print(
    "Image hash validation: PASS"
)


# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            RESIZE_SIZE,
            RESIZE_SIZE,
        )
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# RESNET
# ============================================================

print()
print(
    "Loading frozen ResNet-50..."
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

cnn = models.resnet50(
    weights=None
)

cnn.fc = nn.Linear(
    cnn.fc.in_features,
    1,
)

cnn.load_state_dict(
    torch.load(
        RESNET_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    ),
    strict=True,
)

cnn.fc = nn.Identity()

cnn.eval()
cnn.to(device)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print()
print(
    "Extracting frozen ResNet features..."
)

features = []

paths = manifest[
    "resolved_path"
].tolist()

with torch.no_grad():

    for start in range(
        0,
        len(paths),
        BATCH_SIZE,
    ):

        batch_paths = paths[
            start:
            start + BATCH_SIZE
        ]

        tensors = []

        for path in batch_paths:

            image = (
                Image.open(
                    path
                )
                .convert(
                    "RGB"
                )
            )

            tensors.append(
                TRANSFORM(
                    image
                )
            )

        batch = torch.stack(
            tensors
        ).to(
            device
        )

        z = cnn(
            batch
        )

        features.append(
            z.detach()
            .cpu()
            .numpy()
        )

        done = min(
            start + len(batch_paths),
            len(paths),
        )

        if (
            done % 250 == 0
            or
            done == len(paths)
        ):

            print(
                f"  {done}/{len(paths)}",
                flush=True,
            )


features = np.concatenate(
    features,
    axis=0,
)


if features.shape != (
    2751,
    2048,
):

    raise RuntimeError(
        f"Unexpected ResNet feature shape: {features.shape}"
    )


# ============================================================
# FROZEN TRAINING PCA
# ============================================================

print()
print(
    "Reconstructing frozen 34B training PCA..."
)

train_feature_data = torch.load(
    TRAIN_FEATURE_FILE,
    map_location="cpu",
    weights_only=False,
)

if isinstance(
    train_feature_data,
    dict
):

    if "features" in train_feature_data:

        train_features = (
            train_feature_data[
                "features"
            ]
        )

    elif "train_features" in train_feature_data:

        train_features = (
            train_feature_data[
                "train_features"
            ]
        )

    else:

        raise RuntimeError(
            "Training feature key not found."
        )

else:

    train_features = (
        train_feature_data
    )


if torch.is_tensor(
    train_features
):

    train_features = (
        train_features
        .detach()
        .cpu()
        .numpy()
    )

train_features = np.asarray(
    train_features,
    dtype=np.float32,
)

if train_features.shape[1] != 2048:

    raise RuntimeError(
        "Frozen training features are not 2048-D."
    )

pca = PCA(
    n_components=6
)

pca.fit(
    train_features
)

print(
    "Frozen PCA explained variance:",
    pca.explained_variance_ratio_,
)

print(
    "Total:",
    float(
        pca.explained_variance_ratio_.sum()
    ),
)

z6 = pca.transform(
    features
)


# ============================================================
# LOAD MLP
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
    ),
    strict=True,
)

mlp.eval()

with torch.no_grad():

    mlp_probability = (
        torch.sigmoid(
            mlp(
                torch.tensor(
                    z6,
                    dtype=torch.float32,
                )
            )
        )
        .numpy()
    )


# ============================================================
# LOAD VQC
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


# ============================================================
# VQC
# ============================================================

vqc_device = qml.device(
    "lightning.qubit",
    wires=N_QUBITS,
)


@qml.qnode(
    vqc_device
)
def vqc_circuit(
    x
):

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


print()
print(
    "Generating frozen VQC predictions..."
)

vqc_probability = []

for i, row in enumerate(
    z6,
    1,
):

    value = float(
        vqc_circuit(
            row
        )
    )

    p = (
        value + 1.0
    ) / 2.0

    vqc_probability.append(
        float(
            np.clip(
                p,
                EPS,
                1.0 - EPS,
            )
        )
    )

    if (
        i % 250 == 0
        or
        i == len(z6)
    ):

        print(
            f"  {i}/{len(z6)}",
            flush=True,
        )


vqc_probability = np.asarray(
    vqc_probability,
    dtype=float,
)


# ============================================================
# ADD RESULTS
# ============================================================

manifest[
    "mlp_probability"
] = mlp_probability

manifest[
    "vqc_probability"
] = vqc_probability

manifest[
    "mlp_entropy"
] = entropy(
    mlp_probability
)

manifest[
    "vqc_entropy"
] = entropy(
    vqc_probability
)


# ============================================================
# COHORT ANALYSIS
# ============================================================

cohort_rows = []

cohorts = [
    (
        "DMR_IR",
        dmr.index,
    ),
    (
        "Mendeley",
        mendeley.index,
    ),
]

for cohort_name, indices in cohorts:

    subset = manifest.loc[
        indices
    ]

    y = subset[
        "label"
    ].astype(int).to_numpy()

    for model_name, probability_column, entropy_column in [

        (
            "MLP",
            "mlp_probability",
            "mlp_entropy",
        ),

        (
            "VQC",
            "vqc_probability",
            "vqc_entropy",
        ),

    ]:

        p = subset[
            probability_column
        ].to_numpy()

        u = subset[
            entropy_column
        ].to_numpy()

        m = metrics(
            y,
            p,
        )

        cohort_rows.append({

            "cohort":
                cohort_name,

            "model":
                model_name,

            **m,

            "error_detection_auroc":
                error_detection_auroc(
                    y,
                    p,
                    u,
                ),

        })


cohort_results = pd.DataFrame(
    cohort_rows
)


# ============================================================
# OVERALL SEPARATE-COHORT SUMMARY
# ============================================================

summary_rows = []

for cohort_name, indices in cohorts:

    subset = manifest.loc[
        indices
    ]

    y = subset[
        "label"
    ].astype(int).to_numpy()

    for model_name, probability_column in [

        (
            "MLP",
            "mlp_probability",
        ),

        (
            "VQC",
            "vqc_probability",
        ),

    ]:

        p = subset[
            probability_column
        ].to_numpy()

        m = metrics(
            y,
            p,
        )

        summary_rows.append({

            "cohort":
                cohort_name,

            "model":
                model_name,

            **m,

        })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SOURCE DATA
# ============================================================

source_columns = [

    "image_path",
    "resolved_path",
    "source",
    "top_level_class",
    "label",
    "label_name",
    "subject_id",
    "relative_path",
    "width",
    "height",
    "image_mode",
    "sha256",
    "mlp_probability",
    "vqc_probability",
    "mlp_entropy",
    "vqc_entropy",
]

manifest[
    source_columns
].to_csv(
    SOURCE_DIR
    / "THERMOGRAPHY_EXTERNAL_PREDICTIONS.csv",
    index=False,
)


# ============================================================
# TABLES
# ============================================================

cohort_results.to_csv(
    TABLE_DIR
    / "TABLE_41C_THERMOGRAPHY_RESULTS.csv",
    index=False,
)

summary_df.to_csv(
    TABLE_DIR
    / "TABLE_41C_THERMOGRAPHY_SUMMARY.csv",
    index=False,
)


# ============================================================
# FIGURE 1 - AUROC
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.2,
    )
)

labels = []
values = []

for _, row in summary_df.iterrows():

    labels.append(
        f"{row['cohort']}\n{row['model']}"
    )

    values.append(
        row["roc_auc"]
    )

plt.bar(
    np.arange(
        len(values)
    ),
    values,
)

plt.xticks(
    np.arange(
        len(values)
    ),
    labels,
    rotation=20,
    ha="right",
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "Thermography external-transfer performance"
)

plt.ylim(
    0.0,
    1.0,
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_AUROC.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_AUROC.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_AUROC_400DPI.png",
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
        5.2,
    )
)

labels = []
values = []

for _, row in summary_df.iterrows():

    labels.append(
        f"{row['cohort']}\n{row['model']}"
    )

    values.append(
        row["mean_entropy"]
    )

plt.bar(
    np.arange(
        len(values)
    ),
    values,
)

plt.xticks(
    np.arange(
        len(values)
    ),
    labels,
    rotation=20,
    ha="right",
)

plt.ylabel(
    "Mean predictive entropy"
)

plt.title(
    "Thermography predictive uncertainty"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_ENTROPY.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_ENTROPY.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41C_THERMOGRAPHY_ENTROPY_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP41C_COMPLETE",

    "dataset":
        "Thermography",

    "total_images":
        int(
            len(manifest)
        ),

    "dmr_ir_images":
        int(
            len(dmr)
        ),

    "mendeley_images":
        int(
            len(mendeley)
        ),

    "dmr_ir_labels":
        {
            "Healthy":
                int(
                    (
                        dmr[
                            "label_name"
                        ]
                        ==
                        "Healthy"
                    ).sum()
                ),

            "Sick":
                int(
                    (
                        dmr[
                            "label_name"
                        ]
                        ==
                        "Sick"
                    ).sum()
                ),
        },

    "mendeley_labels":
        {
            "Benign":
                int(
                    (
                        mendeley[
                            "label_name"
                        ]
                        ==
                        "Benign"
                    ).sum()
                ),

            "Malignant":
                int(
                    (
                        mendeley[
                            "label_name"
                        ]
                        ==
                        "Malignant"
                    ).sum()
                ),
        },

    "cross_source_subject_overlap":
        0,

    "models":
        [
            "MLP",
            "VQC",
        ],

    "feature_dimension":
        2048,

    "latent_dimension":
        6,

    "frozen_encoder":
        "CBIS-DDSM ResNet-50",

    "analysis_role":
        "independent cross-modal thermography transfer stress test",

    "training_performed":
        False,

    "external_adaptation":
        False,

    "original_images_modified":
        False,

    "cohort_results":
        cohort_rows,

    "source_audit":
        str(
            SOURCE_AUDIT
        ),

    "artifacts":
        {

            "source":
                str(
                    SOURCE_DIR
                    / "THERMOGRAPHY_EXTERNAL_PREDICTIONS.csv"
                ),

            "results":
                str(
                    TABLE_DIR
                    / "TABLE_41C_THERMOGRAPHY_RESULTS.csv"
                ),

            "summary":
                str(
                    TABLE_DIR
                    / "TABLE_41C_THERMOGRAPHY_SUMMARY.csv"
                ),

            "figures":
                str(
                    FIG_DIR
                ),
        },
}


(
    METRIC_DIR
    / "STEP41C_FINAL_RESULTS.json"
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

    "input_manifest":
        str(
            MASTER_MANIFEST
        ),

    "source_audit":
        str(
            SOURCE_AUDIT
        ),

    "cohorts":
        {
            "DMR_IR":
                {
                    "task":
                        "Healthy_vs_Sick",

                    "records":
                        2394,

                    "healthy":
                        1263,

                    "sick":
                        1131,
                },

            "Mendeley":
                {
                    "task":
                        "Benign_vs_Malignant",

                    "records":
                        357,

                    "benign":
                        252,

                    "malignant":
                        105,
                },
        },

    "preprocessing":
        {
            "resize":
                [
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ],

            "center_crop":
                [
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ],

            "normalization":
                "ImageNet mean/std",

        },

    "frozen_encoder":
        "ResNet-50",

    "frozen_features":
        2048,

    "frozen_pca":
        "training-fitted 6-D PCA",

    "mlp_parameters":
        25,

    "vqc_qubits":
        6,

    "vqc_depth":
        2,

    "vqc_parameters":
        24,

    "training":
        False,

    "external_adaptation":
        False,

    "cohort_merging":
        False,

    "interpretation":
        "Cross-modal external transfer/stress test. "
        "DMR-IR Healthy/Sick and Mendeley Benign/Malignant "
        "are analyzed as separate diagnostic tasks.",
}


(
    CONFIG_DIR
    / "STEP41C_CONFIGURATION.json"
).write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# ENVIRONMENT
# ============================================================

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
    "STEP 41C COMPLETE"
)
print("=" * 100)

print()
print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "Source:",
    SOURCE_DIR
    / "THERMOGRAPHY_EXTERNAL_PREDICTIONS.csv",
)

print(
    "Results:",
    METRIC_DIR
    / "STEP41C_FINAL_RESULTS.json",
)

print(
    "Tables:",
    TABLE_DIR,
)

print(
    "Figures:",
    FIG_DIR,
)

print()
print(
    "STATUS: STEP41C_COMPLETE"
)