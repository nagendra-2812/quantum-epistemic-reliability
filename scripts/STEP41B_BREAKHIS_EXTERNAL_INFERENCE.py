from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import pennylane as qml

from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

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

BREAKHIS_ROOT = (
    ROOT
    / "BreaKHis_v1"
    / "BreaKHis_v1"
)

MANIFEST_DIR = (
    PROJECT
    / "manifests"
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

# Frozen CBIS artifacts.
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

LATENT_FILE = (
    RUN34B
    / "latent"
    / "SHARED_6D_LATENTS.pt"
)

TRAIN_FEATURE_FILE = (
    RUN34B
    / "features"
    / "TRAIN_RESNET50_2048_FEATURES.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP41B_BREAKHIS_EXTERNAL_INFERENCE"
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
            h.update(block)

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

    total = len(y)
    value = 0.0

    for i in range(
        bins
    ):

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

        value += (
            n / total
        ) * abs(
            float(
                y[mask].mean()
            )
            -
            float(
                p[mask].mean()
            )
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


def calc_metrics(y, p):

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
            float(
                np.mean(
                    (
                        p - y
                    ) ** 2
                )
            ),

        "nll":
            float(
                -np.mean(
                    y * np.log(
                        np.clip(
                            p,
                            EPS,
                            1.0 - EPS,
                        )
                    )
                    +
                    (
                        1.0 - y
                    )
                    *
                    np.log(
                        np.clip(
                            1.0 - p,
                            EPS,
                            1.0 - EPS,
                        )
                    )
                )
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


# ============================================================
# FIND MANIFEST COLUMNS ROBUSTLY
# ============================================================

def find_column(
    columns,
    candidates,
):

    lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:

            return lower[
                candidate.lower()
            ]

    for c in columns:

        text = str(c).lower()

        for candidate in candidates:

            if candidate.lower() in text:

                return c

    return None


# ============================================================
# LOAD FOLDS
# ============================================================

print()
print("=" * 100)
print(
    "STEP 41B - BreaKHis EXTERNAL INFERENCE"
)
print("=" * 100)

fold_frames = []
fold_reports = []

for fold in range(
    1,
    6,
):

    path = (
        MANIFEST_DIR
        /
        f"BreaKHis_FOLD_{fold:02d}_VERIFIED_v1.csv"
    )

    if not path.is_file():

        raise RuntimeError(
            f"Missing fold manifest: {path}"
        )

    df = pd.read_csv(
        path
    )

    path_col = find_column(
        df.columns,
        [
            "image_path",
            "image_file",
            "file_path",
            "path",
            "filepath",
            "image",
        ],
    )

    label_col = find_column(
        df.columns,
        [
            "binary_label",
            "label",
            "class",
            "target",
        ],
    )

    split_col = find_column(
        df.columns,
        [
            "split",
            "partition",
            "subset",
            "set",
        ],
    )

    patient_col = find_column(
        df.columns,
        [
            "patient_id",
            "patient",
            "patient_identifier",
        ],
    )

    case_col = find_column(
        df.columns,
        [
            "case_id",
            "case",
            "specimen_id",
        ],
    )

    if path_col is None:
        raise RuntimeError(
            f"Could not identify image-path column in fold {fold}."
        )

    if label_col is None:
        raise RuntimeError(
            f"Could not identify label column in fold {fold}."
        )

    if split_col is None:
        raise RuntimeError(
            f"Could not identify split column in fold {fold}."
        )

    split_values = (
        df[
            split_col
        ]
        .astype(str)
        .str.lower()
    )

    test_mask = split_values.isin(
        [
            "test",
            "testing",
            "internal_test",
        ]
    )

    test_df = df[
        test_mask
    ].copy()

    if len(test_df) == 0:

        raise RuntimeError(
            f"Fold {fold} has no identifiable test records."
        )

    test_df[
        "fold"
    ] = fold

    test_df[
        "__path"
    ] = test_df[
        path_col
    ].astype(str)

    test_df[
        "__label_raw"
    ] = test_df[
        label_col
    ]

    test_df[
        "__split"
    ] = test_df[
        split_col
    ].astype(str)

    if patient_col:

        test_df[
            "__patient"
        ] = test_df[
            patient_col
        ].astype(str)

    else:

        test_df[
            "__patient"
        ] = ""

    if case_col:

        test_df[
            "__case"
        ] = test_df[
            case_col
        ].astype(str)

    else:

        test_df[
            "__case"
        ] = ""

    fold_frames.append(
        test_df
    )

    fold_reports.append({

        "fold":
            fold,

        "manifest_rows":
            int(len(df)),

        "test_rows":
            int(len(test_df)),

        "path_column":
            path_col,

        "label_column":
            label_col,

        "split_column":
            split_col,

        "patient_column":
            patient_col,

        "case_column":
            case_col,
    })


all_test = pd.concat(
    fold_frames,
    ignore_index=True,
)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):

    text = str(
        value
    ).strip().lower()

    if text in [
        "0",
        "benign",
        "b",
    ]:

        return 0

    if text in [
        "1",
        "malignant",
        "m",
    ]:

        return 1

    try:

        number = int(
            float(
                value
            )
        )

        if number in [
            0,
            1,
        ]:

            return number

    except Exception:

        pass

    raise RuntimeError(
        f"Unsupported BreaKHis label: {value}"
    )


all_test[
    "label"
] = all_test[
    "__label_raw"
].apply(
    normalize_label
)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image(
    value
):

    raw = str(
        value
    ).strip()

    candidates = []

    p = Path(
        raw
    )

    if p.is_absolute():

        candidates.append(
            p
        )

    candidates.append(
        BREAKHIS_ROOT
        /
        raw
    )

    candidates.append(
        ROOT
        /
        raw
    )

    # Also try relative to dataset directory.
    candidates.append(
        ROOT
        /
        "BreaKHis_v1"
        /
        raw
    )

    for candidate in candidates:

        if candidate.is_file():

            return candidate.resolve()

    return None


resolved_paths = []

missing_paths = []

for i, raw in enumerate(
    all_test[
        "__path"
    ]
):

    resolved = resolve_image(
        raw
    )

    if resolved is None:

        missing_paths.append(
            raw
        )

        resolved_paths.append(
            ""
        )

    else:

        resolved_paths.append(
            str(
                resolved
            )
        )

all_test[
    "resolved_path"
] = resolved_paths


if missing_paths:

    raise RuntimeError(
        "Unresolved BreaKHis test image paths: "
        + str(
            len(
                missing_paths
            )
        )
    )


# ============================================================
# TEST-SET INTEGRITY
# ============================================================

unique_test_paths = (
    all_test[
        "resolved_path"
    ]
    .nunique()
)

if unique_test_paths != 7909:

    raise RuntimeError(
        "The pooled five-fold BreaKHis test set does not "
        f"contain exactly 7909 unique images. Found {unique_test_paths}."
    )

duplicate_path_count = (
    len(all_test)
    -
    unique_test_paths
)

if duplicate_path_count != 0:

    raise RuntimeError(
        f"BreaKHis pooled test set contains "
        f"{duplicate_path_count} duplicate image paths."
    )

print()
print(
    "Pooled external test images:",
    len(all_test),
)

print(
    "Unique images:",
    unique_test_paths,
)

print(
    "Label counts:",
    all_test[
        "label"
    ].value_counts()
    .sort_index()
    .to_dict(),
)


# ============================================================
# LOAD FROZEN RESNET-50
# ============================================================

print()
print(
    "Loading frozen CBIS ResNet-50..."
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

state = torch.load(
    RESNET_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

cnn.load_state_dict(
    state,
    strict=True,
)

cnn.fc = nn.Identity()

cnn.eval()
cnn.to(device)


# ============================================================
# EXACT CBIS PREPROCESSING
# ============================================================

def load_breakhis_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    return image


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
# EXTRACT FEATURES
# ============================================================

features = []

paths = all_test[
    "resolved_path"
].tolist()

print()
print(
    "Extracting frozen ResNet features..."
)

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
                load_breakhis_image(
                    path
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

        processed = min(
            start
            +
            len(batch_paths),
            len(paths),
        )

        if (
            processed % 250
            == 0
            or
            processed
            == len(paths)
        ):

            print(
                f"  {processed}/{len(paths)}",
                flush=True,
            )


features = np.concatenate(
    features,
    axis=0,
)


if features.shape != (
    7909,
    2048,
):

    raise RuntimeError(
        f"Unexpected feature shape: {features.shape}"
    )


# ============================================================
# FROZEN PCA
# ============================================================

print()
print(
    "Loading/fitting frozen 34B PCA representation..."
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
            "Training-feature key not identified."
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
        "Frozen training feature dimension is not 2048."
    )

pca = PCA(
    n_components=6
)

pca.fit(
    train_features
)

print(
    "Frozen PCA variance:",
    pca.explained_variance_ratio_,
)

print(
    "Total variance:",
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
    )
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

N_QUBITS = 6
VQC_DEPTH = 2

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

    expectation = float(
        vqc_circuit(
            row
        )
    )

    probability = (
        expectation
        +
        1.0
    ) / 2.0

    vqc_probability.append(
        float(
            np.clip(
                probability,
                EPS,
                1.0 - EPS,
            )
        )
    )

    if (
        i % 250
        == 0
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
# COMMON SOURCE DATA
# ============================================================

all_test[
    "mlp_probability"
] = mlp_probability

all_test[
    "vqc_probability"
] = vqc_probability

all_test[
    "mlp_entropy"
] = entropy(
    mlp_probability
)

all_test[
    "vqc_entropy"
] = entropy(
    vqc_probability
)

all_test[
    "resnet_feature_norm"
] = np.linalg.norm(
    features,
    axis=1,
)

all_test[
    "latent_norm"
] = np.linalg.norm(
    z6,
    axis=1,
)


# ============================================================
# FOLD-WISE RESULTS
# ============================================================

fold_rows = []

for fold in range(
    1,
    6,
):

    subset = all_test[
        all_test[
            "fold"
        ]
        ==
        fold
    ]

    y = subset[
        "label"
    ].to_numpy()

    for model_name, prob_column in [
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
            prob_column
        ].to_numpy()

        metric = calc_metrics(
            y,
            p,
        )

        fold_rows.append({

            "fold":
                fold,

            "model":
                model_name,

            "n":
                len(subset),

            **metric,
        })


fold_results = pd.DataFrame(
    fold_rows
)


# ============================================================
# POOLED RESULTS
# ============================================================

y = all_test[
    "label"
].to_numpy()

pooled_rows = []

for model_name, prob_column in [
    (
        "MLP",
        "mlp_probability",
    ),
    (
        "VQC",
        "vqc_probability",
    ),
]:

    p = all_test[
        prob_column
    ].to_numpy()

    metric = calc_metrics(
        y,
        p,
    )

    pooled_rows.append({

        "scope":
            "pooled_5fold_test",

        "model":
            model_name,

        "n":
            len(y),

        **metric,
    })


pooled_results = pd.DataFrame(
    pooled_rows
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_columns = [

    "fold",
    "__path",
    "resolved_path",
    "__patient",
    "__case",
    "label",
    "mlp_probability",
    "vqc_probability",
    "mlp_entropy",
    "vqc_entropy",
    "resnet_feature_norm",
    "latent_norm",
]

all_test[
    prediction_columns
].to_csv(
    SOURCE_DIR
    / "BREAKHIS_EXTERNAL_PREDICTIONS.csv",
    index=False,
)


# ============================================================
# SAVE TABLES
# ============================================================

fold_results.to_csv(
    TABLE_DIR
    / "TABLE_41B_BREAKHIS_FOLD_RESULTS.csv",
    index=False,
)

pooled_results.to_csv(
    TABLE_DIR
    / "TABLE_41B_BREAKHIS_POOLED_RESULTS.csv",
    index=False,
)


# ============================================================
# FIGURE 1 - FOLD AUROC
# ============================================================

fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for model_name in [
    "MLP",
    "VQC",
]:

    subset = fold_results[
        fold_results[
            "model"
        ]
        ==
        model_name
    ]

    plt.plot(
        subset[
            "fold"
        ],
        subset[
            "roc_auc"
        ],
        marker="o",
        linewidth=1.8,
        label=model_name,
    )

plt.xlabel(
    "BreaKHis fold"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "BreaKHis external fold performance"
)

plt.xticks(
    [
        1,
        2,
        3,
        4,
        5,
    ]
)

plt.ylim(
    0.0,
    1.0,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_FOLD_AUROC.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_FOLD_AUROC.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_FOLD_AUROC_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 - ENTROPY
# ============================================================

entropy_rows = []

for model_name, entropy_column in [
    (
        "MLP",
        "mlp_entropy",
    ),
    (
        "VQC",
        "vqc_entropy",
    ),
]:

    grouped = (
        all_test
        .groupby(
            "fold"
        )[entropy_column]
        .mean()
        .reset_index()
    )

    for _, row in grouped.iterrows():

        entropy_rows.append({

            "fold":
                int(
                    row["fold"]
                ),

            "model":
                model_name,

            "mean_entropy":
                float(
                    row[
                        entropy_column
                    ]
                ),
        })


entropy_df = pd.DataFrame(
    entropy_rows
)

entropy_df.to_csv(
    TABLE_DIR
    / "TABLE_41B_BREAKHIS_ENTROPY_BY_FOLD.csv",
    index=False,
)


fig = plt.figure(
    figsize=(
        7.0,
        5.4,
    )
)

for model_name in [
    "MLP",
    "VQC",
]:

    subset = entropy_df[
        entropy_df[
            "model"
        ]
        ==
        model_name
    ]

    plt.plot(
        subset[
            "fold"
        ],
        subset[
            "mean_entropy"
        ],
        marker="o",
        linewidth=1.8,
        label=model_name,
    )

plt.xlabel(
    "BreaKHis fold"
)

plt.ylabel(
    "Mean predictive entropy"
)

plt.title(
    "BreaKHis external predictive uncertainty"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_ENTROPY.pdf",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_ENTROPY.svg",
    bbox_inches="tight",
)

plt.savefig(
    FIG_DIR
    / "FIGURE_41B_BREAKHIS_ENTROPY_400DPI.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP41B_COMPLETE",

    "dataset":
        "BreaKHis",

    "external_test_images":
        int(
            len(all_test)
        ),

    "unique_images":
        int(
            unique_test_paths
        ),

    "folds":
        5,

    "patient_disjoint_freeze":
        True,

    "case_disjoint_freeze":
        True,

    "model_origin":
        "CBIS-DDSM frozen development pipeline",

    "analysis_role":
        "cross_domain_external_transfer_stress_test",

    "training_performed":
        False,

    "external_adaptation":
        False,

    "cbis_preprocessing_reused":
        True,

    "image_size":
        IMAGE_SIZE,

    "resize_size":
        RESIZE_SIZE,

    "feature_dimension":
        2048,

    "latent_dimension":
        6,

    "fold_results":
        fold_rows,

    "pooled_results":
        pooled_rows,

    "source_files":
        {

            "manifest_directory":
                str(
                    MANIFEST_DIR
                ),

            "resnet_checkpoint":
                str(
                    RESNET_CHECKPOINT
                ),

            "mlp_checkpoint":
                str(
                    MLP_CHECKPOINT
                ),

            "vqc_checkpoint":
                str(
                    VQC_CHECKPOINT
                ),
        },
}


(
    METRIC_DIR
    / "STEP41B_FINAL_RESULTS.json"
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

    "dataset":
        "BreaKHis",

    "folds":
        5,

    "test_images":
        7909,

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
                "ImageNet",

        },

    "frozen_encoder":
        "ResNet-50",

    "frozen_features":
        2048,

    "frozen_pca":
        "training-fitted six-dimensional PCA",

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

    "interpretation":
        "Cross-domain transfer/stress test; not in-domain clinical validation.",
}


(
    CONFIG_DIR
    / "STEP41B_CONFIGURATION.json"
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
print("=" * 100)
print("STEP 41B COMPLETE")
print("=" * 100)

print()
print(
    pooled_results.to_string(
        index=False
    )
)

print()
print(
    "Fold results:",
    TABLE_DIR
    / "TABLE_41B_BREAKHIS_FOLD_RESULTS.csv",
)

print(
    "Pooled results:",
    TABLE_DIR
    / "TABLE_41B_BREAKHIS_POOLED_RESULTS.csv",
)

print(
    "Predictions:",
    SOURCE_DIR
    / "BREAKHIS_EXTERNAL_PREDICTIONS.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print(
    "Results:",
    METRIC_DIR
    / "STEP41B_FINAL_RESULTS.json",
)

print()
print(
    "STATUS: STEP41B_COMPLETE"
)