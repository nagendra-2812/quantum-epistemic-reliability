from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import random

import numpy as np
import pandas as pd
import pydicom

from PIL import Image, ImageFilter, ImageEnhance

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
# CONFIGURATION
# ============================================================

SEED = 2026

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    ROOT
    / "experiments"
    / "STEP34A_V2_FINAL_ASUS_PUBLICATION"
    / "CBIS_V2_FINAL_PHYSICAL_INPUT_MANIFEST.csv"
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

TRAIN_FEATURES = (
    RUN34B
    / "features"
    / "TRAIN_RESNET50_2048_FEATURES.pt"
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
    / "STEP39_CORRUPTION_ROBUSTNESS"
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

N_QUBITS = 6
VQC_DEPTH = 2

BATCH_SIZE = 16

SEVERITIES = [
    1,
    2,
    3,
]

CORRUPTIONS = [
    "gaussian_noise",
    "gaussian_blur",
    "contrast_reduction",
    "brightness_shift",
    "resolution_reduction",
]

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
        *
        np.log(
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
            count
            /
            total
            *
            abs(
                accuracy
                -
                confidence
            )
        )

    return float(
        value
    )


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


def calculate_metrics(
    y,
    p,
):

    prediction = (
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
                    prediction
                    ==
                    y
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
# FROZEN DICOM PREPROCESSING
# EXACTLY MATCHES 34B
# ============================================================

def dicom_to_rgb(path):

    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    arr = (
        ds.pixel_array
        .astype(
            np.float32
        )
    )

    if arr.ndim != 2:

        raise RuntimeError(
            f"Expected 2-D DICOM: {path}"
        )

    finite = arr[
        np.isfinite(arr)
    ]

    if finite.size == 0:

        raise RuntimeError(
            f"No finite pixels: {path}"
        )

    lo, hi = np.percentile(
        finite,
        [
            1.0,
            99.0,
        ],
    )

    if hi <= lo:
        hi = lo + 1.0

    arr = np.clip(
        arr,
        lo,
        hi,
    )

    arr = (
        (
            arr - lo
        )
        /
        (
            hi - lo
        )
        *
        255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    rgb = np.stack(
        [
            arr,
            arr,
            arr,
        ],
        axis=-1,
    )

    return Image.fromarray(
        rgb
    )


EVAL_TRANSFORM = transforms.Compose([

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
# CORRUPTION FUNCTIONS
# ============================================================

def corrupt_image(
    image,
    corruption,
    severity,
):

    image = image.copy()

    if corruption == "gaussian_noise":

        arr = np.asarray(
            image
        ).astype(
            np.float32
        )

        sigma = {
            1: 8.0,
            2: 16.0,
            3: 32.0,
        }[
            severity
        ]

        rng = np.random.default_rng(
            SEED
            + severity * 100
        )

        noise = rng.normal(
            0.0,
            sigma,
            size=arr.shape,
        )

        arr = np.clip(
            arr + noise,
            0,
            255,
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            arr
        )


    if corruption == "gaussian_blur":

        radius = {
            1: 1.0,
            2: 2.0,
            3: 4.0,
        }[
            severity
        ]

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )


    if corruption == "contrast_reduction":

        factor = {
            1: 0.75,
            2: 0.50,
            3: 0.25,
        }[
            severity
        ]

        enhancer = (
            ImageEnhance.Contrast(
                image
            )
        )

        return enhancer.enhance(
            factor
        )


    if corruption == "brightness_shift":

        factor = {
            1: 0.85,
            2: 0.70,
            3: 0.55,
        }[
            severity
        ]

        enhancer = (
            ImageEnhance.Brightness(
                image
            )
        )

        return enhancer.enhance(
            factor
        )


    if corruption == "resolution_reduction":

        factor = {
            1: 0.75,
            2: 0.50,
            3: 0.25,
        }[
            severity
        ]

        width, height = (
            image.size
        )

        small = image.resize(
            (
                max(
                    32,
                    int(
                        width
                        *
                        factor
                    )
                ),
                max(
                    32,
                    int(
                        height
                        *
                        factor
                    )
                ),
            ),
            Image.Resampling.BILINEAR,
        )

        return small.resize(
            (
                width,
                height,
            ),
            Image.Resampling.BILINEAR,
        )


    raise ValueError(
        f"Unknown corruption: {corruption}"
    )


# ============================================================
# LOAD MANIFEST
# ============================================================

print()
print("=" * 100)
print("STEP 39 - CORRUPTION / DISTRIBUTION-SHIFT ROBUSTNESS")
print("=" * 100)

if not MANIFEST.is_file():

    raise RuntimeError(
        f"Manifest not found: {MANIFEST}"
    )

manifest = pd.read_csv(
    MANIFEST
)

test_manifest = manifest[
    manifest[
        "experimental_split"
    ]
    ==
    "internal_test"
].copy()

if len(
    test_manifest
) != 578:

    raise RuntimeError(
        "Expected exactly 578 internal-test records."
        f" Found {len(test_manifest)}."
    )

required_columns = [
    "patient_id",
    "binary_label",
    "resolved_full_mammogram_dicom",
    "input_status",
]

missing = [
    c for c in required_columns
    if c not in test_manifest.columns
]

if missing:

    raise RuntimeError(
        f"Manifest missing columns: {missing}"
    )

if not np.all(
    test_manifest[
        "input_status"
    ]
    ==
    "VERIFIED"
):

    raise RuntimeError(
        "Not all internal-test inputs are VERIFIED."
    )

print()
print(
    "Internal-test records:",
    len(
        test_manifest
    ),
)

print(
    "Internal-test patients:",
    test_manifest[
        "patient_id"
    ].nunique()
)


# ============================================================
# CHECK PHYSICAL INPUTS
# ============================================================

for i, row in test_manifest.iterrows():

    path = Path(
        row[
            "resolved_full_mammogram_dicom"
        ]
    )

    if not path.is_file():

        raise RuntimeError(
            f"Missing DICOM: {path}"
        )

print(
    "All test DICOM paths verified."
)


# ============================================================
# LOAD RESNET-50
# ============================================================

print()
print(
    "Loading frozen 34A-v2 ResNet-50..."
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
    RUN34B
    / ".."
    / "STEP34A_V2_FINAL_ASUS_PUBLICATION"
    / "checkpoints"
    / "BEST_RESNET50_STATE_DICT.pt",
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
# LOAD PCA FROM SAVED 34B REPRESENTATION
# ============================================================

print()
print(
    "Loading frozen 34B PCA..."
)

train_feature_data = torch.load(
    TRAIN_FEATURES,
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
            "Could not identify training features."
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


if train_features.ndim != 2:

    raise RuntimeError(
        f"Unexpected train feature shape: "
        f"{train_features.shape}"
    )

if train_features.shape[1] != 2048:

    raise RuntimeError(
        "Expected 2048-D ResNet features."
    )


pca = PCA(
    n_components=6,
)

pca.fit(
    train_features
)

print(
    "PCA explained variance:",
    pca.explained_variance_ratio_,
)

print(
    "PCA total explained variance:",
    float(
        pca.explained_variance_ratio_.sum()
    ),
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
    vqc_device,
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


def vqc_predict(
    z_np
):

    values = []

    for row in z_np:

        e = float(
            vqc_circuit(
                row
            )
        )

        p = (
            e + 1.0
        ) / 2.0

        values.append(
            float(
                np.clip(
                    p,
                    EPS,
                    1.0 - EPS,
                )
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    rows,
    corruption=None,
    severity=0,
):

    tensors = []

    for n, (_, row) in enumerate(
        rows.iterrows(),
        1,
    ):

        path = Path(
            row[
                "resolved_full_mammogram_dicom"
            ]
        )

        image = dicom_to_rgb(
            path
        )

        if corruption is not None:

            image = corrupt_image(
                image,
                corruption,
                severity,
            )

        tensor = EVAL_TRANSFORM(
            image
        )

        tensors.append(
            tensor
        )

        if n % 100 == 0:

            print(
                f"  prepared {n}/{len(rows)}",
                flush=True,
            )

    features = []

    with torch.no_grad():

        for start in range(
            0,
            len(tensors),
            BATCH_SIZE,
        ):

            batch = torch.stack(
                tensors[
                    start:
                    start + BATCH_SIZE
                ]
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

    return np.concatenate(
        features,
        axis=0,
    )


# ============================================================
# MAIN ROBUSTNESS RUN
# ============================================================

y = (
    test_manifest[
        "binary_label"
    ]
    .astype(int)
    .to_numpy()
)

record_index = np.arange(
    len(test_manifest)
)

patient_ids = (
    test_manifest[
        "patient_id"
    ]
    .astype(str)
    .to_numpy()
)


conditions = [

    (
        "clean",
        0,
    )

]

for corruption in CORRUPTIONS:

    for severity in SEVERITIES:

        conditions.append(
            (
                corruption,
                severity,
            )
        )


results_rows = []
source_rows = []

clean_mlp = None
clean_vqc = None


for condition, severity in conditions:

    if condition == "clean":

        corruption = None

    else:

        corruption = condition


    print()
    print("-" * 100)

    print(
        "Condition:",
        condition,
    )

    print(
        "Severity:",
        severity,
    )


    features2048 = extract_features(
        test_manifest,
        corruption,
        severity,
    )


    z6 = pca.transform(
        features2048
    )


    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    vqc_probability = (
        vqc_predict(
            z6
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlp_metrics = calculate_metrics(
        y,
        mlp_probability,
    )

    vqc_metrics = calculate_metrics(
        y,
        vqc_probability,
    )


    if condition == "clean":

        clean_mlp = mlp_metrics
        clean_vqc = vqc_metrics


    for model_name, probabilities, model_metrics in [

        (
            "MLP",
            mlp_probability,
            mlp_metrics,
        ),

        (
            "VQC",
            vqc_probability,
            vqc_metrics,
        ),

    ]:

        reference = (
            clean_mlp
            if model_name == "MLP"
            else clean_vqc
        )

        results_rows.append({

            "model":
                model_name,

            "condition":
                condition,

            "severity":
                int(
                    severity
                ),

            **model_metrics,

            "roc_auc_delta":
                float(
                    model_metrics[
                        "roc_auc"
                    ]
                    -
                    reference[
                        "roc_auc"
                    ]
                ),

            "auprc_delta":
                float(
                    model_metrics[
                        "auprc"
                    ]
                    -
                    reference[
                        "auprc"
                    ]
                ),

            "brier_delta":
                float(
                    model_metrics[
                        "brier"
                    ]
                    -
                    reference[
                        "brier"
                    ]
                ),

            "nll_delta":
                float(
                    model_metrics[
                        "nll"
                    ]
                    -
                    reference[
                        "nll"
                    ]
                ),

            "ece_delta":
                float(
                    model_metrics[
                        "ece_10bin"
                    ]
                    -
                    reference[
                        "ece_10bin"
                    ]
                ),

            "entropy_delta":
                float(
                    model_metrics[
                        "mean_entropy"
                    ]
                    -
                    reference[
                        "mean_entropy"
                    ]
                ),
        })


        for i in range(
            len(y)
        ):

            source_rows.append({

                "record_index":
                    int(
                        record_index[i]
                    ),

                "patient_id":
                    patient_ids[i],

                "label":
                    int(
                        y[i]
                    ),

                "model":
                    model_name,

                "condition":
                    condition,

                "severity":
                    int(
                        severity
                    ),

                "probability":
                    float(
                        probabilities[i]
                    ),

                "entropy":
                    float(
                        entropy(
                            [
                                probabilities[i]
                            ]
                        )[0]
                    ),

                "feature_norm_2048":
                    float(
                        np.linalg.norm(
                            features2048[i]
                        )
                    ),

                "latent_norm_6":
                    float(
                        np.linalg.norm(
                            z6[i]
                        )
                    ),
            })


results_df = pd.DataFrame(
    results_rows
)

source_df = pd.DataFrame(
    source_rows
)


# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

patient_rows = []

for (
    model_name,
    condition,
    severity,
) in (
    results_df[
        [
            "model",
            "condition",
            "severity",
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None,
    )
):

    temp = source_df[
        (
            source_df[
                "model"
            ]
            ==
            model_name
        )
        &
        (
            source_df[
                "condition"
            ]
            ==
            condition
        )
        &
        (
            source_df[
                "severity"
            ]
            ==
            severity
        )
    ]


    patient = (
        temp
        .groupby(
            "patient_id"
        )
        .agg(
            label=(
                "label",
                "first",
            ),
            probability=(
                "probability",
                "mean",
            ),
            entropy=(
                "entropy",
                "mean",
            ),
        )
        .reset_index()
    )


    patient_rows.append({

        "model":
            model_name,

        "condition":
            condition,

        "severity":
            severity,

        "patient_count":
            len(patient),

        "roc_auc":
            safe_auc(
                patient[
                    "label"
                ].to_numpy(),
                patient[
                    "probability"
                ].to_numpy(),
            ),

        "auprc":
            safe_auprc(
                patient[
                    "label"
                ].to_numpy(),
                patient[
                    "probability"
                ].to_numpy(),
            ),

        "brier":
            brier(
                patient[
                    "label"
                ].to_numpy(),
                patient[
                    "probability"
                ].to_numpy(),
            ),

        "nll":
            nll(
                patient[
                    "label"
                ].to_numpy(),
                patient[
                    "probability"
                ].to_numpy(),
            ),

        "ece_10bin":
            ece(
                patient[
                    "label"
                ].to_numpy(),
                patient[
                    "probability"
                ].to_numpy(),
            ),

        "mean_entropy":
            float(
                patient[
                    "entropy"
                ].mean()
            ),
    })


patient_df = pd.DataFrame(
    patient_rows
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    TABLE_DIR
    / "TABLE_39_CORRUPTION_ROBUSTNESS.csv",
    index=False,
)

patient_df.to_csv(
    TABLE_DIR
    / "TABLE_39_PATIENT_LEVEL_ROBUSTNESS.csv",
    index=False,
)

source_df.to_csv(
    SOURCE_DIR
    / "CORRUPTION_ROBUSTNESS_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# FIGURE 1 - ROC-AUC VS SEVERITY
# ============================================================

for model_name in [
    "MLP",
    "VQC",
]:

    fig = plt.figure(
        figsize=(
            7.0,
            5.4,
        )
    )

    for corruption in [
        "gaussian_noise",
        "gaussian_blur",
        "contrast_reduction",
        "brightness_shift",
        "resolution_reduction",
    ]:

        subset = results_df[
            (
                results_df[
                    "model"
                ]
                ==
                model_name
            )
            &
            (
                results_df[
                    "condition"
                ]
                ==
                corruption
            )
        ]

        plt.plot(
            subset[
                "severity"
            ],
            subset[
                "roc_auc"
            ],
            marker="o",
            linewidth=1.6,
            label=corruption,
        )

    plt.xlabel(
        "Corruption severity"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        f"{model_name} robustness to image corruption"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.legend(
        fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ROCAUC.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ROCAUC.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ROCAUC_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 2 - ENTROPY VS SEVERITY
# ============================================================

for model_name in [
    "MLP",
    "VQC",
]:

    fig = plt.figure(
        figsize=(
            7.0,
            5.4,
        )
    )

    for corruption in CORRUPTIONS:

        subset = results_df[
            (
                results_df[
                    "model"
                ]
                ==
                model_name
            )
            &
            (
                results_df[
                    "condition"
                ]
                ==
                corruption
            )
        ]

        plt.plot(
            subset[
                "severity"
            ],
            subset[
                "mean_entropy"
            ],
            marker="o",
            linewidth=1.6,
            label=corruption,
        )

    plt.xlabel(
        "Corruption severity"
    )

    plt.ylabel(
        "Mean predictive entropy"
    )

    plt.title(
        f"{model_name} uncertainty under image corruption"
    )

    plt.legend(
        fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ENTROPY.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ENTROPY.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / f"FIGURE_39_{model_name}_ENTROPY_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP39_COMPLETE",

    "dataset":
        "CBIS-DDSM",

    "test_records":
        int(
            len(
                test_manifest
            )
        ),

    "test_patients":
        int(
            test_manifest[
                "patient_id"
            ].nunique()
        ),

    "image_size":
        IMAGE_SIZE,

    "resize_size":
        RESIZE_SIZE,

    "corruptions":
        CORRUPTIONS,

    "severities":
        SEVERITIES,

    "models":
        [
            "MLP",
            "VQC",
        ],

    "resnet_features":
        "2048-D",

    "pca_dimensions":
        6,

    "pca_fit":
        "training features only",

    "original_data_modified":
        False,

    "model_retraining":
        False,

    "results":
        results_rows,

    "artifacts":
        {

            "record_table":
                str(
                    TABLE_DIR
                    / "TABLE_39_CORRUPTION_ROBUSTNESS.csv"
                ),

            "patient_table":
                str(
                    TABLE_DIR
                    / "TABLE_39_PATIENT_LEVEL_ROBUSTNESS.csv"
                ),

            "source":
                str(
                    SOURCE_DIR
                    / "CORRUPTION_ROBUSTNESS_SOURCE_DATA.csv"
                ),

            "figures":
                str(
                    FIG_DIR
                ),
        },
}


(
    METRIC_DIR
    / "STEP39_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CONFIGURATION
# ============================================================

config = {

    "seed":
        SEED,

    "image_size":
        IMAGE_SIZE,

    "resize_size":
        RESIZE_SIZE,

    "dicom_percentile_clip":
        [
            1.0,
            99.0,
        ],

    "imagenet_mean":
        [
            0.485,
            0.456,
            0.406,
        ],

    "imagenet_std":
        [
            0.229,
            0.224,
            0.225,
        ],

    "corruptions":
        CORRUPTIONS,

    "severities":
        SEVERITIES,

    "resnet_features":
        2048,

    "pca_dimensions":
        6,

    "pca_fit_split":
        "train",

    "inference_only":
        True,

    "source_manifest":
        str(
            MANIFEST
        ),

    "source_vqc_checkpoint":
        str(
            VQC_CHECKPOINT
        ),

    "source_mlp_checkpoint":
        str(
            MLP_CHECKPOINT
        ),
}


(
    CONFIG_DIR
    / "STEP39_CONFIGURATION.json"
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
print("=" * 100)
print("STEP 39 COMPLETE")
print("=" * 100)

print()
print(
    "Record-level results:",
    TABLE_DIR
    / "TABLE_39_CORRUPTION_ROBUSTNESS.csv",
)

print(
    "Patient-level results:",
    TABLE_DIR
    / "TABLE_39_PATIENT_LEVEL_ROBUSTNESS.csv",
)

print(
    "Source data:",
    SOURCE_DIR
    / "CORRUPTION_ROBUSTNESS_SOURCE_DATA.csv",
)

print(
    "Figures:",
    FIG_DIR,
)

print(
    "Results:",
    METRIC_DIR
    / "STEP39_FINAL_RESULTS.json",
)

print()
print(
    "STATUS: STEP39_COMPLETE"
)