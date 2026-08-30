from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pydicom

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

import pennylane as qml

warnings.filterwarnings(
    "ignore"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 2026

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

RUN34A = (
    ROOT
    / "experiments"
    / "STEP34A_V2_FINAL_ASUS_PUBLICATION"
)

INPUT_MANIFEST = (
    RUN34A
    / "CBIS_V2_FINAL_PHYSICAL_INPUT_MANIFEST.csv"
)

BACKBONE_CHECKPOINT = (
    RUN34A
    / "checkpoints"
    / "BEST_RESNET50_STATE_DICT.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP34B_FINAL_MATCHED_MLP_VQC"
)

FEATURE_DIR = OUT / "features"
LATENT_DIR = OUT / "latent"
CHECKPOINT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
FIG_DIR = OUT / "figures"
SOURCE_DIR = FIG_DIR / "source_data"
TABLE_DIR = OUT / "tables"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    FEATURE_DIR,
    LATENT_DIR,
    CHECKPOINT_DIR,
    PRED_DIR,
    METRIC_DIR,
    FIG_DIR,
    SOURCE_DIR,
    TABLE_DIR,
    CONFIG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


IMAGE_SIZE = 512
BATCH_SIZE_CNN = 8

HEAD_BATCH_SIZE = 64

MAX_EPOCHS_MLP = 50
MAX_EPOCHS_VQC = 30

PATIENCE = 7

LR_MLP = 0.01
LR_VQC = 0.02

WEIGHT_DECAY = 1e-4

LATENT_DIM = 6

N_QUBITS = 6
VQC_DEPTH = 2

EPS = 1e-6


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(
    SEED
)


# ============================================================
# HELPERS
# ============================================================

def clean(x):

    if pd.isna(x):
        return ""

    return str(
        x
    ).strip()


def sha256_file(
    path
):

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


def compute_metrics(
    y,
    p,
):

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

    cm = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        tn
        /
        max(
            tn + fp,
            1,
        )
    )

    return {

        "n":
            int(
                len(y)
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "sensitivity":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

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

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


def patient_aggregate(
    frame
):

    rows = []

    for patient, group in frame.groupby(
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

            "record_count":
                int(
                    len(group)
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# DICOM PREPROCESSING
# ============================================================

def dicom_to_rgb(
    path
):

    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    arr = ds.pixel_array.astype(
        np.float32
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
        * 255.0
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
            576,
            576,
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


class ImageFrameDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        frame,
    ):

        self.frame = (
            frame
            .reset_index(drop=True)
        )

    def __len__(
        self
    ):

        return len(
            self.frame
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.frame.iloc[
            idx
        ]

        path = Path(
            row[
                "resolved_full_mammogram_dicom"
            ]
        )

        image = dicom_to_rgb(
            path
        )

        image = EVAL_TRANSFORM(
            image
        )

        return (
            image,
            int(
                row[
                    "binary_label"
                ]
            ),
            str(
                row[
                    "patient_id"
                ]
            ),
            str(
                path
            ),
        )


# ============================================================
# LOAD INPUT MANIFEST
# ============================================================

print()
print("=" * 90)
print(
    "LOADING FROZEN 34A-v2 INPUT MANIFEST"
)
print("=" * 90)

if not INPUT_MANIFEST.is_file():

    raise RuntimeError(
        f"Input manifest not found: {INPUT_MANIFEST}"
    )

if not BACKBONE_CHECKPOINT.is_file():

    raise RuntimeError(
        f"34A checkpoint not found: {BACKBONE_CHECKPOINT}"
    )

frame = pd.read_csv(
    INPUT_MANIFEST
)

required = {
    "patient_id",
    "experimental_split",
    "binary_label",
    "resolved_full_mammogram_dicom",
}

missing = (
    required
    -
    set(frame.columns)
)

if missing:

    raise RuntimeError(
        "Missing input-manifest columns: "
        + str(
            sorted(
                missing
            )
        )
    )

print()
print(
    "Rows:",
    len(frame)
)

print(
    "Split counts:",
    frame[
        "experimental_split"
    ].value_counts().to_dict()
)


# ============================================================
# PATIENT-LEVEL SPLIT CHECK
# ============================================================

split_patient_sets = {}

for split in [
    "train",
    "calibration",
    "internal_test",
]:

    split_patient_sets[
        split
    ] = set(
        frame[
            frame[
                "experimental_split"
            ]
            == split
        ][
            "patient_id"
        ]
        .astype(str)
    )


overlap = {

    "train_calibration":
        len(
            split_patient_sets[
                "train"
            ]
            &
            split_patient_sets[
                "calibration"
            ]
        ),

    "train_internal_test":
        len(
            split_patient_sets[
                "train"
            ]
            &
            split_patient_sets[
                "internal_test"
            ]
        ),

    "calibration_internal_test":
        len(
            split_patient_sets[
                "calibration"
            ]
            &
            split_patient_sets[
                "internal_test"
            ]
        ),
}

if any(
    overlap.values()
):

    raise RuntimeError(
        f"Patient leakage detected: {overlap}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_frame = frame[
    frame[
        "experimental_split"
    ]
    == "train"
].copy()

cal_frame = frame[
    frame[
        "experimental_split"
    ]
    == "calibration"
].copy()

test_frame = frame[
    frame[
        "experimental_split"
    ]
    == "internal_test"
].copy()


train_loader_cnn = DataLoader(
    ImageFrameDataset(
        train_frame
    ),
    batch_size=BATCH_SIZE_CNN,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

cal_loader_cnn = DataLoader(
    ImageFrameDataset(
        cal_frame
    ),
    batch_size=BATCH_SIZE_CNN,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader_cnn = DataLoader(
    ImageFrameDataset(
        test_frame
    ),
    batch_size=BATCH_SIZE_CNN,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# LOAD 34A RESNET-50 AS FROZEN FEATURE EXTRACTOR
# ============================================================

print()
print("=" * 90)
print(
    "LOADING FROZEN 34A-v2 RESNET-50"
)
print("=" * 90)

cnn_device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

cnn = models.resnet50(
    weights=None
)

# The 34A-v2 checkpoint is a binary classifier:
#     fc.weight = [1, 2048]
#     fc.bias   = [1]
#
# Therefore construct the same architecture BEFORE loading,
# then remove the classifier for 2048-D feature extraction.

cnn.fc = nn.Linear(
    cnn.fc.in_features,
    1,
)

state = torch.load(
    BACKBONE_CHECKPOINT,
    map_location="cpu",
)

missing_keys, unexpected_keys = (
    cnn.load_state_dict(
        state,
        strict=True,
    )
)

# Remove classifier AFTER loading.
cnn.fc = nn.Identity()

cnn = cnn.to(
    cnn_device
)

cnn.eval()

for p in cnn.parameters():

    p.requires_grad = False


print(
    "Feature device:",
    cnn_device
)

print(
    "Missing keys:",
    missing_keys
)

print(
    "Unexpected keys:",
    unexpected_keys
)


# ============================================================
# EXTRACT 2048-D FEATURES
# ============================================================

@torch.no_grad()
def extract_features(
    loader,
    name,
):

    features = []
    labels = []
    patient_ids = []
    paths = []

    print()
    print(
        f"Extracting {name} features..."
    )

    for n, (
        x,
        y,
        pids,
        path_batch,
    ) in enumerate(
        loader,
        1,
    ):

        x = x.to(
            cnn_device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=cnn_device.type,
            dtype=torch.float16,
            enabled=(
                cnn_device.type
                == "cuda"
            ),
        ):

            z = cnn(
                x
            )

        features.append(
            z.float()
            .cpu()
        )

        labels.extend(
            y.numpy().tolist()
        )

        patient_ids.extend(
            list(
                pids
            )
        )

        paths.extend(
            list(
                path_batch
            )
        )

        if n % 25 == 0:

            print(
                f"  batches: {n}",
                flush=True,
            )

    return (
        torch.cat(
            features,
            dim=0,
        ),
        torch.tensor(
            labels,
            dtype=torch.float32,
        ),
        patient_ids,
        paths,
    )


train_features, train_y, train_pid, train_paths = (
    extract_features(
        train_loader_cnn,
        "train",
    )
)

cal_features, cal_y, cal_pid, cal_paths = (
    extract_features(
        cal_loader_cnn,
        "calibration",
    )
)

test_features, test_y, test_pid, test_paths = (
    extract_features(
        test_loader_cnn,
        "internal_test",
    )
)


print()
print(
    "Train features:",
    tuple(
        train_features.shape
    )
)

print(
    "Calibration features:",
    tuple(
        cal_features.shape
    )
)

print(
    "Internal-test features:",
    tuple(
        test_features.shape
    )
)


# ============================================================
# SAVE 2048-D FEATURES
# ============================================================

torch.save(
    {
        "features":
            train_features,

        "labels":
            train_y,

        "patient_id":
            train_pid,

        "paths":
            train_paths,

        "source":
            str(
                BACKBONE_CHECKPOINT
            ),
    },
    FEATURE_DIR
    / "TRAIN_RESNET50_2048_FEATURES.pt",
)

torch.save(
    {
        "features":
            cal_features,

        "labels":
            cal_y,

        "patient_id":
            cal_pid,

        "paths":
            cal_paths,
    },
    FEATURE_DIR
    / "CALIBRATION_RESNET50_2048_FEATURES.pt",
)

torch.save(
    {
        "features":
            test_features,

        "labels":
            test_y,

        "patient_id":
            test_pid,

        "paths":
            test_paths,
    },
    FEATURE_DIR
    / "TEST_RESNET50_2048_FEATURES.pt",
)


# ============================================================
# PCA FIT ONLY ON TRAIN
# ============================================================

print()
print("=" * 90)
print(
    "FITTING PCA -> 6-D LATENT"
)
print("=" * 90)

pca = PCA(
    n_components=LATENT_DIM,
    random_state=SEED,
)

train_X_np = (
    train_features
    .numpy()
)

cal_X_np = (
    cal_features
    .numpy()
)

test_X_np = (
    test_features
    .numpy()
)

train_z_np = pca.fit_transform(
    train_X_np
)

cal_z_np = pca.transform(
    cal_X_np
)

test_z_np = pca.transform(
    test_X_np
)

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    float(
        pca.explained_variance_ratio_.sum()
    )
)


train_z = torch.tensor(
    train_z_np,
    dtype=torch.float32,
)

cal_z = torch.tensor(
    cal_z_np,
    dtype=torch.float32,
)

test_z = torch.tensor(
    test_z_np,
    dtype=torch.float32,
)


torch.save(
    {
        "train_z":
            train_z,

        "train_y":
            train_y,

        "train_patient_id":
            train_pid,

        "calibration_z":
            cal_z,

        "calibration_y":
            cal_y,

        "calibration_patient_id":
            cal_pid,

        "internal_test_z":
            test_z,

        "internal_test_y":
            test_y,

        "internal_test_patient_id":
            test_pid,

        "latent_dim":
            LATENT_DIM,

        "pca_components":
            torch.tensor(
                pca.components_,
                dtype=torch.float32,
            ),

        "pca_mean":
            torch.tensor(
                pca.mean_,
                dtype=torch.float32,
            ),

        "explained_variance_ratio":
            pca.explained_variance_ratio_,

        "seed":
            SEED,

    },
    LATENT_DIR
    / "SHARED_6D_LATENTS.pt",
)


pd.DataFrame(
    train_z_np,
    columns=[
        f"z{i+1}"
        for i in range(LATENT_DIM)
    ],
).assign(
    label=train_y.numpy(),
    patient_id=train_pid,
).to_csv(
    LATENT_DIR
    / "TRAIN_6D_LATENTS.csv",
    index=False,
)

pd.DataFrame(
    cal_z_np,
    columns=[
        f"z{i+1}"
        for i in range(LATENT_DIM)
    ],
).assign(
    label=cal_y.numpy(),
    patient_id=cal_pid,
).to_csv(
    LATENT_DIR
    / "CALIBRATION_6D_LATENTS.csv",
    index=False,
)

pd.DataFrame(
    test_z_np,
    columns=[
        f"z{i+1}"
        for i in range(LATENT_DIM)
    ],
).assign(
    label=test_y.numpy(),
    patient_id=test_pid,
).to_csv(
    LATENT_DIR
    / "TEST_6D_LATENTS.csv",
    index=False,
)


# ============================================================
# LATENT LOADERS
# ============================================================

train_latent_loader = DataLoader(
    TensorDataset(
        train_z,
        train_y,
    ),
    batch_size=HEAD_BATCH_SIZE,
    shuffle=True,
)

cal_latent_loader = DataLoader(
    TensorDataset(
        cal_z,
        cal_y,
    ),
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False,
)

test_latent_loader = DataLoader(
    TensorDataset(
        test_z,
        test_y,
    ),
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MATCHED CLASSICAL MLP
# ============================================================

class MatchedMLP(
    nn.Module
):

    def __init__(
        self
    ):

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
        x
    ):

        return self.network(
            x
        ).squeeze(
            -1
        )


mlp = MatchedMLP()

mlp_parameters = sum(
    p.numel()
    for p in mlp.parameters()
    if p.requires_grad
)

if mlp_parameters != 25:

    raise RuntimeError(
        f"Expected 25 MLP parameters, got {mlp_parameters}"
    )

print()
print(
    "=" * 90
)

print(
    "MATCHED CLASSICAL MLP"
)

print(
    "Trainable parameters:",
    mlp_parameters
)


# ============================================================
# VQC
# ============================================================

dev = qml.device(
    "lightning.qubit",
    wires=N_QUBITS,
)


@qml.qnode(
    dev,
    interface="torch",
    diff_method="adjoint",
)
def vqc_circuit(
    inputs,
    theta,
):

    # --------------------------------------------------------
    # Non-trainable input encoding
    # --------------------------------------------------------

    for q in range(
        N_QUBITS
    ):

        qml.RY(
            inputs[q],
            wires=q,
        )

        qml.RZ(
            inputs[q],
            wires=q,
        )

    # --------------------------------------------------------
    # Two trainable RY/RZ layers
    # theta shape = (2, 6, 2)
    # exact trainable parameters = 24
    # --------------------------------------------------------

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
                    0
                ],
                wires=q,
            )

            qml.RZ(
                theta[
                    layer,
                    q,
                    1
                ],
                wires=q,
            )

        # ring topology
        for q in range(
            N_QUBITS
        ):

            qml.CNOT(
                wires=[
                    q,
                    (
                        q + 1
                    )
                    %
                    N_QUBITS,
                ]
            )

    return qml.expval(
        qml.PauliZ(
            0
        )
    )


class VQCHead(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.theta = nn.Parameter(
            0.05
            *
            torch.randn(
                VQC_DEPTH,
                N_QUBITS,
                2,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x
    ):

        outputs = []

        for i in range(
            x.shape[0]
        ):

            f = vqc_circuit(
                x[i],
                self.theta,
            )

            p = (
                f
                + 1.0
            ) / 2.0

            p = torch.clamp(
                p,
                EPS,
                1.0 - EPS,
            )

            outputs.append(
                p
            )

        return torch.stack(
            outputs
        )


vqc = VQCHead()

vqc_parameters = sum(
    p.numel()
    for p in vqc.parameters()
    if p.requires_grad
)

if vqc_parameters != 24:

    raise RuntimeError(
        f"Expected 24 VQC parameters, got {vqc_parameters}"
    )

print()
print(
    "=" * 90
)

print(
    "6-QUBIT VQC"
)

print(
    "Qubits:",
    N_QUBITS
)

print(
    "Depth:",
    VQC_DEPTH
)

print(
    "Trainable parameters:",
    vqc_parameters
)


# ============================================================
# GENERIC MLP TRAINING
# ============================================================

def predict_mlp(
    model,
    features
):

    model.eval()

    probs = []

    with torch.no_grad():

        for start in range(
            0,
            len(features),
            HEAD_BATCH_SIZE,
        ):

            x = features[
                start:
                start
                + HEAD_BATCH_SIZE
            ]

            logits = model(
                x
            )

            p = torch.sigmoid(
                logits
            )

            probs.extend(
                p.numpy().tolist()
            )

    return np.asarray(
        probs,
        dtype=float,
    )


def train_mlp(
    model
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_MLP,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    history = []
    bad = 0

    for epoch in range(
        1,
        MAX_EPOCHS_MLP + 1,
    ):

        model.train()

        total_loss = 0.0
        total_n = 0

        for x, y in train_latent_loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                float(
                    loss.detach()
                )
                *
                len(y)
            )

            total_n += len(y)

        p_cal = predict_mlp(
            model,
            cal_z,
        )

        auc = roc_auc_score(
            cal_y.numpy(),
            p_cal,
        )

        mean_loss = (
            total_loss
            /
            max(
                total_n,
                1,
            )
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                mean_loss,

            "calibration_roc_auc":
                float(auc),

        })

        print(
            f"MLP epoch {epoch:02d}/{MAX_EPOCHS_MLP} "
            f"| loss={mean_loss:.6f} "
            f"| cal ROC-AUC={auc:.6f}",
            flush=True,
        )

        if auc > best_auc:

            best_auc = auc

            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .clone()
                for k, v
                in model.state_dict().items()
            }

            bad = 0

        else:

            bad += 1

            if bad >= PATIENCE:
                break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_auc,
        best_epoch,
        history,
    )


# ============================================================
# VQC TRAINING
# ============================================================

def predict_vqc(
    model,
    features
):

    model.eval()

    probs = []

    # Small batches because each item invokes a tiny circuit.
    batch_size = 32

    with torch.no_grad():

        for start in range(
            0,
            len(features),
            batch_size,
        ):

            x = features[
                start:
                start
                + batch_size
            ]

            p = model(
                x
            )

            probs.extend(
                p.cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        probs,
        dtype=float,
    )


def train_vqc(
    model
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_VQC,
    )

    best_auc = -np.inf
    best_epoch = -1
    best_theta = None
    history = []
    bad = 0

    for epoch in range(
        1,
        MAX_EPOCHS_VQC + 1,
    ):

        model.train()

        total_loss = 0.0
        total_n = 0

        # ----------------------------------------------------
        # Deterministic mini-batches
        # ----------------------------------------------------

        perm = torch.randperm(
            len(train_z)
        )

        for start in range(
            0,
            len(train_z),
            32,
        ):

            idx = perm[
                start:
                start + 32
            ]

            x = train_z[
                idx
            ]

            y = train_y[
                idx
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            p = model(
                x
            )

            loss = -torch.mean(
                y
                *
                torch.log(
                    torch.clamp(
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
                torch.log(
                    torch.clamp(
                        1.0 - p,
                        EPS,
                        1.0,
                    )
                )
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                float(
                    loss.detach()
                )
                *
                len(y)
            )

            total_n += len(y)

        p_cal = predict_vqc(
            model,
            cal_z,
        )

        auc = roc_auc_score(
            cal_y.numpy(),
            p_cal,
        )

        mean_loss = (
            total_loss
            /
            max(
                total_n,
                1,
            )
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                mean_loss,

            "calibration_roc_auc":
                float(auc),

        })

        print(
            f"VQC epoch {epoch:02d}/{MAX_EPOCHS_VQC} "
            f"| loss={mean_loss:.6f} "
            f"| cal ROC-AUC={auc:.6f}",
            flush=True,
        )

        if auc > best_auc:

            best_auc = auc

            best_epoch = epoch

            best_theta = (
                model.theta
                .detach()
                .clone()
            )

            bad = 0

        else:

            bad += 1

            if bad >= PATIENCE:
                break

    if best_theta is None:

        raise RuntimeError(
            "VQC did not produce a valid checkpoint."
        )

    with torch.no_grad():

        model.theta.copy_(
            best_theta
        )

    return (
        model,
        best_auc,
        best_epoch,
        history,
    )


# ============================================================
# TRAIN MLP
# ============================================================

print()
print(
    "=" * 90
)

print(
    "TRAINING MATCHED 25-PARAMETER MLP"
)

print(
    "=" * 90
)

mlp, mlp_cal_auc, mlp_best_epoch, mlp_history = (
    train_mlp(
        mlp
    )
)

MLP_CHECKPOINT = (
    CHECKPOINT_DIR
    / "MATCHED_MLP_25PARAM_BEST.pt"
)

torch.save(
    mlp.state_dict(),
    MLP_CHECKPOINT,
)


# ============================================================
# TRAIN VQC
# ============================================================

print()
print(
    "=" * 90
)

print(
    "TRAINING 6-QUBIT DEPTH-2 VQC"
)

print(
    "=" * 90
)

vqc, vqc_cal_auc, vqc_best_epoch, vqc_history = (
    train_vqc(
        vqc
    )
)

VQC_CHECKPOINT = (
    CHECKPOINT_DIR
    / "VQC_6Q_DEPTH2_24PARAM_BEST.pt"
)

torch.save(
    vqc.state_dict(),
    VQC_CHECKPOINT,
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

p_mlp_test = predict_mlp(
    mlp,
    test_z,
)

p_vqc_test = predict_vqc(
    vqc,
    test_z,
)


# ============================================================
# RECORD-LEVEL RESULTS
# ============================================================

mlp_test_metrics = compute_metrics(
    test_y.numpy(),
    p_mlp_test,
)

vqc_test_metrics = compute_metrics(
    test_y.numpy(),
    p_vqc_test,
)


# ============================================================
# PATIENT-LEVEL RESULTS
# ============================================================

mlp_record_df = pd.DataFrame({

    "patient_id":
        test_pid,

    "label":
        test_y.numpy().astype(int),

    "probability":
        p_mlp_test,

})

mlp_patient_df = (
    patient_aggregate(
        mlp_record_df
    )
)


vqc_record_df = pd.DataFrame({

    "patient_id":
        test_pid,

    "label":
        test_y.numpy().astype(int),

    "probability":
        p_vqc_test,

})

vqc_patient_df = (
    patient_aggregate(
        vqc_record_df
    )
)


mlp_patient_metrics = compute_metrics(
    mlp_patient_df[
        "label"
    ].to_numpy(),
    mlp_patient_df[
        "probability"
    ].to_numpy(),
)

vqc_patient_metrics = compute_metrics(
    vqc_patient_df[
        "label"
    ].to_numpy(),
    vqc_patient_df[
        "probability"
    ].to_numpy(),
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

mlp_record_df.to_csv(
    PRED_DIR
    / "MLP_RECORD_LEVEL_PREDICTIONS.csv",
    index=False,
)

mlp_patient_df.to_csv(
    PRED_DIR
    / "MLP_PATIENT_LEVEL_PREDICTIONS.csv",
    index=False,
)

vqc_record_df.to_csv(
    PRED_DIR
    / "VQC_RECORD_LEVEL_PREDICTIONS.csv",
    index=False,
)

vqc_patient_df.to_csv(
    PRED_DIR
    / "VQC_PATIENT_LEVEL_PREDICTIONS.csv",
    index=False,
)


combined_patient = (
    mlp_patient_df[
        [
            "patient_id",
            "label",
            "probability",
        ]
    ]
    .rename(
        columns={
            "probability":
                "mlp_probability"
        }
    )
    .merge(
        vqc_patient_df[
            [
                "patient_id",
                "probability",
            ]
        ].rename(
            columns={
                "probability":
                    "vqc_probability"
            }
        ),
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
)

combined_patient.to_csv(
    PRED_DIR
    / "PATIENT_LEVEL_MATCHED_MLP_VQC.csv",
    index=False,
)


# ============================================================
# RESULT TABLE
# ============================================================

results_table = pd.DataFrame([

    {

        "model":
            "Matched_MLP",

        "trainable_parameters":
            mlp_parameters,

        "best_calibration_auc":
            mlp_cal_auc,

        "best_epoch":
            mlp_best_epoch,

        "record_test_auc":
            mlp_test_metrics[
                "roc_auc"
            ],

        "record_test_auprc":
            mlp_test_metrics[
                "auprc"
            ],

        "record_test_accuracy":
            mlp_test_metrics[
                "accuracy"
            ],

        "patient_test_auc":
            mlp_patient_metrics[
                "roc_auc"
            ],

        "patient_test_auprc":
            mlp_patient_metrics[
                "auprc"
            ],

        "patient_test_accuracy":
            mlp_patient_metrics[
                "accuracy"
            ],

    },

    {

        "model":
            "VQC",

        "trainable_parameters":
            vqc_parameters,

        "best_calibration_auc":
            vqc_cal_auc,

        "best_epoch":
            vqc_best_epoch,

        "record_test_auc":
            vqc_test_metrics[
                "roc_auc"
            ],

        "record_test_auprc":
            vqc_test_metrics[
                "auprc"
            ],

        "record_test_accuracy":
            vqc_test_metrics[
                "accuracy"
            ],

        "patient_test_auc":
            vqc_patient_metrics[
                "roc_auc"
            ],

        "patient_test_auprc":
            vqc_patient_metrics[
                "auprc"
            ],

        "patient_test_accuracy":
            vqc_patient_metrics[
                "accuracy"
            ],

    },

])

results_table.to_csv(
    TABLE_DIR
    / "TABLE_34B_MATCHED_MLP_VQC.csv",
    index=False,
)


# ============================================================
# LATENT EXPLAINED VARIANCE TABLE
# ============================================================

pd.DataFrame({

    "component":
        [
            i + 1
            for i
            in range(
                LATENT_DIM
            )
        ],

    "explained_variance_ratio":
        pca.explained_variance_ratio_,

}).to_csv(
    TABLE_DIR
    / "TABLE_34B_PCA_VARIANCE.csv",
    index=False,
)


# ============================================================
# SIMPLE MODEL DIFFERENCE
# ============================================================

delta = {

    "patient_auc_vqc_minus_mlp":
        float(
            vqc_patient_metrics[
                "roc_auc"
            ]
            -
            mlp_patient_metrics[
                "roc_auc"
            ]
        ),

    "patient_auprc_vqc_minus_mlp":
        float(
            vqc_patient_metrics[
                "auprc"
            ]
            -
            mlp_patient_metrics[
                "auprc"
            ]
        ),

    "record_auc_vqc_minus_mlp":
        float(
            vqc_test_metrics[
                "roc_auc"
            ]
            -
            mlp_test_metrics[
                "roc_auc"
            ]
        ),

    "record_auprc_vqc_minus_mlp":
        float(
            vqc_test_metrics[
                "auprc"
            ]
            -
            mlp_test_metrics[
                "auprc"
            ]
        ),
}


# ============================================================
# CONFIGURATION
# ============================================================

configuration = {

    "experiment":
        "STEP34B_MATCHED_MLP_VQC",

    "seed":
        SEED,

    "source_checkpoint":
        str(
            BACKBONE_CHECKPOINT
        ),

    "source_checkpoint_sha256":
        sha256_file(
            BACKBONE_CHECKPOINT
        ),

    "input_manifest":
        str(
            INPUT_MANIFEST
        ),

    "input_manifest_sha256":
        sha256_file(
            INPUT_MANIFEST
        ),

    "records":
        int(len(frame)),

    "train_records":
        int(len(train_frame)),

    "calibration_records":
        int(len(cal_frame)),

    "internal_test_records":
        int(len(test_frame)),

    "train_patients":
        int(
            len(
                split_patient_sets[
                    "train"
                ]
            )
        ),

    "calibration_patients":
        int(
            len(
                split_patient_sets[
                    "calibration"
                ]
            )
        ),

    "internal_test_patients":
        int(
            len(
                split_patient_sets[
                    "internal_test"
                ]
            )
        ),

    "patient_overlap":
        overlap,

    "backbone":
        "34A-v2 frozen ImageNet-initialized ResNet-50",

    "backbone_feature_dimension":
        2048,

    "latent_dimension":
        LATENT_DIM,

    "latent_method":
        "PCA fit on training features only",

    "mlp":
        {
            "architecture":
                "Linear(6->3)+tanh+Linear(3->1)",

            "parameters":
                mlp_parameters,

            "expected_parameters":
                25,
        },

    "vqc":
        {
            "qubits":
                N_QUBITS,

            "depth":
                VQC_DEPTH,

            "encoding":
                "non-trainable RY/RZ angle encoding",

            "ansatz":
                "two trainable RY/RZ layers",

            "entanglement":
                "ring CNOT",

            "parameters":
                vqc_parameters,

            "expected_parameters":
                24,

            "output":
                "Pauli-Z expectation on q0",

            "probability":
                "(<Z0>+1)/2 clipped to epsilon bounds",

            "finite_shot_training":
                False,
        },

    "status":
        "FROZEN_DETERMINISTIC_34B",
}


(
    CONFIG_DIR
    / "STEP34B_CONFIGURATION.json"
).write_text(
    json.dumps(
        configuration,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# METRICS JSON
# ============================================================

metrics_json = {

    "status":
        "STEP34B_COMPLETE",

    "matched_parameter_counts":
        {
            "mlp":
                mlp_parameters,

            "vqc":
                vqc_parameters,
        },

    "calibration":
        {
            "mlp_auc":
                float(
                    mlp_cal_auc
                ),

            "mlp_best_epoch":
                int(
                    mlp_best_epoch
                ),

            "vqc_auc":
                float(
                    vqc_cal_auc
                ),

            "vqc_best_epoch":
                int(
                    vqc_best_epoch
                ),
        },

    "record_level":
        {
            "mlp":
                mlp_test_metrics,

            "vqc":
                vqc_test_metrics,
        },

    "patient_level":
        {
            "mlp":
                mlp_patient_metrics,

            "vqc":
                vqc_patient_metrics,
        },

    "delta_vqc_minus_mlp":
        delta,

    "pca_explained_variance_ratio":
        pca.explained_variance_ratio_.tolist(),

    "pca_total_explained_variance":
        float(
            pca.explained_variance_ratio_.sum()
        ),
}


(
    METRIC_DIR
    / "STEP34B_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        metrics_json,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# TRAINING HISTORY
# ============================================================

pd.DataFrame(
    mlp_history
).to_csv(
    METRIC_DIR
    / "MLP_TRAINING_HISTORY.csv",
    index=False,
)

pd.DataFrame(
    vqc_history
).to_csv(
    METRIC_DIR
    / "VQC_TRAINING_HISTORY.csv",
    index=False,
)


# ============================================================
# SOURCE DATA FOR COMPARISON
# ============================================================

comparison_source = pd.DataFrame({

    "patient_id":
        combined_patient[
            "patient_id"
        ],

    "label":
        combined_patient[
            "label"
        ],

    "mlp_probability":
        combined_patient[
            "mlp_probability"
        ],

    "vqc_probability":
        combined_patient[
            "vqc_probability"
        ],

})

comparison_source.to_csv(
    SOURCE_DIR
    / "PATIENT_MATCHED_COMPARISON_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# CONFUSION SOURCE
# ============================================================

mlp_patient_pred = (
    combined_patient[
        "mlp_probability"
    ]
    .to_numpy()
    >= 0.5
).astype(int)

vqc_patient_pred = (
    combined_patient[
        "vqc_probability"
    ]
    .to_numpy()
    >= 0.5
).astype(int)

y_patient = (
    combined_patient[
        "label"
    ]
    .to_numpy()
)

mlp_cm = confusion_matrix(
    y_patient,
    mlp_patient_pred,
    labels=[
        0,
        1,
    ],
)

vqc_cm = confusion_matrix(
    y_patient,
    vqc_patient_pred,
    labels=[
        0,
        1,
    ],
)

pd.DataFrame({

    "true_class":
        [
            "benign",
            "malignant",
        ],

    "mlp_true_negative_or_positive_counts":
        [
            int(
                mlp_cm[0, 0]
            ),
            int(
                mlp_cm[1, 1]
            ),
        ],

    "vqc_true_negative_or_positive_counts":
        [
            int(
                vqc_cm[0, 0]
            ),
            int(
                vqc_cm[1, 1]
            ),
        ],

}).to_csv(
    SOURCE_DIR
    / "PATIENT_CONFUSION_SOURCE_DATA.csv",
    index=False,
)


# ============================================================
# REPRODUCTION README
# ============================================================

readme = f"""
STEP 34B Ã¢â‚¬â€ MATCHED CLASSICAL MLP VS 6-QUBIT VQC

Dataset:
CBIS-DDSM ASUS-local v2 cohort

Canonical/available input:
3401 selected records from the 34A-v2 physical input manifest.

Frozen split:
train={len(train_frame)}
calibration={len(cal_frame)}
internal_test={len(test_frame)}

Patient counts:
train={len(split_patient_sets['train'])}
calibration={len(split_patient_sets['calibration'])}
internal_test={len(split_patient_sets['internal_test'])}

Patient overlap:
{overlap}

Backbone:
Frozen 34A-v2 ResNet-50

Backbone feature dimension:
2048

Compact shared representation:
PCA to 6 dimensions

PCA fitting:
TRAIN ONLY

Classical head:
Linear(6->3)+tanh+Linear(3->1)

Classical trainable parameters:
25

Quantum head:
6 qubits
2 trainable RY/RZ ansatz layers
ring CNOT entanglement

Quantum trainable parameters:
24

Quantum input encoding:
non-trainable RY/RZ angle encoding

Quantum output:
Pauli-Z expectation on q0

Quantum probability mapping:
(<Z0> + 1) / 2

Finite-shot noise during deterministic training:
No

Model selection:
best calibration ROC-AUC

Seed:
{SEED}

Primary output:
matched deterministic classical-versus-quantum comparison

Next:
shot uncertainty
parameter perturbation / epistemic-style analysis
calibration
conformal prediction
selective prediction
"""

(
    OUT
    / "README_REPRODUCTION.txt"
).write_text(
    readme,
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

    "torch":
        torch.__version__,

    "torchvision":
        __import__(
            "torchvision"
        ).__version__,

    "pennylane":
        qml.__version__,

    "cuda_available":
        bool(
            torch.cuda.is_available()
        ),

    "cuda_version":
        torch.version.cuda,

    "gpu":
        (
            torch.cuda.get_device_name(
                0
            )
            if torch.cuda.is_available()
            else "CPU"
        ),

    "quantum_backend":
        "lightning.qubit",
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

checksum_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if (
        path.name
        ==
        "SHA256_INVENTORY.csv"
    ):
        continue

    checksum_rows.append({

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
    checksum_rows
).to_csv(
    OUT
    / "SHA256_INVENTORY.csv",
    index=False,
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 100)
print("STEP 34B COMPLETE")
print("=" * 100)

print()
print(
    "MLP parameters:",
    mlp_parameters
)

print(
    "VQC parameters:",
    vqc_parameters
)

print()
print(
    "MLP best calibration ROC-AUC:",
    mlp_cal_auc
)

print(
    "VQC best calibration ROC-AUC:",
    vqc_cal_auc
)

print()
print(
    "MLP record-level test ROC-AUC:",
    mlp_test_metrics[
        "roc_auc"
    ]
)

print(
    "VQC record-level test ROC-AUC:",
    vqc_test_metrics[
        "roc_auc"
    ]
)

print()
print(
    "MLP patient-level test ROC-AUC:",
    mlp_patient_metrics[
        "roc_auc"
    ]
)

print(
    "VQC patient-level test ROC-AUC:",
    vqc_patient_metrics[
        "roc_auc"
    ]
)

print()
print(
    "VQC - MLP patient-level ROC-AUC:",
    delta[
        "patient_auc_vqc_minus_mlp"
    ]
)

print()
print(
    "MLP checkpoint:",
    MLP_CHECKPOINT
)

print(
    "VQC checkpoint:",
    VQC_CHECKPOINT
)

print(
    "Shared latent:",
    LATENT_DIR
    / "SHARED_6D_LATENTS.pt"
)

print(
    "Predictions:",
    PRED_DIR
)

print(
    "Metrics:",
    METRIC_DIR
    / "STEP34B_FINAL_RESULTS.json"
)

print()
print(
    "STATUS: STEP34B_COMPLETE"
)
