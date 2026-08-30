from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import hashlib
import random
import platform
import time
import warnings

import numpy as np
import pandas as pd

from PIL import Image

import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 2026

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP34A_V2_ASUS_CBIS_RESNET50_PUBLICATION"
)

CHECKPOINT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
SPLIT_DIR = OUT / "splits"
FIG_DIR = OUT / "figures"
FIG_DATA_DIR = FIG_DIR / "source_data"
TABLE_DIR = OUT / "tables"
CONFIG_DIR = OUT / "configuration"
EVIDENCE_DIR = OUT / "evidence"

for d in [
    OUT,
    CHECKPOINT_DIR,
    PRED_DIR,
    METRIC_DIR,
    SPLIT_DIR,
    FIG_DIR,
    FIG_DATA_DIR,
    TABLE_DIR,
    CONFIG_DIR,
    EVIDENCE_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

if DEVICE.type != "cuda":

    raise RuntimeError(
        "Publication Step 34A requires the ASUS CUDA GPU, "
        "but CUDA is not available."
    )


# Frozen publication contract
BATCH_SIZE = 16
EPOCHS_MAX = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
IMAGE_SIZE = 512
NUM_WORKERS = 0

# ------------------------------------------------------------
# The canonical manifest's source_table determines the
# physical root.
# ------------------------------------------------------------

SOURCE_ROOTS = {
    "calc_train":
        ROOT / "calc_train",

    "calc_test":
        ROOT / "calc_test",

    "mass_train":
        ROOT / "mass_train",

    "mass_test":
        ROOT / "mass_test",
}


# ============================================================
# SEED
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(1024 * 1024),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


# ============================================================
# RESOLVE FULL MAMMOGRAM PATH
# ============================================================

def resolve_full_mammogram(row):

    source_table = str(
        row[
            "source_table"
        ]
    ).strip()

    relative = str(
        row[
            "image_file_path_metadata"
        ]
    ).strip()

    if not relative:
        return None

    source_root = (
        SOURCE_ROOTS.get(
            source_table
        )
    )

    if source_root is None:
        return None

    candidate = (
        source_root
        / Path(
            relative
        )
    )

    if candidate.is_file():
        return candidate

    return None


# ============================================================
# READ DICOM
# ============================================================

def dicom_to_rgb(path):

    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    pixels = ds.pixel_array.astype(
        np.float32
    )

    if pixels.ndim != 2:

        raise RuntimeError(
            f"Expected 2-D mammogram pixel array: {path}"
        )

    finite = pixels[
        np.isfinite(
            pixels
        )
    ]

    if finite.size == 0:

        raise RuntimeError(
            f"No finite pixel values: {path}"
        )

    low, high = np.percentile(
        finite,
        [
            1.0,
            99.0,
        ],
    )

    if high <= low:

        high = low + 1.0

    pixels = np.clip(
        pixels,
        low,
        high,
    )

    pixels = (
        (
            pixels
            - low
        )
        /
        (
            high
            - low
        )
    )

    pixels = (
        pixels
        * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    rgb = np.stack(
        [
            pixels,
            pixels,
            pixels,
        ],
        axis=-1,
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


# ============================================================
# DATASET
# ============================================================

class CBISDataset(
    Dataset
):

    def __init__(
        self,
        frame,
        transform,
    ):

        self.frame = (
            frame
            .reset_index(drop=True)
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.frame
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.frame.iloc[
            idx
        ]

        image = dicom_to_rgb(
            Path(
                row[
                    "resolved_image_path"
                ]
            )
        )

        image = self.transform(
            image
        )

        label = torch.tensor(
            float(
                row[
                    "binary_label"
                ]
            ),
            dtype=torch.float32,
        )

        return (
            image,
            label,
            row[
                "patient_id"
            ],
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (
            576,
            576,
        )
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(
            0.90,
            1.0,
        ),
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        7
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


eval_transform = transforms.Compose([

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


# ============================================================
# METRICS
# ============================================================

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
            int(len(y)),

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

        "tp":
            int(tp),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),
    }


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    loader,
):

    model.eval()

    y_all = []
    p_all = []
    patient_all = []

    with torch.no_grad():

        for (
            x,
            y,
            patient_ids,
        ) in loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            logits = (
                model(x)
                .squeeze(1)
            )

            probs = torch.sigmoid(
                logits
            )

            y_all.extend(
                y.cpu()
                .numpy()
                .tolist()
            )

            p_all.extend(
                probs.cpu()
                .numpy()
                .tolist()
            )

            patient_all.extend(
                list(
                    patient_ids
                )
            )

    return (
        np.asarray(
            y_all,
            dtype=int,
        ),
        np.asarray(
            p_all,
            dtype=float,
        ),
        patient_all,
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_auc(
    y,
    p,
    seed=2026,
    n_bootstrap=2000,
):

    rng = np.random.default_rng(
        seed
    )

    observed = roc_auc_score(
        y,
        p,
    )

    values = []

    n = len(y)

    for _ in range(
        n_bootstrap
    ):

        idx = rng.integers(
            0,
            n,
            n,
        )

        yy = y[
            idx
        ]

        pp = p[
            idx
        ]

        if len(
            np.unique(
                yy
            )
        ) < 2:
            continue

        values.append(
            roc_auc_score(
                yy,
                pp,
            )
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    lower, upper = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "estimate":
            float(observed),

        "lower_95":
            float(lower),

        "upper_95":
            float(upper),

        "n_bootstrap_valid":
            int(len(values)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    started = time.time()

    print()
    print("=" * 100)
    print("STEP 34A-v2 — LOCAL ASUS CBIS-DDSM RESNET-50")
    print("=" * 100)

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not MANIFEST.is_file():

        raise RuntimeError(
            f"Manifest not found: {MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    print()
    print(
        "Manifest:",
        MANIFEST,
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Manifest SHA256:",
        sha256_file(
            MANIFEST
        ),
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = {
        "patient_id",
        "experimental_split",
        "binary_label",
        "source_table",
        "image_file_path_metadata",
    }

    missing_columns = (
        required
        -
        set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Missing columns: "
            + str(
                sorted(
                    missing_columns
                )
            )
        )

    # --------------------------------------------------------
    # Split normalization
    # --------------------------------------------------------

    df[
        "patient_id"
    ] = (
        df[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "experimental_split"
    ] = (
        df[
            "experimental_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df[
        "binary_label"
    ] = (
        df[
            "binary_label"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # Physical path resolution
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPLETE PHYSICAL PATH RESOLUTION")
    print("=" * 100)

    resolved = []
    failed_rows = []

    for idx, row in df.iterrows():

        path = resolve_full_mammogram(
            row
        )

        if path is None:

            resolved.append(
                ""
            )

            failed_rows.append(
                int(idx)
            )

        else:

            resolved.append(
                str(path)
            )

    df[
        "resolved_image_path"
    ] = resolved

    print()
    print(
        "Total manifest records:",
        len(df),
    )

    print(
        "Resolved:",
        len(df)
        -
        len(failed_rows),
    )

    print(
        "Unresolved:",
        len(failed_rows),
    )

    if failed_rows:

        failures = (
            df.iloc[
                failed_rows
            ].copy()
        )

        failure_file = (
            EVIDENCE_DIR
            / "UNRESOLVED_FULL_MAMMOGRAM_PATHS.csv"
        )

        failures.to_csv(
            failure_file,
            index=False,
        )

        print()
        print(
            "First unresolved records:"
        )

        for idx in failed_rows[:20]:

            row = df.iloc[
                idx
            ]

            print()
            print(
                "source_table:",
                row[
                    "source_table"
                ],
            )

            print(
                "metadata path:",
                row[
                    "image_file_path_metadata"
                ],
            )

        raise RuntimeError(
            "NOT ALL FULL MAMMOGRAM PATHS RESOLVED. "
            f"See {failure_file}"
        )

    # --------------------------------------------------------
    # DICOM readability preflight
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("DICOM READABILITY PREFLIGHT")
    print("=" * 100)

    read_failures = []

    for i, path_string in enumerate(
        df[
            "resolved_image_path"
        ],
        1,
    ):

        path = Path(
            path_string
        )

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=False,
                force=True,
            )

            arr = ds.pixel_array

            if arr.ndim != 2:

                raise RuntimeError(
                    "DICOM pixel data not 2-D"
                )

        except Exception as exc:

            read_failures.append({
                "row":
                    int(i - 1),

                "path":
                    str(path),

                "error":
                    repr(exc),
            })

        if i % 250 == 0:

            print(
                f"Checked {i}/{len(df)} DICOM files...",
                flush=True,
            )

    print()
    print(
        "DICOM readability failures:",
        len(read_failures),
    )

    if read_failures:

        failure_file = (
            EVIDENCE_DIR
            / "DICOM_READ_FAILURES.csv"
        )

        pd.DataFrame(
            read_failures
        ).to_csv(
            failure_file,
            index=False,
        )

        raise RuntimeError(
            "DICOM readability preflight failed. "
            f"See {failure_file}"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train = df[
        df[
            "experimental_split"
        ]
        == "train"
    ].copy()

    calibration = df[
        df[
            "experimental_split"
        ]
        == "calibration"
    ].copy()

    test = df[
        df[
            "experimental_split"
        ]
        == "internal_test"
    ].copy()

    train_patients = set(
        train[
            "patient_id"
        ]
    )

    calibration_patients = set(
        calibration[
            "patient_id"
        ]
    )

    test_patients = set(
        test[
            "patient_id"
        ]
    )

    overlaps = {
        "train_calibration":
            sorted(
                train_patients
                &
                calibration_patients
            ),

        "train_test":
            sorted(
                train_patients
                &
                test_patients
            ),

        "calibration_test":
            sorted(
                calibration_patients
                &
                test_patients
            ),
    }

    if any(
        len(v)
        for v in overlaps.values()
    ):

        raise RuntimeError(
            "Patient leakage detected."
        )

    print()
    print("=" * 100)
    print("FROZEN LOCAL EXPERIMENT SPLIT")
    print("=" * 100)

    print(
        "Train records:",
        len(train),
    )

    print(
        "Calibration records:",
        len(calibration),
    )

    print(
        "Internal-test records:",
        len(test),
    )

    print(
        "Train patients:",
        len(train_patients),
    )

    print(
        "Calibration patients:",
        len(calibration_patients),
    )

    print(
        "Internal-test patients:",
        len(test_patients),
    )

    print(
        "Patient overlap:",
        {
            k:
                len(v)
            for k, v
            in overlaps.items()
        },
    )

    train.to_csv(
        SPLIT_DIR
        / "TRAIN_RECORDS.csv",
        index=False,
    )

    calibration.to_csv(
        SPLIT_DIR
        / "CALIBRATION_RECORDS.csv",
        index=False,
    )

    test.to_csv(
        SPLIT_DIR
        / "INTERNAL_TEST_RECORDS.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_ds = CBISDataset(
        train,
        train_transform,
    )

    calibration_ds = CBISDataset(
        calibration,
        eval_transform,
    )

    test_ds = CBISDataset(
        test,
        eval_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    calibration_loader = DataLoader(
        calibration_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # ResNet-50
    # --------------------------------------------------------

    weights = (
        models.ResNet50_Weights.DEFAULT
    )

    model = models.resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    model = model.to(
        DEVICE
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_MAX,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_auc = -np.inf
    best_epoch = -1
    best_state = None

    print()
    print("=" * 100)
    print("RESNET-50 TRAINING")
    print("=" * 100)

    start_training = time.time()

    for epoch in range(
        1,
        EPOCHS_MAX + 1,
    ):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for (
            x,
            y,
            _,
        ) in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                logits = (
                    model(x)
                    .squeeze(1)
                )

                loss = criterion(
                    logits,
                    y,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                float(
                    loss.detach()
                )
                * len(y)
            )

            n_seen += len(y)

        scheduler.step()

        # ----------------------------------------------------
        # Calibration evaluation
        # ----------------------------------------------------

        y_cal, p_cal, _ = predict(
            model,
            calibration_loader,
        )

        cal_metrics = compute_metrics(
            y_cal,
            p_cal,
        )

        mean_loss = (
            running_loss
            /
            max(
                n_seen,
                1,
            )
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                float(mean_loss),

            "calibration_roc_auc":
                cal_metrics[
                    "roc_auc"
                ],

            "calibration_auprc":
                cal_metrics[
                    "auprc"
                ],

            "learning_rate":
                float(
                    current_lr
                ),
        })

        print(
            f"epoch {epoch:02d}/{EPOCHS_MAX} "
            f"| loss={mean_loss:.6f} "
            f"| calibration ROC-AUC="
            f"{cal_metrics['roc_auc']:.6f} "
            f"| lr={current_lr:.7f}",
            flush=True,
        )

        if (
            cal_metrics[
                "roc_auc"
            ]
            >
            best_auc
        ):

            best_auc = (
                cal_metrics[
                    "roc_auc"
                ]
            )

            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for (
                    k,
                    v
                )
                in model.state_dict().items()
            }

    training_seconds = (
        time.time()
        -
        start_training
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    BEST = (
        CHECKPOINT_DIR
        / "BEST_RESNET50_STATE_DICT.pt"
    )

    LAST = (
        CHECKPOINT_DIR
        / "LAST_RESNET50_STATE_DICT.pt"
    )

    torch.save(
        best_state,
        BEST,
    )

    torch.save(
        model.state_dict(),
        LAST,
    )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Final predictions
    # --------------------------------------------------------

    y_cal, p_cal, cal_ids = predict(
        model,
        calibration_loader,
    )

    y_test, p_test, test_ids = predict(
        model,
        test_loader,
    )

    cal_metrics = compute_metrics(
        y_cal,
        p_cal,
    )

    test_metrics = compute_metrics(
        y_test,
        p_test,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pd.DataFrame({

        "split":
            "calibration",

        "patient_id":
            cal_ids,

        "label":
            y_cal,

        "probability":
            p_cal,

    }).to_csv(
        PRED_DIR
        / "CALIBRATION_PREDICTIONS.csv",
        index=False,
    )

    pd.DataFrame({

        "split":
            "internal_test",

        "patient_id":
            test_ids,

        "label":
            y_test,

        "probability":
            p_test,

    }).to_csv(
        PRED_DIR
        / "INTERNAL_TEST_PREDICTIONS.csv",
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        METRIC_DIR
        / "TRAINING_HISTORY.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap = {
        "roc_auc":
            bootstrap_auc(
                y_test,
                p_test,
                seed=SEED,
                n_bootstrap=2000,
            )
    }

    # --------------------------------------------------------
    # ROC source
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test,
        p_test,
    )

    pd.DataFrame({

        "fpr":
            fpr,

        "tpr":
            tpr,

        "threshold":
            thresholds,

    }).to_csv(
        FIG_DATA_DIR
        / "ROC_SOURCE_DATA.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PR source
    # --------------------------------------------------------

    precision, recall, thresholds_pr = (
        precision_recall_curve(
            y_test,
            p_test,
        )
    )

    pd.DataFrame({

        "precision":
            precision,

        "recall":
            recall,

    }).to_csv(
        FIG_DATA_DIR
        / "PR_SOURCE_DATA.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion source
    # --------------------------------------------------------

    pred_test = (
        p_test
        >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        pred_test,
        labels=[
            0,
            1,
        ],
    )

    pd.DataFrame(
        cm,
        index=[
            "True_Benign",
            "True_Malignant",
        ],
        columns=[
            "Pred_Benign",
            "Pred_Malignant",
        ],
    ).to_csv(
        FIG_DATA_DIR
        / "CONFUSION_MATRIX_SOURCE_DATA.csv"
    )

    # --------------------------------------------------------
    # Primary table
    # --------------------------------------------------------

    pd.DataFrame([{

        "Dataset":
            "CBIS-DDSM",

        "Model":
            "ResNet-50",

        "ROC-AUC":
            test_metrics[
                "roc_auc"
            ],

        "AUPRC":
            test_metrics[
                "auprc"
            ],

        "Accuracy":
            test_metrics[
                "accuracy"
            ],

        "Sensitivity":
            test_metrics[
                "sensitivity"
            ],

        "Specificity":
            test_metrics[
                "specificity"
            ],

        "Balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],

        "F1":
            test_metrics[
                "f1"
            ],

        "Brier":
            test_metrics[
                "brier"
            ],

    }]).to_csv(
        TABLE_DIR
        / "TABLE_01_CBIS_RESNET50.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    # FIGURE 1: ROC
    fig = plt.figure(
        figsize=(
            6.5,
            5.2,
        )
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2.0,
    )

    plt.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "CBIS-DDSM ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_01_ROC.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_01_ROC.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_01_ROC_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    # FIGURE 2: PR
    fig = plt.figure(
        figsize=(
            6.5,
            5.2,
        )
    )

    plt.plot(
        recall,
        precision,
        linewidth=2.0,
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "CBIS-DDSM Precision–Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_02_PR.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_02_PR.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_02_PR_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    # FIGURE 3: Training
    hist = pd.DataFrame(
        history
    )

    fig = plt.figure(
        figsize=(
            6.5,
            5.2,
        )
    )

    plt.plot(
        hist["epoch"],
        hist["train_loss"],
        marker="o",
        linewidth=1.8,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training BCE loss"
    )

    plt.title(
        "CBIS-DDSM Training History"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_03_TRAINING_HISTORY.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_03_TRAINING_HISTORY.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_03_TRAINING_HISTORY_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    # FIGURE 4: confusion
    fig = plt.figure(
        figsize=(
            5.8,
            5.2,
        )
    )

    plt.imshow(
        cm,
    )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Benign",
            "Malignant",
        ],
    )

    plt.yticks(
        [
            0,
            1,
        ],
        [
            "Benign",
            "Malignant",
        ],
    )

    for i in range(
        2
    ):

        for j in range(
            2
        ):

            plt.text(
                j,
                i,
                str(
                    cm[
                        i,
                        j
                    ]
                ),
                ha="center",
                va="center",
                fontsize=13,
            )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )

    plt.title(
        "CBIS-DDSM Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "FIGURE_04_CONFUSION_MATRIX.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_04_CONFUSION_MATRIX.svg",
        bbox_inches="tight",
    )

    plt.savefig(
        FIG_DIR
        / "FIGURE_04_CONFUSION_MATRIX_400DPI.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics_json = {

        "step":
            "34A-v2",

        "status":
            "STEP34A_V2_COMPLETE",

        "dataset":
            "CBIS-DDSM",

        "unit_of_analysis":
            "record/abnormality with patient-disjoint split",

        "model":
            "ImageNet-pretrained ResNet-50",

        "classifier":
            "single linear output layer",

        "loss":
            "BCEWithLogitsLoss",

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            "CosineAnnealingLR",

        "gradient_clipping":
            GRAD_CLIP,

        "mixed_precision":
            True,

        "seed":
            SEED,

        "input_resolution":
            "512x512",

        "train_records":
            len(train),

        "calibration_records":
            len(calibration),

        "internal_test_records":
            len(test),

        "train_patients":
            len(train_patients),

        "calibration_patients":
            len(calibration_patients),

        "internal_test_patients":
            len(test_patients),

        "patient_overlap":
            {
                k:
                    len(v)
                for k, v
                in overlaps.items()
            },

        "best_epoch":
            int(best_epoch),

        "best_calibration_roc_auc":
            float(best_auc),

        "calibration_metrics":
            cal_metrics,

        "internal_test_metrics":
            test_metrics,

        "bootstrap_roc_auc_95":
            bootstrap[
                "roc_auc"
            ],

        "training_seconds":
            float(
                training_seconds
            ),

        "manifest":
            str(MANIFEST),

        "manifest_sha256":
            sha256_file(
                MANIFEST
            ),

        "best_checkpoint":
            str(BEST),

        "last_checkpoint":
            str(LAST),

        "status_note":
            (
                "ASUS local publication v2. "
                "The historical Kaggle Step-34A artifacts "
                "were unavailable locally and are not being "
                "represented as reproduced byte-for-byte."
            ),
    }

    (
        METRIC_DIR
        / "FINAL_METRICS.json"
    ).write_text(
        json.dumps(
            metrics_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    configuration = {

        "step":
            "34A-v2",

        "dataset":
            "CBIS-DDSM",

        "manifest":
            str(MANIFEST),

        "manifest_sha256":
            sha256_file(
                MANIFEST
            ),

        "model":
            "ResNet-50",

        "pretrained":
            "ImageNet",

        "fine_tuning":
            "full-network",

        "input_resolution":
            "512x512",

        "source_object":
            "resolved full mammogram DICOM",

        "mask_input":
            False,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "epochs_max":
            EPOCHS_MAX,

        "scheduler":
            "CosineAnnealingLR",

        "gradient_clipping":
            GRAD_CLIP,

        "mixed_precision":
            True,

        "class_weighting":
            "disabled",

        "seed":
            SEED,

        "patient_disjoint":
            True,
    }

    (
        CONFIG_DIR
        / "RESOLVED_CONFIGURATION.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = {

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "machine":
            platform.machine(),

        "processor":
            platform.processor(),

        "torch":
            torch.__version__,

        "torchvision":
            __import__(
                "torchvision"
            ).__version__,

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),

        "cuda_version":
            torch.version.cuda,

        "gpu":
            torch.cuda.get_device_name(
                0
            ),
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

    # --------------------------------------------------------
    # SHA256 inventory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme = f"""
STEP 34A-v2 — ASUS CBIS-DDSM PUBLICATION BASELINE

Historical Kaggle Step-34A was not locally recoverable.
This is an explicitly identified ASUS-local v2 run.

Dataset:
CBIS-DDSM

Manifest:
{MANIFEST}

Manifest SHA256:
{sha256_file(MANIFEST)}

Records:
train={len(train)}
calibration={len(calibration)}
internal_test={len(test)}

Patients:
train={len(train_patients)}
calibration={len(calibration_patients)}
internal_test={len(test_patients)}

Patient overlap:
{ {k: len(v) for k, v in overlaps.items()} }

Model:
ImageNet-pretrained ResNet-50
full-network fine-tuning

Input:
512x512
DICOM full mammogram
grayscale percentile clipping
3-channel replication

Optimization:
AdamW
lr={LEARNING_RATE}
weight_decay={WEIGHT_DECAY}
CosineAnnealingLR
gradient clipping={GRAD_CLIP}
mixed precision=True
maximum epochs={EPOCHS_MAX}

Seed:
{SEED}

Publication artifacts:
checkpoints/
predictions/
metrics/
splits/
figures/
figures/source_data/
tables/
configuration/
evidence/
SHA256_INVENTORY.csv
"""

    (
        OUT
        / "README_REPRODUCTION.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("STEP 34A-v2 COMPLETE")
    print("=" * 100)

    print()
    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best calibration ROC-AUC:",
        best_auc,
    )

    print()
    print(
        "Calibration metrics:",
        cal_metrics,
    )

    print()
    print(
        "Internal-test metrics:",
        test_metrics,
    )

    print()
    print(
        "Bootstrap ROC-AUC 95% CI:",
        bootstrap[
            "roc_auc"
        ],
    )

    print()
    print(
        "Best checkpoint:",
        BEST,
    )

    print(
        "Test predictions:",
        PRED_DIR
        / "INTERNAL_TEST_PREDICTIONS.csv",
    )

    print(
        "Metrics:",
        METRIC_DIR
        / "FINAL_METRICS.json",
    )

    print(
        "Figures:",
        FIG_DIR,
    )

    print(
        "SHA256 inventory:",
        OUT
        / "SHA256_INVENTORY.csv",
    )

    print()
    print(
        "STATUS: STEP34A_V2_COMPLETE"
    )


if __name__ == "__main__":

    main()