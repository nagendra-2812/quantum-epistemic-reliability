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
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

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

RECON = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
)

AVAILABLE = (
    RECON
    / "CBIS_PUBLICATION_AVAILABLE_RECORDS.csv"
)

MAPPING = (
    RECON
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP34A_V2_FINAL_ASUS_PUBLICATION"
)

CHECKPOINT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
SPLIT_DIR = OUT / "splits"
FIG_DIR = OUT / "figures"
SOURCE_DIR = FIG_DIR / "source_data"
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
    SOURCE_DIR,
    TABLE_DIR,
    CONFIG_DIR,
    EVIDENCE_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# Publication-oriented settings
IMAGE_SIZE = 512
BATCH_SIZE = 8
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# BASIC CHECKS
# ============================================================

print()
print("=" * 100)
print("STEP 34A-v2 Ã¢â‚¬â€ FINAL ASUS CBIS RESNET-50 PUBLICATION RUN")
print("=" * 100)

if DEVICE.type != "cuda":

    raise RuntimeError(
        "CUDA GPU is not available. "
        "Do not run publication training on CPU."
    )

print()
print(
    "Python:",
    sys.version.split()[0]
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA:",
    torch.version.cuda
)


# ============================================================
# SEED
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

            h.update(block)

    return h.hexdigest()


# ============================================================
# LOAD FROZEN AVAILABLE COHORT
# ============================================================

if not AVAILABLE.is_file():

    raise RuntimeError(
        f"Available manifest not found: {AVAILABLE}"
    )

if not MAPPING.is_file():

    raise RuntimeError(
        f"Physical mapping not found: {MAPPING}"
    )

available = pd.read_csv(
    AVAILABLE
)

mapping = pd.read_csv(
    MAPPING
)

print()
print("=" * 100)
print("FROZEN LOCAL COHORT")
print("=" * 100)

print(
    "Available canonical records:",
    len(available),
)

print(
    "Canonical mapping rows:",
    len(mapping),
)

# ============================================================
# STABLE KEY
# ============================================================

def record_key(row):

    return (
        str(
            row.get(
                "patient_id",
                ""
            )
        ).strip(),

        str(
            row.get(
                "source_table",
                ""
            )
        ).strip(),

        str(
            row.get(
                "abnormality_id",
                ""
            )
        ).strip(),

        str(
            row.get(
                "image_view",
                ""
            )
        ).strip(),

        str(
            row.get(
                "laterality",
                ""
            )
        ).strip(),
    )


available_keys = {
    record_key(
        row
    )
    for _, row
    in available.iterrows()
}

mapping_lookup = {}

for _, row in mapping.iterrows():

    mapping_lookup[
        record_key(row)
    ] = row


# ============================================================
# CREATE ONE-IMAGE-PER-RECORD DATAFRAME
#
# Selection rule:
#   among physical DICOM objects already associated with the
#   validated record, choose the valid pixel-bearing DICOM
#   with the largest pixel area.
#
# This prevents choosing a small ROI/crop where a larger image
# object is available.
# ============================================================

print()
print("=" * 100)
print("FINAL INPUT IMAGE SELECTION")
print("=" * 100)


def inspect_candidate(path):

    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=False,
        force=True,
    )

    if not hasattr(
        ds,
        "PixelData",
    ):

        raise RuntimeError(
            "No PixelData"
        )

    arr = ds.pixel_array

    if arr.ndim != 2:

        raise RuntimeError(
            f"Expected 2-D image, got ndim={arr.ndim}"
        )

    rows = int(
        getattr(
            ds,
            "Rows",
            arr.shape[0],
        )
        or arr.shape[0]
    )

    cols = int(
        getattr(
            ds,
            "Columns",
            arr.shape[1],
        )
        or arr.shape[1]
    )

    if rows <= 0 or cols <= 0:

        raise RuntimeError(
            "Invalid image dimensions"
        )

    description = str(
        getattr(
            ds,
            "SeriesDescription",
            "",
        )
        or ""
    )

    image_type = str(
        getattr(
            ds,
            "ImageType",
            "",
        )
        or ""
    )

    text = (
        description
        + " "
        + image_type
    ).lower()

    score = float(
        rows * cols
    )

    if "roi" in text:
        score -= 1e9

    if "crop" in text:
        score -= 1e9

    if "mask" in text:
        score -= 1e9

    return {
        "path":
            str(path),

        "rows":
            rows,

        "columns":
            cols,

        "pixel_count":
            rows * cols,

        "description":
            description,

        "image_type":
            image_type,

        "score":
            score,
    }


selected_rows = []
record_failures = []
candidate_read_failures = []

for n, (_, row) in enumerate(
    available.iterrows(),
    1,
):

    key = record_key(
        row
    )

    mapped = mapping_lookup.get(
        key
    )

    if mapped is None:

        record_failures.append({

            "patient_id":
                str(
                    row[
                        "patient_id"
                    ]
                ),

            "source_table":
                str(
                    row[
                        "source_table"
                    ]
                ),

            "abnormality_id":
                str(
                    row[
                        "abnormality_id"
                    ]
                ),

            "image_view":
                str(
                    row[
                        "image_view"
                    ]
                ),

            "failure_reason":
                "MAPPING_NOT_FOUND",
        })

        continue


    raw = str(
        mapped.get(
            "physical_dicom_paths",
            ""
        )
        or ""
    )

    paths = [
        Path(
            x
        )
        for x in raw.split("|")
        if str(x).strip()
    ]

    paths = [
        p
        for p in paths
        if p.is_file()
    ]

    valid = []

    for p in paths:

        try:

            info = inspect_candidate(
                p
            )

            valid.append(
                info
            )

        except Exception as exc:

            candidate_read_failures.append({

                "patient_id":
                    str(
                        row[
                            "patient_id"
                        ]
                    ),

                "source_table":
                    str(
                        row[
                            "source_table"
                        ]
                    ),

                "abnormality_id":
                    str(
                        row[
                            "abnormality_id"
                        ]
                    ),

                "image_view":
                    str(
                        row[
                            "image_view"
                        ]
                    ),

                "path":
                    str(p),

                "error":
                    repr(exc),
            })


    if not valid:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "NO_VALID_PIXEL_DICOM",
        })

        continue


    # --------------------------------------------------------
    # Select maximum score.
    # --------------------------------------------------------

    best_score = max(
        x[
            "score"
        ]
        for x in valid
    )

    best = [
        x
        for x in valid
        if x[
            "score"
        ] == best_score
    ]

    # If exact score tie, choose largest pixel count.
    if len(best) > 1:

        max_pixels = max(
            x[
                "pixel_count"
            ]
            for x in best
        )

        best = [
            x
            for x in best
            if x[
                "pixel_count"
            ] == max_pixels
        ]


    if len(best) != 1:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "AMBIGUOUS_IMAGE_SELECTION",

            "candidate_count":
                len(valid),

            "candidate_paths":
                "|".join(
                    x[
                        "path"
                    ]
                    for x in valid
                ),
        })

        continue


    chosen = best[0]

    item = row.to_dict()

    item.update({

        "resolved_full_mammogram_dicom":
            chosen[
                "path"
            ],

        "resolved_rows":
            chosen[
                "rows"
            ],

        "resolved_columns":
            chosen[
                "columns"
            ],

        "resolved_pixel_count":
            chosen[
                "pixel_count"
            ],

        "resolved_series_description":
            chosen[
                "description"
            ],

        "resolved_image_type":
            chosen[
                "image_type"
            ],

        "resolved_candidate_count":
            len(valid),

        "candidate_read_failure_count":
            len(paths)
            -
            len(valid),

        "selection_rule":
            (
                "largest valid pixel-bearing DICOM "
                "after ROI/crop/mask exclusion"
            ),

        "input_status":
            "VERIFIED",
    })

    selected_rows.append(
        item
    )


    if (
        (n + 1) % 250
        == 0
    ):

        print(
            f"Processed {n + 1}/{len(available)}...",
            flush=True,
        )


selected_df = pd.DataFrame(
    selected_rows
)

failure_df = pd.DataFrame(
    record_failures
)

candidate_failure_df = pd.DataFrame(
    candidate_read_failures
)


# ============================================================
# REPORT INPUT SELECTION
# ============================================================

print()
print("=" * 100)
print("FINAL INPUT SELECTION RESULT")
print("=" * 100)

print(
    "Available canonical records:",
    len(available)
)

print(
    "Selected input records:",
    len(selected_df)
)

print(
    "Record-level selection failures:",
    len(failure_df)
)

print(
    "Individual candidate DICOM read failures:",
    len(candidate_failure_df)
)


# ============================================================
# SPLIT COUNTS
# ============================================================

if len(selected_df):

    print()
    print(
        "Selected split counts:"
    )

    print(
        selected_df[
            "experimental_split"
        ]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "Selected patient counts:"
    )

    print(
        selected_df.groupby(
            "experimental_split"
        )[
            "patient_id"
        ]
        .nunique()
        .to_dict()
    )


# ============================================================
# PATIENT DISJOINTNESS
# ============================================================

def patients_for(
    frame,
    split,
):

    return set(
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


train_patients = patients_for(
    selected_df,
    "train",
)

cal_patients = patients_for(
    selected_df,
    "calibration",
)

test_patients = patients_for(
    selected_df,
    "internal_test",
)

overlap = {

    "train_calibration":
        sorted(
            train_patients
            &
            cal_patients
        ),

    "train_internal_test":
        sorted(
            train_patients
            &
            test_patients
        ),

    "calibration_internal_test":
        sorted(
            cal_patients
            &
            test_patients
        ),
}


print()
print(
    "Patient overlaps:",
    {
        k:
            len(v)
        for k, v
        in overlap.items()
    }
)


# ============================================================
# SAVE FROZEN INPUT MANIFEST
# ============================================================

if len(failure_df) > 0:

    failure_df.to_csv(
        OUT
        / "CBIS_V2_INPUT_SELECTION_FAILURES.csv",
        index=False,
    )

    print()
    print(
        "WARNING: some canonical records could not be selected."
    )

    print(
        "These records are explicitly excluded from the "
        "ASUS-local v2 publication cohort."
    )

    print(
        "Failure file:",
        OUT
        / "CBIS_V2_INPUT_SELECTION_FAILURES.csv",
    )


if len(selected_df) == 0:

    raise RuntimeError(
        "No usable CBIS records remain."
    )


if any(
    len(v) > 0
    for v in overlap.values()
):

    raise RuntimeError(
        "Patient overlap detected after input freezing."
    )


FINAL_INPUT = (
    OUT
    / "CBIS_V2_FINAL_PHYSICAL_INPUT_MANIFEST.csv"
)

selected_df.to_csv(
    FINAL_INPUT,
    index=False,
)

candidate_failure_df.to_csv(
    OUT
    / "CBIS_V2_CANDIDATE_DICOM_READ_FAILURES.csv",
    index=False,
)


# ============================================================
# ASUS LOCAL V2 COHORT FREEZE
# ============================================================

CANONICAL_RECORD_COUNT = int(
    len(available)
)

SELECTED_RECORD_COUNT = int(
    len(selected_df)
)

EXCLUDED_RECORD_COUNT = int(
    len(available)
    -
    len(selected_df)
)

COHORT_FREEZE_NOTE = (
    "ASUS-local CBIS publication cohort. "
    "Records without a uniquely selectable valid local "
    "pixel-bearing DICOM were excluded explicitly. "
    "No substitution or synthetic reconstruction was used."
)

# ============================================================
# DATASET
# ============================================================

def dicom_to_image(path):

    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    arr = ds.pixel_array.astype(
        np.float32
    )

    if arr.ndim != 2:

        raise RuntimeError(
            f"Non-2D image: {path}"
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
            arr
            - lo
        )
        /
        (
            hi
            - lo
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

        image = dicom_to_image(
            row[
                "resolved_full_mammogram_dicom"
            ]
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
            str(
                row[
                    "patient_id"
                ]
            ),
        )


train = selected_df[
    selected_df[
        "experimental_split"
    ]
    == "train"
].copy()

calibration = selected_df[
    selected_df[
        "experimental_split"
    ]
    == "calibration"
].copy()

test = selected_df[
    selected_df[
        "experimental_split"
    ]
    == "internal_test"
].copy()


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
        [
            0.485,
            0.456,
            0.406,
        ],
        [
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
        [
            0.485,
            0.456,
            0.406,
        ],
        [
            0.229,
            0.224,
            0.225,
        ],
    ),
])


train_loader = DataLoader(
    CBISDataset(
        train,
        train_transform,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

calibration_loader = DataLoader(
    CBISDataset(
        calibration,
        eval_transform,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    CBISDataset(
        test,
        eval_transform,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 100)
print("BUILDING RESNET-50")
print("=" * 100)

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

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
)

scaler = torch.amp.GradScaler(
    "cuda"
)


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    loader,
):

    model.eval()

    ys = []
    ps = []
    ids = []

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

            p = torch.sigmoid(
                logits
            )

            ys.extend(
                y.cpu()
                .numpy()
                .tolist()
            )

            ps.extend(
                p.cpu()
                .numpy()
                .tolist()
            )

            ids.extend(
                list(
                    patient_ids
                )
            )

    return (
        np.asarray(
            ys,
            dtype=int,
        ),
        np.asarray(
            ps,
            dtype=float,
        ),
        ids,
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y,
    p,
):

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

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 100)
print("RESNET-50 TRAINING STARTED")
print("=" * 100)

history = []

best_auc = -np.inf
best_epoch = -1
best_state = None

training_start = time.time()

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()

    total_loss = 0.0
    total_n = 0

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

        total_loss += (
            float(
                loss.detach()
            )
            *
            len(y)
        )

        total_n += len(y)

    scheduler.step()

    y_cal, p_cal, _ = predict(
        model,
        calibration_loader,
    )

    cal_metrics = metrics(
        y_cal,
        p_cal,
    )

    epoch_loss = (
        total_loss
        /
        max(
            total_n,
            1,
        )
    )

    lr = (
        optimizer.param_groups[0][
            "lr"
        ]
    )

    history.append({

        "epoch":
            epoch,

        "train_loss":
            float(
                epoch_loss
            ),

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
                lr
            ),
    })

    print(
        f"epoch {epoch:02d}/{MAX_EPOCHS} "
        f"| loss={epoch_loss:.6f} "
        f"| calibration ROC-AUC="
        f"{cal_metrics['roc_auc']:.6f} "
        f"| lr={lr:.7f}",
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
    training_start
)


# ============================================================
# SAVE CHECKPOINTS
# ============================================================

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


# ============================================================
# FINAL TEST
# ============================================================

y_cal, p_cal, cal_ids = predict(
    model,
    calibration_loader,
)

y_test, p_test, test_ids = predict(
    model,
    test_loader,
)

cal_metrics = metrics(
    y_cal,
    p_cal,
)

test_metrics = metrics(
    y_test,
    p_test,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

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


# ============================================================
# ROC / PR SOURCE DATA
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    p_test,
)

pd.DataFrame({

    "false_positive_rate":
        fpr,

    "true_positive_rate":
        tpr,

    "threshold":
        roc_thresholds,

}).to_csv(
    SOURCE_DIR
    / "ROC_SOURCE_DATA.csv",
    index=False,
)


precision, recall, pr_thresholds = (
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
    SOURCE_DIR
    / "PR_SOURCE_DATA.csv",
    index=False,
)


pred_test = (
    p_test >= 0.5
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
        "true_benign",
        "true_malignant",
    ],
    columns=[
        "pred_benign",
        "pred_malignant",
    ],
).to_csv(
    SOURCE_DIR
    / "CONFUSION_MATRIX_SOURCE_DATA.csv"
)


# ============================================================
# BOOTSTRAP ROC-AUC
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

    vals = []

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

        vals.append(
            roc_auc_score(
                yy,
                pp,
            )
        )

    vals = np.asarray(
        vals,
        dtype=float,
    )

    low, high = np.percentile(
        vals,
        [
            2.5,
            97.5,
        ],
    )

    return {

        "estimate":
            float(observed),

        "lower_95":
            float(low),

        "upper_95":
            float(high),

        "valid_bootstrap_samples":
            int(
                len(vals)
            ),
    }


bootstrap = bootstrap_auc(
    y_test,
    p_test,
    SEED,
    2000,
)


# ============================================================
# PRIMARY TABLE
# ============================================================

primary_table = pd.DataFrame([{

    "Dataset":
        "CBIS-DDSM",

    "Cohort":
        "ASUS-CBIS-v2",

    "Model":
        "ResNet-50",

    "Test_records":
        len(test),

    "Test_patients":
        len(test_patients),

    "ROC_AUC":
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

    "Balanced_accuracy":
        test_metrics[
            "balanced_accuracy"
        ],

    "Sensitivity":
        test_metrics[
            "sensitivity"
        ],

    "Specificity":
        test_metrics[
            "specificity"
        ],

    "F1":
        test_metrics[
            "f1"
        ],

    "Brier":
        test_metrics[
            "brier"
        ],

    "ROC_AUC_CI_low":
        bootstrap[
            "lower_95"
        ],

    "ROC_AUC_CI_high":
        bootstrap[
            "upper_95"
        ],
}])


primary_table.to_csv(
    TABLE_DIR
    / "TABLE_01_CBIS_RESNET50.csv",
    index=False,
)


# ============================================================
# FIGURE 1 Ã¢â‚¬â€ ROC
# ============================================================

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


# ============================================================
# FIGURE 2 Ã¢â‚¬â€ PR
# ============================================================

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
    "CBIS-DDSM PrecisionÃ¢â‚¬â€œRecall Curve"
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


# ============================================================
# FIGURE 3 Ã¢â‚¬â€ TRAINING HISTORY
# ============================================================

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
    hist[
        "epoch"
    ],
    hist[
        "train_loss"
    ],
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
    "CBIS-DDSM ResNet-50 Training History"
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


# ============================================================
# FIGURE 4 Ã¢â‚¬â€ CONFUSION MATRIX
# ============================================================

fig = plt.figure(
    figsize=(
        5.8,
        5.2,
    )
)

plt.imshow(
    cm
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


# ============================================================
# METRICS JSON
# ============================================================

metrics_json = {

    "status":
        "STEP34A_V2_COMPLETE",

    "canonical_record_count":
        CANONICAL_RECORD_COUNT,

    "selected_record_count":
        SELECTED_RECORD_COUNT,

    "excluded_record_count":
        EXCLUDED_RECORD_COUNT,

    "cohort_freeze_note":
        COHORT_FREEZE_NOTE,

    "run":
        "ASUS_LOCAL_CBIS_V2",

    "dataset":
        "CBIS-DDSM",

    "canonical_records":
        3568,

    "physically_available_records":
        3432,

    "physically_unavailable_records":
        136,

    "train_records":
        len(train),

    "calibration_records":
        len(calibration),

    "internal_test_records":
        len(test),

    "train_patients":
        len(train_patients),

    "calibration_patients":
        len(cal_patients),

    "internal_test_patients":
        len(test_patients),

    "patient_overlap":
        {
            k:
                len(v)
            for k, v
            in overlap.items()
        },

    "model":
        "ImageNet-pretrained ResNet-50",

    "fine_tuning":
        "full-network",

    "input_resolution":
        "512x512",

    "input_source":
        "resolved local DICOM",

    "preprocessing":
        "deterministic 1st-99th percentile clipping + grayscale to 3 channels",

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

    "max_epochs":
        MAX_EPOCHS,

    "best_epoch":
        int(
            best_epoch
        ),

    "best_calibration_roc_auc":
        float(
            best_auc
        ),

    "calibration_metrics":
        cal_metrics,

    "internal_test_metrics":
        test_metrics,

    "bootstrap_roc_auc":
        bootstrap,

    "training_seconds":
        float(
            training_seconds
        ),

    "seed":
        SEED,

    "canonical_manifest":
        str(
            AVAILABLE
        ),

    "canonical_manifest_sha256":
        sha256_file(
            AVAILABLE
        ),

    "final_input_manifest":
        str(
            FINAL_INPUT
        ),

    "best_checkpoint":
        str(
            BEST
        ),

    "last_checkpoint":
        str(
            LAST
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


# ============================================================
# CONFIGURATION
# ============================================================

configuration = {

    "run":
        "ASUS_LOCAL_CBIS_V2",

    "dataset":
        "CBIS-DDSM",

    "canonical_records":
        3568,

    "available_records":
        3432,

    "unavailable_records":
        136,

    "train_records":
        len(train),

    "calibration_records":
        len(calibration),

    "internal_test_records":
        len(test),

    "train_patients":
        len(train_patients),

    "calibration_patients":
        len(cal_patients),

    "internal_test_patients":
        len(test_patients),

    "model":
        "ResNet-50",

    "weights":
        "ImageNet",

    "fine_tuning":
        "full-network",

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "max_epochs":
        MAX_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "optimizer":
        "AdamW",

    "scheduler":
        "CosineAnnealingLR",

    "gradient_clip":
        GRAD_CLIP,

    "mixed_precision":
        True,

    "class_weighting":
        "disabled",

    "selection_rule":
        (
            "largest valid pixel-bearing DICOM "
            "after ROI/crop/mask exclusion"
        ),

    "seed":
        SEED,

    "status":
        "FROZEN",
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

    "gpu_count":
        torch.cuda.device_count(),
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
# SPLIT FILES
# ============================================================

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


# ============================================================
# SHA-256 INVENTORY
# ============================================================

checksum_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
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
# README
# ============================================================

readme = f"""
STEP 34A-v2 Ã¢â‚¬â€ ASUS LOCAL CBIS-DDSM PUBLICATION RUN

Historical Kaggle Step-34A is not locally recoverable.
This is an explicitly identified ASUS-local v2 experiment.

Canonical CBIS records:
3568

Physically available:
3432

Physically unavailable:
136

Frozen local split:
Train:
  records={len(train)}
  patients={len(train_patients)}

Calibration:
  records={len(calibration)}
  patients={len(cal_patients)}

Internal test:
  records={len(test)}
  patients={len(test_patients)}

Patient overlap:
train-calibration = {len(overlap['train_calibration'])}
train-test = {len(overlap['train_internal_test'])}
calibration-test = {len(overlap['calibration_internal_test'])}

Model:
ImageNet-pretrained ResNet-50

Input:
Resolved local DICOM full-image object
512x512
grayscale percentile clipping
3-channel replication

Optimization:
AdamW
learning rate={LEARNING_RATE}
weight decay={WEIGHT_DECAY}
CosineAnnealingLR
gradient clipping={GRAD_CLIP}
mixed precision=True
maximum epochs={MAX_EPOCHS}

Best epoch:
{best_epoch}

Best calibration ROC-AUC:
{best_auc}

Internal-test ROC-AUC:
{test_metrics['roc_auc']}

Internal-test AUPRC:
{test_metrics['auprc']}

Bootstrap ROC-AUC 95% CI:
{bootstrap['lower_95']} to {bootstrap['upper_95']}

Publication artifacts:
- checkpoint
- predictions
- metrics
- bootstrap CI
- split CSVs
- ROC source CSV
- PR source CSV
- confusion source CSV
- vector PDF figures
- vector SVG figures
- 400-DPI PNG figures
- environment
- configuration
- SHA256 inventory
"""

(
    OUT
    / "README_REPRODUCTION.txt"
).write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

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
    bootstrap,
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
    "Final input manifest:",
    FINAL_INPUT,
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