from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

DATA_ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    DATA_ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST_EXPERIMENTAL.csv"
)

OUTPUT_ROOT = (
    DATA_ROOT
    / "experiments"
    / "cbis_classical_pilot"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom(path: str) -> np.ndarray:

    ds = pydicom.dcmread(
        path,
        force=True,
    )

    arr = ds.pixel_array.astype(
        np.float32
    )

    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected 2D DICOM image, got "
            f"{arr.shape}: {path}"
        )

    if getattr(
        ds,
        "PhotometricInterpretation",
        ""
    ).upper() == "MONOCHROME1":

        arr = arr.max() - arr

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = float(
        np.percentile(
            arr,
            1.0,
        )
    )

    hi = float(
        np.percentile(
            arr,
            99.0,
        )
    )

    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())

    if hi <= lo:
        arr = np.zeros_like(arr)

    else:

        arr = np.clip(
            arr,
            lo,
            hi,
        )

        arr = (
            (arr - lo)
            / (hi - lo)
        )

    return arr


def resize_tensor(
    image: np.ndarray,
) -> torch.Tensor:

    x = torch.from_numpy(
        image
    ).float()

    x = x.unsqueeze(0)

    x = torch.nn.functional.interpolate(
        x.unsqueeze(0),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    x = x.repeat(
        3,
        1,
        1,
    )

    return x


class CBISDataset(Dataset):

    def __init__(
        self,
        rows,
    ):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(
        self,
        index,
    ):

        row = self.rows[index]

        image = load_dicom(
            row[
                "physical_cropped_dicom"
            ]
        )

        tensor = resize_tensor(
            image
        )

        label = torch.tensor(
            float(row["binary_label"]),
            dtype=torch.float32,
        )

        return (
            tensor,
            label,
            row["patient_id"],
        )


def make_model():

    model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features,
        1,
    )

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0
    n = 0

    for x, y, _ in loader:

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

        logits = (
            model(x)
            .squeeze(1)
        )

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        batch_n = y.shape[0]

        running_loss += (
            loss.item()
            * batch_n
        )

        n += batch_n

    return (
        running_loss / max(n, 1)
    )


@torch.no_grad()
def predict(
    model,
    loader,
):

    model.eval()

    labels = []
    probabilities = []
    patients = []

    for x, y, pids in loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        logits = (
            model(x)
            .squeeze(1)
        )

        prob = torch.sigmoid(
            logits
        )

        labels.extend(
            y.cpu().numpy().tolist()
        )

        probabilities.extend(
            prob.cpu().numpy().tolist()
        )

        patients.extend(
            list(pids)
        )

    return (
        np.asarray(
            labels,
            dtype=np.float32,
        ),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
        patients,
    )


def metrics(
    y,
    p,
):

    pred = (
        p >= 0.5
    ).astype(int)

    out = {
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

        "recall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
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
    }

    return out


def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print("CBIS-DDSM CLASSICAL PILOT")
    print("=" * 80)

    print(
        "Manifest:",
        MANIFEST,
    )

    print(
        "Device:",
        DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    rows = list(
        csv.DictReader(
            MANIFEST.open(
                encoding="utf-8-sig",
                newline="",
            )
        )
    )

    required = {
        "patient_id",
        "binary_label",
        "experimental_split",
        "physical_cropped_dicom",
    }

    missing = (
        required
        - set(rows[0].keys())
    )

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {missing}"
        )

    split_rows = {}

    for split in (
        "train",
        "calibration",
        "internal_test",
    ):

        split_rows[split] = [
            r
            for r in rows
            if r[
                "experimental_split"
            ] == split
        ]

    print()
    print(
        "Records:",
        {
            k: len(v)
            for k, v in split_rows.items()
        }
    )

    # ------------------------------------------------------------
    # Input existence/readability audit for the actual pilot.
    # ------------------------------------------------------------

    for split, items in split_rows.items():

        bad = []

        for r in items:

            path = Path(
                r[
                    "physical_cropped_dicom"
                ]
            )

            if not path.is_file():
                bad.append(
                    str(path)
                )

        print(
            f"{split}: missing cropped DICOMs = {len(bad)}"
        )

        if bad:
            raise RuntimeError(
                f"{split} has missing physical cropped inputs."
            )

    train_ds = CBISDataset(
        split_rows["train"]
    )

    cal_ds = CBISDataset(
        split_rows["calibration"]
    )

    test_ds = CBISDataset(
        split_rows["internal_test"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    cal_loader = DataLoader(
        cal_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model().to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = -1.0
    best_epoch = -1

    checkpoint = (
        OUTPUT_ROOT
        / "best_model.pt"
    )

    print()
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        y_cal, p_cal, _ = predict(
            model,
            cal_loader,
        )

        auc = roc_auc_score(
            y_cal,
            p_cal,
        )

        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"| train loss={loss:.6f} "
            f"| calibration ROC-AUC={auc:.6f}",
            flush=True,
        )

        if auc > best_auc:

            best_auc = auc
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "seed":
                        SEED,
                    "epoch":
                        epoch,
                    "calibration_roc_auc":
                        auc,
                },
                checkpoint,
            )

    # ------------------------------------------------------------
    # Restore best checkpoint.
    # ------------------------------------------------------------

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state[
            "model_state_dict"
        ]
    )

    y_train, p_train, ids_train = predict(
        model,
        DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        ),
    )

    y_cal, p_cal, ids_cal = predict(
        model,
        cal_loader,
    )

    y_test, p_test, ids_test = predict(
        model,
        test_loader,
    )

    results = {
        "experiment": "CBIS-DDSM classical pilot",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "model": "ResNet-50 ImageNet pretrained",
        "input": "physical cropped DICOM",
        "device": str(DEVICE),
        "best_calibration_roc_auc":
            float(best_auc),
        "best_epoch": int(best_epoch),
        "train_records":
            len(train_ds),
        "calibration_records":
            len(cal_ds),
        "internal_test_records":
            len(test_ds),
        "calibration_metrics":
            metrics(
                y_cal,
                p_cal,
            ),
        "internal_test_metrics":
            metrics(
                y_test,
                p_test,
            ),
        "elapsed_seconds":
            float(time.time() - start),
    }

    # ------------------------------------------------------------
    # Save predictions for later reliability analysis.
    # ------------------------------------------------------------

    pred_file = (
        OUTPUT_ROOT
        / "PILOT_PREDICTIONS.csv"
    )

    with pred_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "split",
            "patient_id",
            "label",
            "probability",
        ])

        for pid, y, p in zip(
            ids_train,
            y_train,
            p_train,
        ):

            w.writerow([
                "train",
                pid,
                int(y),
                float(p),
            ])

        for pid, y, p in zip(
            ids_cal,
            y_cal,
            p_cal,
        ):

            w.writerow([
                "calibration",
                pid,
                int(y),
                float(p),
            ])

        for pid, y, p in zip(
            ids_test,
            y_test,
            p_test,
        ):

            w.writerow([
                "internal_test",
                pid,
                int(y),
                float(p),
            ])

    results_file = (
        OUTPUT_ROOT
        / "PILOT_RESULTS.json"
    )

    results_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("PILOT COMPLETE")
    print("=" * 80)
    print(
        "Best calibration ROC-AUC:",
        best_auc,
    )
    print(
        "Best epoch:",
        best_epoch,
    )
    print()
    print(
        "Calibration:",
        results[
            "calibration_metrics"
        ],
    )
    print()
    print(
        "Internal test:",
        results[
            "internal_test_metrics"
        ],
    )
    print()
    print(
        "Checkpoint:",
        checkpoint,
    )
    print(
        "Predictions:",
        pred_file,
    )
    print(
        "Results:",
        results_file,
    )
    print()
    print("STATUS: PILOT COMPLETE")


if __name__ == "__main__":
    main()