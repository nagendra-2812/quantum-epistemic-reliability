from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
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

ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

LATENT_FILE = (
    ROOT
    / "experiments"
    / "cbis_core_vqc_pilot"
    / "SHARED_LATENTS.pt"
)

OUTPUT_ROOT = (
    ROOT
    / "experiments"
    / "cbis_matched_classical_224"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026
LATENT_DIM = 32
FEATURES = 7
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MatchedClassical224(nn.Module):

    def __init__(self):
        super().__init__()

        # Exactly 32 x 7 = 224 trainable parameters.
        self.projection = nn.Linear(
            LATENT_DIM,
            FEATURES,
            bias=False,
        )

    def forward(self, z):

        h = self.projection(
            z.float()
        )

        # Fixed aggregation: no additional trainable
        # parameters are introduced.
        return h.mean(
            dim=1
        )


def metrics(y, p):

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, p)
        ),
    }


@torch.no_grad()
def predict(
    model,
    z,
    y,
):

    model.eval()

    probabilities = []

    for start in range(
        0,
        len(z),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(z),
        )

        logits = model(
            z[start:end]
        )

        probabilities.extend(
            torch.sigmoid(
                logits
            ).cpu().numpy().tolist()
        )

    return (
        y.numpy(),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
    )


def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print(
        "CBIS-DDSM EXACT 224-PARAMETER CLASSICAL PILOT"
    )
    print("=" * 80)

    data = torch.load(
        LATENT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    train_z = data["train_z"].float()
    train_y = data["train_y"].float()

    cal_z = data["calibration_z"].float()
    cal_y = data["calibration_y"].float()

    test_z = data["internal_test_z"].float()
    test_y = data["internal_test_y"].float()

    print(
        "Train:",
        tuple(train_z.shape)
    )

    print(
        "Calibration:",
        tuple(cal_z.shape)
    )

    print(
        "Internal test:",
        tuple(test_z.shape)
    )

    model = MatchedClassical224()

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        params
    )

    assert params == 224

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_auc = -1.0
    best_epoch = -1

    checkpoint = (
        OUTPUT_ROOT
        / "best_matched_classical_224.pt"
    )

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        permutation = torch.randperm(
            len(train_z)
        )

        total_loss = 0.0
        total_n = 0

        for start in range(
            0,
            len(train_z),
            BATCH_SIZE,
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            xb = train_z[idx]
            yb = train_y[idx]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            n = len(yb)

            total_loss += (
                loss.item()
                * n
            )

            total_n += n

        avg_loss = (
            total_loss
            / total_n
        )

        y_cal, p_cal = predict(
            model,
            cal_z,
            cal_y,
        )

        auc = roc_auc_score(
            y_cal,
            p_cal,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(
                avg_loss
            ),
            "calibration_roc_auc": float(
                auc
            ),
        })

        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"| loss={avg_loss:.6f} "
            f"| calibration ROC-AUC={auc:.6f}",
            flush=True,
        )

        if auc > best_auc:

            best_auc = auc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                checkpoint,
            )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    )

    y_cal, p_cal = predict(
        model,
        cal_z,
        cal_y,
    )

    y_test, p_test = predict(
        model,
        test_z,
        test_y,
    )

    cal_metrics = metrics(
        y_cal,
        p_cal,
    )

    test_metrics = metrics(
        y_test,
        p_test,
    )

    prediction_file = (
        OUTPUT_ROOT
        / "MATCHED_CLASSICAL_224_PREDICTIONS.csv"
    )

    with prediction_file.open(
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
            data["calibration_patient_id"],
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
            data["internal_test_patient_id"],
            y_test,
            p_test,
        ):

            w.writerow([
                "internal_test",
                pid,
                int(y),
                float(p),
            ])

    results = {
        "experiment":
            "CBIS-DDSM exact 224-parameter classical comparator",

        "seed":
            SEED,

        "latent_dim":
            LATENT_DIM,

        "trainable_parameters":
            params,

        "architecture":
            "32 -> 7 bias-free linear -> fixed mean",

        "best_epoch":
            best_epoch,

        "best_calibration_auc":
            float(best_auc),

        "calibration_metrics":
            cal_metrics,

        "internal_test_metrics":
            test_metrics,

        "history":
            history,

        "checkpoint":
            str(checkpoint),

        "predictions":
            str(prediction_file),

        "status":
            "PILOT_COMPLETE",
    }

    results_file = (
        OUTPUT_ROOT
        / "MATCHED_CLASSICAL_224_RESULTS.json"
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
    print("MATCHED CLASSICAL PILOT COMPLETE")
    print("=" * 80)

    print(
        "Trainable parameters:",
        params,
    )

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
        "Checkpoint:",
        checkpoint,
    )

    print(
        "Predictions:",
        prediction_file,
    )

    print(
        "Results:",
        results_file,
    )

    print()
    print(
        "STATUS: PILOT_COMPLETE"
    )


if __name__ == "__main__":
    main()