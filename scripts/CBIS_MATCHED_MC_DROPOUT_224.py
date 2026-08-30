from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

LATENT_FILE = (
    ROOT
    / "experiments"
    / "cbis_core_vqc_pilot"
    / "SHARED_LATENTS.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_matched_mc_dropout_224"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026
LATENT_DIM = 32
FEATURES = 7

DROPOUT = 0.30
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
MC_PASSES = 50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MatchedMCDropout224(nn.Module):

    def __init__(self):

        super().__init__()

        # Exactly 32 × 7 = 224 trainable parameters.
        self.projection = nn.Linear(
            LATENT_DIM,
            FEATURES,
            bias=False,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

    def forward(self, z):

        z = z.float()

        h = self.projection(
            z
        )

        h = self.dropout(
            h
        )

        # Fixed aggregation: no trainable parameters.
        return h.mean(
            dim=1
        )


def safe_auc(y, score):

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_auprc(y, score):

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y,
            score,
        )
    )


def entropy(p):

    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        1e-7,
        1.0 - 1e-7,
    )

    return -(
        p * np.log(p)
        + (1.0 - p)
        * np.log(1.0 - p)
    )


@torch.no_grad()
def mc_predict(
    model,
    z,
):

    # Keep dropout ACTIVE.
    model.train()

    outputs = []

    for _ in range(
        MC_PASSES
    ):

        logits = model(
            z
        )

        p = torch.sigmoid(
            logits
        )

        outputs.append(
            p.cpu().numpy()
        )

    return np.stack(
        outputs,
        axis=0,
    )


def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print(
        "CBIS-DDSM EXACT 224-PARAMETER MC-DROPOUT"
    )
    print("=" * 80)

    data = torch.load(
        LATENT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    train_z = data[
        "train_z"
    ].float()

    train_y = data[
        "train_y"
    ].float()

    cal_z = data[
        "calibration_z"
    ].float()

    cal_y = data[
        "calibration_y"
    ].float()

    test_z = data[
        "internal_test_z"
    ].float()

    test_y = data[
        "internal_test_y"
    ].float()

    test_patient_ids = list(
        data[
            "internal_test_patient_id"
        ]
    )

    model = MatchedMCDropout224()

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

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

    print(
        "Dropout:",
        DROPOUT,
    )

    print(
        "MC passes:",
        MC_PASSES,
    )

    print(
        "Trainable parameters:",
        parameters,
    )

    if parameters != 224:
        raise RuntimeError(
            f"Expected 224 parameters, got {parameters}"
        )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_auc = -1.0
    best_epoch = -1

    checkpoint = (
        OUT
        / "best_matched_mc_dropout_224.pt"
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

        # Deterministic calibration checkpoint selection.
        model.eval()

        with torch.no_grad():

            cal_logits = model(
                cal_z
            )

            cal_p = torch.sigmoid(
                cal_logits
            ).cpu().numpy()

        cal_auc = safe_auc(
            cal_y.numpy().astype(int),
            cal_p,
        )

        history.append({
            "epoch": epoch,
            "train_loss":
                float(avg_loss),
            "calibration_auc":
                float(cal_auc),
        })

        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"| loss={avg_loss:.6f} "
            f"| calibration ROC-AUC={cal_auc:.6f}",
            flush=True,
        )

        if cal_auc > best_auc:

            best_auc = cal_auc
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

    samples = mc_predict(
        model,
        test_z,
    )

    mean_p = samples.mean(
        axis=0
    )

    variance = samples.var(
        axis=0,
        ddof=1,
    )

    std = samples.std(
        axis=0,
        ddof=1,
    )

    predictive_entropy = entropy(
        mean_p
    )

    expected_entropy = entropy(
        samples
    ).mean(
        axis=0
    )

    mutual_information = (
        predictive_entropy
        - expected_entropy
    )

    prediction = (
        mean_p >= 0.5
    ).astype(int)

    y = test_y.numpy().astype(
        int
    )

    error = (
        prediction != y
    ).astype(int)

    confidence = np.maximum(
        mean_p,
        1.0 - mean_p,
    )

    uncertainty = {
        "predictive_variance":
            variance,
        "predictive_std":
            std,
        "predictive_entropy":
            predictive_entropy,
        "expected_entropy":
            expected_entropy,
        "mutual_information":
            mutual_information,
    }

    error_results = {}

    for name, score in (
        uncertainty.items()
    ):

        error_results[name] = {
            "error_detection_auroc":
                safe_auc(
                    error,
                    score,
                ),

            "error_detection_auprc":
                safe_auprc(
                    error,
                    score,
                ),

            "mean_uncertainty_correct":
                float(
                    score[
                        error == 0
                    ].mean()
                ),

            "mean_uncertainty_incorrect":
                float(
                    score[
                        error == 1
                    ].mean()
                ),
        }

    # --------------------------------------------------------
    # Confidence-based AURC
    # --------------------------------------------------------

    order = np.argsort(
        -confidence
    )

    sorted_errors = error[
        order
    ]

    cumulative_errors = np.cumsum(
        sorted_errors
    )

    coverages = np.linspace(
        1.0 / len(error),
        1.0,
        101,
    )

    risks = []

    for coverage in coverages:

        k = max(
            1,
            int(
                round(
                    coverage
                    * len(error)
                )
            ),
        )

        risks.append(
            float(
                cumulative_errors[
                    k - 1
                ]
                / k
            )
        )

    aurc = float(
        np.trapezoid(
            risks,
            coverages,
        )
    )

    result_df = pd.DataFrame({
        "patient_id":
            test_patient_ids,
        "label":
            y,
        "mean_probability":
            mean_p,
        "prediction":
            prediction,
        "error":
            error,
        "confidence":
            confidence,
        "predictive_variance":
            variance,
        "predictive_std":
            std,
        "predictive_entropy":
            predictive_entropy,
        "expected_entropy":
            expected_entropy,
        "mutual_information":
            mutual_information,
    })

    prediction_file = (
        OUT
        / "MATCHED_MC_DROPOUT_224_PER_RECORD.csv"
    )

    result_df.to_csv(
        prediction_file,
        index=False,
    )

    results = {
        "experiment":
            "CBIS-DDSM exact 224-parameter MC-Dropout",

        "seed":
            SEED,

        "trainable_parameters":
            parameters,

        "dropout":
            DROPOUT,

        "mc_passes":
            MC_PASSES,

        "best_epoch":
            best_epoch,

        "best_calibration_auc":
            float(best_auc),

        "internal_test_records":
            int(len(y)),

        "error_detection":
            error_results,

        "confidence_aurc":
            aurc,

        "checkpoint":
            str(checkpoint),

        "prediction_file":
            str(prediction_file),

        "status":
            "PHASE5_MATCHED_MC_DROPOUT_COMPLETE",
    }

    results_file = (
        OUT
        / "MATCHED_MC_DROPOUT_224_RESULTS.json"
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
    print(
        "MATCHED 224-PARAMETER MC-DROPOUT SUMMARY"
    )
    print("=" * 80)

    for name, r in (
        error_results.items()
    ):

        print()
        print(name)

        print(
            "  Error-detection AUROC:",
            r[
                "error_detection_auroc"
            ],
        )

        print(
            "  Error-detection AUPRC:",
            r[
                "error_detection_auprc"
            ],
        )

        print(
            "  Mean uncertainty correct:",
            r[
                "mean_uncertainty_correct"
            ],
        )

        print(
            "  Mean uncertainty incorrect:",
            r[
                "mean_uncertainty_incorrect"
            ],
        )

    print()
    print(
        "AURC:",
        aurc,
    )

    print(
        "Trainable parameters:",
        parameters,
    )

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
        "STATUS: PHASE5_MATCHED_MC_DROPOUT_COMPLETE"
    )


if __name__ == "__main__":
    main()