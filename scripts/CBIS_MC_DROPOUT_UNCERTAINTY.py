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
    / "cbis_mc_dropout_uncertainty"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026
LATENT_DIM = 32

HIDDEN_DIM = 16
DROPOUT = 0.30

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

MC_PASSES = 50


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MCDropoutHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                LATENT_DIM,
                HIDDEN_DIM,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                HIDDEN_DIM,
                1,
            ),
        )

    def forward(self, z):

        return self.net(
            z.float()
        ).squeeze(1)


def binary_entropy(p):

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


def safe_auc(y, score):

    if len(
        np.unique(y)
    ) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_auprc(y, score):

    if len(
        np.unique(y)
    ) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y,
            score,
        )
    )


@torch.no_grad()
def mc_predict(
    model,
    z,
):

    # IMPORTANT:
    # Dropout remains ACTIVE during inference.
    model.train()

    outputs = []

    for _ in range(
        MC_PASSES
    ):

        logits = model(
            z
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        outputs.append(
            probabilities
            .cpu()
            .numpy()
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
        "CBIS-DDSM MC-DROPOUT UNCERTAINTY"
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

    test_patient = list(
        data[
            "internal_test_patient_id"
        ]
    )

    model = MCDropoutHead()

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
        / "best_mc_dropout.pt"
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

            xb = train_z[
                idx
            ]

            yb = train_y[
                idx
            ]

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

        average_loss = (
            total_loss
            / total_n
        )

        # Deterministic evaluation for checkpoint selection.
        model.eval()

        with torch.no_grad():

            cal_probability = (
                torch.sigmoid(
                    model(
                        cal_z
                    )
                )
                .cpu()
                .numpy()
            )

        cal_auc = safe_auc(
            cal_y.numpy().astype(int),
            cal_probability,
        )

        history.append({
            "epoch": epoch,
            "train_loss":
                float(average_loss),
            "calibration_auc":
                float(cal_auc),
        })

        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"| loss={average_loss:.6f} "
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

    # --------------------------------------------------------
    # MC predictions
    # --------------------------------------------------------

    test_samples = mc_predict(
        model,
        test_z,
    )

    mean_probability = (
        test_samples.mean(
            axis=0
        )
    )

    probability_variance = (
        test_samples.var(
            axis=0,
            ddof=1,
        )
    )

    probability_std = (
        test_samples.std(
            axis=0,
            ddof=1,
        )
    )

    predictive_entropy = (
        binary_entropy(
            mean_probability
        )
    )

    expected_entropy = (
        binary_entropy(
            test_samples
        ).mean(
            axis=0
        )
    )

    mutual_information = (
        predictive_entropy
        - expected_entropy
    )

    prediction = (
        mean_probability
        >= 0.5
    ).astype(int)

    error = (
        prediction
        != test_y.numpy().astype(int)
    ).astype(int)

    confidence = np.maximum(
        mean_probability,
        1.0 - mean_probability,
    )

    # --------------------------------------------------------
    # Error detection
    # --------------------------------------------------------

    uncertainty = {
        "predictive_variance":
            probability_variance,

        "predictive_std":
            probability_std,

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
    # Risk-coverage
    # --------------------------------------------------------

    order = np.argsort(
        -confidence
    )

    errors_sorted = error[
        order
    ]

    cumulative_errors = (
        np.cumsum(
            errors_sorted
        )
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

    # --------------------------------------------------------
    # Save per-record results
    # --------------------------------------------------------

    df = pd.DataFrame({
        "patient_id":
            test_patient,

        "label":
            test_y.numpy().astype(int),

        "mean_probability":
            mean_probability,

        "prediction":
            prediction,

        "error":
            error,

        "predictive_variance":
            probability_variance,

        "predictive_std":
            probability_std,

        "predictive_entropy":
            predictive_entropy,

        "expected_entropy":
            expected_entropy,

        "mutual_information":
            mutual_information,

        "confidence":
            confidence,
    })

    prediction_file = (
        OUT
        / "MC_DROPOUT_UNCERTAINTY_PER_RECORD.csv"
    )

    df.to_csv(
        prediction_file,
        index=False,
    )

    results = {
        "experiment":
            "CBIS-DDSM MC-Dropout uncertainty",

        "seed":
            SEED,

        "dropout":
            DROPOUT,

        "mc_passes":
            MC_PASSES,

        "trainable_parameters":
            parameters,

        "best_epoch":
            best_epoch,

        "best_calibration_auc":
            float(best_auc),

        "internal_test_records":
            int(len(test_z)),

        "error_detection":
            error_results,

        "risk_coverage": {
            "aurc":
                aurc,
        },

        "checkpoint":
            str(checkpoint),

        "prediction_file":
            str(prediction_file),

        "status":
            "PHASE5_MC_DROPOUT_COMPLETE",
    }

    results_file = (
        OUT
        / "MC_DROPOUT_RESULTS.json"
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
        "MC-DROPOUT UNCERTAINTY SUMMARY"
    )
    print("=" * 80)

    for name, value in (
        error_results.items()
    ):

        print()
        print(name)

        print(
            "  Error-detection AUROC:",
            value[
                "error_detection_auroc"
            ],
        )

        print(
            "  Error-detection AUPRC:",
            value[
                "error_detection_auprc"
            ],
        )

        print(
            "  Mean uncertainty correct:",
            value[
                "mean_uncertainty_correct"
            ],
        )

        print(
            "  Mean uncertainty incorrect:",
            value[
                "mean_uncertainty_incorrect"
            ],
        )

    print()
    print(
        "AURC:",
        aurc,
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
        "STATUS: PHASE5_MC_DROPOUT_COMPLETE"
    )


if __name__ == "__main__":
    main()