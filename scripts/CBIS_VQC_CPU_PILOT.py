from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

LATENT_FILE = (
    ROOT
    / "experiments"
    / "cbis_core_vqc_pilot"
    / "SHARED_LATENTS.pt"
)

OUTPUT_ROOT = (
    ROOT
    / "experiments"
    / "cbis_vqc_cpu_pilot"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026
LATENT_DIM = 32

N_QUBITS = 6
DEPTH = 2

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# LOAD LATENTS
# ============================================================

def load_latents():

    if not LATENT_FILE.is_file():
        raise FileNotFoundError(
            f"Latent file not found: {LATENT_FILE}"
        )

    data = torch.load(
        LATENT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "train_z",
        "train_y",
        "train_patient_id",
        "calibration_z",
        "calibration_y",
        "calibration_patient_id",
        "internal_test_z",
        "internal_test_y",
        "internal_test_patient_id",
    }

    missing = required - set(data.keys())

    if missing:
        raise RuntimeError(
            f"Latent file missing keys: {sorted(missing)}"
        )

    return data


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS,
)


@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop",
)
def circuit(
    angles,
    weights,
):

    for q in range(N_QUBITS):

        qml.RY(
            angles[q],
            wires=q,
        )

    for d in range(DEPTH):

        for q in range(N_QUBITS):

            qml.RY(
                weights[d, q, 0],
                wires=q,
            )

            qml.RZ(
                weights[d, q, 1],
                wires=q,
            )

        for q in range(
            N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[
                    q,
                    q + 1,
                ]
            )

        qml.CNOT(
            wires=[
                N_QUBITS - 1,
                0,
            ]
        )

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# CPU-ONLY VQC
# ============================================================

class VQCCPU(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.angle_projection = nn.Linear(
            LATENT_DIM,
            N_QUBITS,
        )

        self.quantum_weights = nn.Parameter(
            0.01
            * torch.randn(
                DEPTH,
                N_QUBITS,
                2,
                dtype=torch.float32,
            )
        )

        self.readout = nn.Linear(
            1,
            1,
        )

    def forward(
        self,
        z,
    ):

        z = z.to(
            device="cpu",
            dtype=torch.float32,
        )

        projected = (
            self.angle_projection(
                z
            )
        )

        angles = (
            torch.pi
            * torch.tanh(
                projected
            )
        )

        outputs = []

        for i in range(
            z.shape[0]
        ):

            q_value = circuit(
                angles[i],
                self.quantum_weights,
            )

            outputs.append(
                q_value
            )

        q_values = torch.stack(
            outputs
        ).reshape(
            -1,
            1,
        ).to(
            dtype=self.readout.weight.dtype
        )

        logits = self.readout(
            q_values
        ).squeeze(
            1
        )

        return logits


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================
# PREDICTION
# ============================================================

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

        batch = z[
            start:end
        ]

        logits = model(
            batch
        )

        p = torch.sigmoid(
            logits
        )

        probabilities.extend(
            p.cpu().numpy().tolist()
        )

    return (
        y.cpu().numpy(),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print("CBIS-DDSM CPU-ONLY 6-QUBIT VQC PILOT")
    print("=" * 80)

    print(
        "Latent file:",
        LATENT_FILE,
    )

    print(
        "PennyLane:",
        qml.__version__,
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "Device: CPU"
    )

    data = load_latents()

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

    print()
    print(
        "TRAIN:",
        tuple(train_z.shape),
    )

    print(
        "CALIBRATION:",
        tuple(cal_z.shape),
    )

    print(
        "INTERNAL TEST:",
        tuple(test_z.shape),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VQCCPU()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Qubits:",
        N_QUBITS,
    )

    print(
        "Depth:",
        DEPTH,
    )

    print(
        "Trainable parameters:",
        total_parameters,
    )

    best_auc = -1.0
    best_epoch = -1

    checkpoint = (
        OUTPUT_ROOT
        / "best_vqc.pt"
    )

    history = []

    start_time = time.time()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        permutation = torch.randperm(
            train_z.shape[0]
        )

        total_loss = 0.0
        total_n = 0

        for pos in range(
            0,
            train_z.shape[0],
            BATCH_SIZE,
        ):

            idx = permutation[
                pos:
                pos + BATCH_SIZE
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

            n = yb.shape[0]

            total_loss += (
                loss.item()
                * n
            )

            total_n += n

        avg_loss = (
            total_loss
            / total_n
        )

        y_cal_np, p_cal = predict(
            model,
            cal_z,
            cal_y,
        )

        cal_auc = roc_auc_score(
            y_cal_np,
            p_cal,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(
                avg_loss
            ),
            "calibration_roc_auc": float(
                cal_auc
            ),
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

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    )

    y_cal_np, p_cal = predict(
        model,
        cal_z,
        cal_y,
    )

    y_test_np, p_test = predict(
        model,
        test_z,
        test_y,
    )

    cal_metrics = compute_metrics(
        y_cal_np,
        p_cal,
    )

    test_metrics = compute_metrics(
        y_test_np,
        p_test,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_file = (
        OUTPUT_ROOT
        / "VQC_PILOT_PREDICTIONS.csv"
    )

    with prediction_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "patient_id",
            "label",
            "probability",
        ])

        for pid, y, p in zip(
            data[
                "calibration_patient_id"
            ],
            y_cal_np,
            p_cal,
        ):

            writer.writerow([
                "calibration",
                pid,
                int(y),
                float(p),
            ])

        for pid, y, p in zip(
            data[
                "internal_test_patient_id"
            ],
            y_test_np,
            p_test,
        ):

            writer.writerow([
                "internal_test",
                pid,
                int(y),
                float(p),
            ])

    results = {
        "experiment":
            "CBIS-DDSM CPU-only 6-qubit VQC pilot",
        "seed":
            SEED,
        "latent_dim":
            LATENT_DIM,
        "qubits":
            N_QUBITS,
        "depth":
            DEPTH,
        "epochs":
            EPOCHS,
        "batch_size":
            BATCH_SIZE,
        "learning_rate":
            LEARNING_RATE,
        "trainable_parameters":
            total_parameters,
        "train_records":
            int(train_z.shape[0]),
        "calibration_records":
            int(cal_z.shape[0]),
        "internal_test_records":
            int(test_z.shape[0]),
        "best_calibration_auc":
            float(best_auc),
        "best_epoch":
            int(best_epoch),
        "calibration_metrics":
            cal_metrics,
        "internal_test_metrics":
            test_metrics,
        "training_seconds":
            float(
                time.time()
                - start_time
            ),
        "checkpoint":
            str(checkpoint),
        "predictions":
            str(prediction_file),
        "status":
            "PILOT_COMPLETE",
    }

    results_file = (
        OUTPUT_ROOT
        / "VQC_PILOT_RESULTS.json"
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
    print("VQC PILOT COMPLETE")
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
    print("STATUS: PILOT_COMPLETE")


if __name__ == "__main__":
    train()