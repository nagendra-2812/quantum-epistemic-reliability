from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
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

import pennylane as qml


# ================================================================
# ASUS-21 PILOT
# ================================================================
#
# Pilot:
#     Fold 01 / Seed 42
#
# Purpose:
#     Validate the frozen ASUS-21 implementation before the
#     automatic full experiment.
#
# Experimental arms:
#     1. softmax
#     2. temperature_scaled
#     3. mc_dropout
#     4. deep_ensemble
#     5. deterministic_vqc
#     6. laplace_vqc
#     7. laplace_mlp

#
# ================================================================


PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

CONFIG_FILE = (
    PROJECT
    / "configs"
    / "ASUS_21_TRAINING_SPECIFICATION_v1.json"
)

FEATURE_ROOT = (
    PROJECT
    / "features"
    / "BreaKHis"
    / "pca_6d"
)

OUTPUT_ROOT = (
    PROJECT
    / "features"
    / "BreaKHis"
    / "asus_21_pilot"
)

AUDIT_FILE = (
    PROJECT
    / "metadata"
    / "ASUS-21_PILOT_AUDIT_v1.json"
)

PILOT_FOLD = 1
PILOT_SEED = 42

# Classical models use CUDA when available.
CLASSICAL_DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# Exact statevector PennyLane default.qubit is CPU based.
VQC_DEVICE = torch.device("cpu")


# ================================================================
# Utility functions
# ================================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entropy(probability):

    probability = np.clip(
        probability,
        1e-7,
        1.0 - 1e-7,
    )

    return -(
        probability * np.log(probability)
        + (
            1.0 - probability
        )
        * np.log(
            1.0 - probability
        )
    )


def ensemble_mutual_information(
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    mean_probability = (
        probabilities.mean(axis=0)
    )

    return (
        entropy(mean_probability)
        - entropy(probabilities).mean(axis=0)
    )


def expected_calibration_error(
    y_true,
    probability,
    bins=10,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    probability = np.asarray(
        probability
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    value = 0.0

    total = len(y_true)

    for index in range(bins):

        if index == bins - 1:

            mask = (
                (probability >= edges[index])
                & (
                    probability
                    <= edges[index + 1]
                )
            )

        else:

            mask = (
                (probability >= edges[index])
                & (
                    probability
                    < edges[index + 1]
                )
            )

        if not np.any(mask):
            continue

        confidence = probability[mask].mean()

        accuracy = y_true[mask].mean()

        value += (
            mask.sum()
            / total
            * abs(
                confidence
                - accuracy
            )
        )

    return float(value)


def aurc(
    y_true,
    probability,
    uncertainty,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    probability = np.asarray(
        probability
    )

    uncertainty = np.asarray(
        uncertainty
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    error = (
        prediction != y_true
    ).astype(float)

    # High uncertainty = rejected first.
    order = np.argsort(
        -uncertainty,
        kind="stable",
    )

    error = error[order]

    n = len(error)

    if n == 0:
        return float("nan")

    cumulative_error = np.cumsum(
        error
    )

    coverage = (
        np.arange(1, n + 1)
        / n
    )

    risk = (
        cumulative_error
        / np.arange(1, n + 1)
    )

    return float(
        np.trapezoid(
            risk,
            coverage,
        )
    )


def compute_metrics(
    y_true,
    probability,
    uncertainty,
):

    probability = np.asarray(
        probability
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),

        "ece": expected_calibration_error(
            y_true,
            probability,
        ),

        "AURC": aurc(
            y_true,
            probability,
            uncertainty,
        ),
    }


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ================================================================
# Classical models
# ================================================================

class SoftmaxModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(
            6,
            1,
        )

    def forward(self, x):

        return self.linear(
            x
        ).squeeze(-1)


class MCDropoutModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                6,
                16,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(-1)


# ================================================================
# Exact ASUS-21 VQC
# ================================================================
#
# Input:
#     six PCA features
#
# Encoding:
#     RY(x_i)
#
# Trainable ansatz:
#     two layers
#     RY + RZ
#     six qubits
#     ring CNOT
#
# Trainable quantum parameters:
#
#     2 layers × 6 qubits × 2 rotations
#     = 24
#
# The classical readout has seven additional parameters.
#
# ================================================================

class DeterministicVQC(nn.Module):

    def __init__(self):

        super().__init__()

        self.n_qubits = 6

        # IMPORTANT:
        # The exact statevector simulator is CPU based.
        self.theta = nn.Parameter(
            torch.empty(
                2,
                6,
                2,
                dtype=torch.float64,
                device=VQC_DEVICE,
            )
        )

        nn.init.uniform_(
            self.theta,
            -0.1,
            0.1,
        )

        self.classifier = nn.Linear(
            6,
            1,
            dtype=torch.float64,
            device=VQC_DEVICE,
        )

        self.qml_device = qml.device(
            "default.qubit",
            wires=self.n_qubits,
        )

        @qml.qnode(
            self.qml_device,
            interface="torch",
            diff_method="backprop",
        )
        def circuit(
            inputs,
            theta,
        ):

            for qubit in range(6):

                qml.RY(
                    inputs[qubit],
                    wires=qubit,
                )

            for layer in range(2):

                for qubit in range(6):

                    qml.RY(
                        theta[
                            layer,
                            qubit,
                            0,
                        ],
                        wires=qubit,
                    )

                    qml.RZ(
                        theta[
                            layer,
                            qubit,
                            1,
                        ],
                        wires=qubit,
                    )

                # Ring CNOT
                for qubit in range(6):

                    qml.CNOT(
                        wires=[
                            qubit,
                            (
                                qubit + 1
                            ) % 6,
                        ]
                    )

            return [
                qml.expval(
                    qml.PauliZ(
                        qubit
                    )
                )
                for qubit in range(6)
            ]

        self.circuit = circuit

    def forward(self, x):

        # VQC remains entirely on CPU.
        x = x.to(
            VQC_DEVICE,
            dtype=torch.float64,
        )

        values = []

        for sample in x:

            expectations = self.circuit(
                sample,
                self.theta,
            )

            expectations = torch.stack(
                expectations
            )

            values.append(
                expectations
            )

        values = torch.stack(
            values
        )

        return self.classifier(
            values
        ).squeeze(-1)


# ================================================================
# Data loading
# ================================================================

def load_input():

    feature_file = (
        FEATURE_ROOT
        / f"fold_{PILOT_FOLD:02d}"
        / f"seed_{PILOT_SEED}"
        / "BreaKHis_FOLD_FEATURES_v1.npz"
    )

    if not feature_file.exists():

        raise FileNotFoundError(
            "Missing ASUS-20B input:\n"
            f"{feature_file}"
        )

    data = np.load(
        feature_file,
        allow_pickle=False,
    )

    required_keys = [

        "train_features_6",
        "train_label",

        "val_features_6",
        "val_label",

        "test_features_6",
        "test_label",

        "train_patient_id",
        "val_patient_id",
        "test_patient_id",
    ]

    for key in required_keys:

        if key not in data.files:

            raise RuntimeError(
                f"Missing required key: {key}"
            )

    return feature_file, data


# ================================================================
# Generic classical training
# ================================================================

def train_classical_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
):

    model = model.to(
        CLASSICAL_DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_x = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    val_x = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    val_y = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None

    n_samples = len(
        train_x
    )

    generator = torch.Generator()

    generator.manual_seed(
        int(
            torch.initial_seed()
        )
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        permutation = torch.randperm(
            n_samples,
            generator=generator,
        )

        for start in range(
            0,
            n_samples,
            batch_size,
        ):

            indices = permutation[
                start:start + batch_size
            ]

            batch_x = train_x[
                indices
            ]

            batch_y = train_y[
                indices
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(
                val_x
            )

            val_probability = (
                torch.sigmoid(
                    val_logits
                )
                .detach()
                .cpu()
                .numpy()
            )

        val_auc = roc_auc_score(
            y_val,
            val_probability,
        )

        if val_auc > best_auc:

            best_auc = float(
                val_auc
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:

        raise RuntimeError(
            "No best model state was generated."
        )

    model.load_state_dict(
        best_state
    )

    return model, {
        "best_validation_roc_auc":
            best_auc,
        "best_epoch":
            best_epoch,
    }


# ================================================================
# VQC training
# ================================================================

def train_vqc_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
):

    # Exact statevector VQC is CPU-only.
    model = model.to(
        VQC_DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_x = torch.tensor(
        X_train,
        dtype=torch.float64,
        device=VQC_DEVICE,
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float64,
        device=VQC_DEVICE,
    )

    val_x = torch.tensor(
        X_val,
        dtype=torch.float64,
        device=VQC_DEVICE,
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None

    n_samples = len(
        train_x
    )

    generator = torch.Generator()

    generator.manual_seed(
        PILOT_SEED
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        permutation = torch.randperm(
            n_samples,
            generator=generator,
        )

        for start in range(
            0,
            n_samples,
            batch_size,
        ):

            indices = permutation[
                start:start + batch_size
            ]

            batch_x = train_x[
                indices
            ]

            batch_y = train_y[
                indices
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(
                val_x
            )

            val_probability = (
                torch.sigmoid(
                    val_logits
                )
                .detach()
                .cpu()
                .numpy()
            )

        val_auc = roc_auc_score(
            y_val,
            val_probability,
        )

        if val_auc > best_auc:

            best_auc = float(
                val_auc
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

        print(
            f"    epoch {epoch:02d}/{epochs} "
            f"| val ROC-AUC={val_auc:.6f}"
        )

    if best_state is None:

        raise RuntimeError(
            "VQC produced no best state."
        )

    model.load_state_dict(
        best_state
    )

    return model, {
        "best_validation_roc_auc":
            best_auc,
        "best_epoch":
            best_epoch,
    }


# ================================================================
# Prediction
# ================================================================

def predict_classical(
    model,
    X,
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    return probability


def predict_vqc(
    model,
    X,
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float64,
        device=VQC_DEVICE,
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    return probability


def predict_mc_dropout(
    model,
    X,
    passes,
):

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    predictions = []

    model.train()

    with torch.no_grad():

        for _ in range(
            passes
        ):

            logits = model(
                tensor
            )

            predictions.append(
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )

    return np.stack(
        predictions,
        axis=0,
    )




# ================================================================
# ASUS-21 LAPLACE IMPLEMENTATION
# ================================================================
# Protocol implemented from the project specification:
#   F_j = mean_i[(d L_i / d theta_j)^2]
#   lambda = 1e-4 * mean(F)
#   Sigma = alpha^2 * diag(F + lambda)^(-1)
#   alpha in {0.25, 0.50, 1.00, 2.00}, selected by validation NLL
#   30 posterior samples
#
# Laplace-VQC: perturb the 24 quantum parameters theta only;
#              keep the 7-parameter classical readout fixed.
# Laplace-MLP: perturb all 25 trainable parameters.
# ================================================================

LAPLACE_ALPHA_GRID = (0.25, 0.50, 1.00, 2.00)
LAPLACE_POSTERIOR_SAMPLES = 30
LAPLACE_DAMPING_FACTOR = 1.0e-4


def _laplace_parameters(model, vqc=False):
    if vqc:
        return [model.theta]
    return [p for p in model.parameters() if p.requires_grad]


def _laplace_parameter_count(model, vqc=False):
    return sum(p.numel() for p in _laplace_parameters(model, vqc))


def _laplace_flatten(model, vqc=False, device=None):
    parameters = _laplace_parameters(model, vqc)
    if device is None:
        device = parameters[0].device
    return torch.cat([
        p.detach().to(device=device, dtype=torch.float64).reshape(-1)
        for p in parameters
    ])


def _laplace_restore(model, flat, vqc=False):
    offset = 0
    with torch.no_grad():
        for p in _laplace_parameters(model, vqc):
            n = p.numel()
            p.copy_(flat[offset:offset+n].reshape(p.shape).to(device=p.device, dtype=p.dtype))
            offset += n


def _laplace_empirical_fisher(model, X, y, vqc=False):
    device = VQC_DEVICE if vqc else CLASSICAL_DEVICE
    dtype = torch.float64 if vqc else torch.float32
    X_tensor = torch.tensor(X, dtype=dtype, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    parameters = _laplace_parameters(model, vqc)
    fisher = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64, device=device)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total = len(X_tensor)
    for i in range(total):
        model.zero_grad(set_to_none=True)
        logits = model(X_tensor[i:i+1])
        loss = criterion(logits.reshape(-1), y_tensor[i:i+1].reshape(-1))
        grads = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False, allow_unused=False)
        flat_grad = torch.cat([g.detach().to(torch.float64).reshape(-1) for g in grads])
        fisher += flat_grad.square()
        if (i + 1) % 250 == 0 or i + 1 == total:
            print(f"    empirical Fisher {i + 1}/{total}")

    fisher /= float(total)
    return fisher


def _laplace_nll(y_true, probability):
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_true*np.log(probability) + (1.0-y_true)*np.log(1.0-probability)))


def _laplace_sample_parameters(mean, fisher, alpha, damping, generator):
    noise = torch.randn(mean.shape, dtype=torch.float64, device=mean.device, generator=generator)
    std = float(alpha) / torch.sqrt(fisher + float(damping))
    return mean + noise * std


def _laplace_predict_single(model, X, mean, fisher, alpha, damping, generator, vqc=False):
    sampled = _laplace_sample_parameters(mean, fisher, alpha, damping, generator)
    _laplace_restore(model, sampled, vqc=vqc)
    probability = predict_vqc(model, X) if vqc else predict_classical(model, X)
    _laplace_restore(model, mean, vqc=vqc)
    return probability


def _laplace_predictive_samples(model, X, mean, fisher, alpha, damping, generator, vqc=False):
    samples = []
    for _ in range(LAPLACE_POSTERIOR_SAMPLES):
        samples.append(_laplace_predict_single(model, X, mean, fisher, alpha, damping, generator, vqc=vqc))
    return np.stack(samples, axis=0)


def _laplace_select_alpha(model, X_val, y_val, mean, fisher, damping, seed, vqc=False):
    device = VQC_DEVICE if vqc else CLASSICAL_DEVICE
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + (41000 if vqc else 42000))
    scores = {}

    for alpha in LAPLACE_ALPHA_GRID:
        val_samples = _laplace_predictive_samples(
            model, X_val, mean, fisher, alpha, damping, generator, vqc=vqc
        )
        mean_probability = val_samples.mean(axis=0)
        score = _laplace_nll(y_val, mean_probability)
        scores[float(alpha)] = float(score)
        print(f"    alpha={alpha:.2f} | validation NLL={score:.8f}")

    selected = min(scores, key=scores.get)
    return float(selected), scores


def _run_laplace_arm(model, arm_name, X_train, y_train, X_val, y_val, X_test, y_test,
                     epochs, batch_size, learning_rate, weight_decay, seed, vqc=False):
    print()
    print("=" * 100)
    print(f"ARM — {arm_name.upper()}")
    print("=" * 100)

    set_seed(seed)
    if vqc:
        model, training_info = train_vqc_model(model, X_train, y_train, X_val, y_val,
                                               epochs, batch_size, learning_rate, weight_decay)
    else:
        model, training_info = train_classical_model(model, X_train, y_train, X_val, y_val,
                                                     epochs, batch_size, learning_rate, weight_decay)

    laplace_count = _laplace_parameter_count(model, vqc=vqc)
    expected = 24 if vqc else 25
    if laplace_count != expected:
        raise RuntimeError(f"{arm_name}: expected {expected} Laplace parameters, got {laplace_count}.")

    print("Laplace parameters:", laplace_count)
    print("Estimating diagonal empirical Fisher from training data only...")
    fisher = _laplace_empirical_fisher(model, X_train, y_train, vqc=vqc)
    mean_fisher = max(float(fisher.mean().detach().cpu().item()), 1e-12)
    damping = LAPLACE_DAMPING_FACTOR * mean_fisher

    device = VQC_DEVICE if vqc else CLASSICAL_DEVICE
    mean = _laplace_flatten(model, vqc=vqc, device=device)
    fisher = fisher.to(device=device, dtype=torch.float64)

    selected_alpha, alpha_scores = _laplace_select_alpha(
        model, X_val, y_val, mean, fisher, damping, seed, vqc=vqc
    )
    print("Selected alpha:", selected_alpha)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + (51000 if vqc else 52000))
    test_samples = _laplace_predictive_samples(
        model, X_test, mean, fisher, selected_alpha, damping, generator, vqc=vqc
    )
    probability = test_samples.mean(axis=0)
    uncertainty = test_samples.var(axis=0, ddof=0)
    metrics = compute_metrics(y_test, probability, uncertainty)

    result = {
        "arm": arm_name,
        "status": "EXECUTED",
        "laplace_trainable_parameters": laplace_count,
        "training": training_info,
        "laplace": {
            "fisher": "diagonal empirical Fisher",
            "fisher_estimation": "per-sample squared training-loss gradients",
            "fisher_split": "training only",
            "damping_formula": "1e-4 * mean(F)",
            "damping": float(damping),
            "alpha_grid": list(LAPLACE_ALPHA_GRID),
            "selected_alpha": float(selected_alpha),
            "validation_nll": {str(k): float(v) for k, v in alpha_scores.items()},
            "posterior_samples": LAPLACE_POSTERIOR_SAMPLES,
            "covariance": "alpha^2 * diag(F + lambda)^(-1)",
            "uncertainty": "posterior predictive variance",
            "parameter_scope": "quantum theta only" if vqc else "all trainable MLP parameters",
        },
        "metrics": metrics,
    }

    print("Posterior samples:", LAPLACE_POSTERIOR_SAMPLES)
    print("Test AURC:", metrics["AURC"])
    return result


class LaplaceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch, 6), got {tuple(x.shape)}")
        if x.shape[1] != 6:
            raise ValueError(f"Expected exactly 6 input features, got {x.shape[1]}")
        return self.network(x).squeeze(-1)


# ================================================================
# Main
# ================================================================

def main():

    print("=" * 100)
    print(
        "ASUS-21 — PILOT EXECUTION"
    )
    print("=" * 100)

    print()
    print("PILOT")
    print("-" * 100)

    print(
        f"Fold: {PILOT_FOLD:02d}"
    )

    print(
        f"Seed: {PILOT_SEED}"
    )

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    print()
    print("CONFIGURATION")
    print("-" * 100)

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            CONFIG_FILE
        )

    config_hash = sha256_file(
        CONFIG_FILE
    )

    config = json.loads(
        CONFIG_FILE.read_text(
            encoding="utf-8-sig"
        )
    )

    if config.get(
        "step"
    ) != "ASUS-21":

        raise RuntimeError(
            "Configuration does not identify "
            "itself as ASUS-21."
        )

    if config.get(
        "status"
    ) != "TRAINING_PROTOCOL_FROZEN":

        raise RuntimeError(
            "ASUS-21 configuration is not frozen."
        )

    print(
        "Configuration:",
        CONFIG_FILE,
    )

    print(
        "Configuration SHA256:",
        config_hash,
    )

    # ------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------

    training = config[
        "training"
    ]

    epochs = int(
        training["epochs"]
    )

    batch_size = int(
        training["batch_size"]
    )

    learning_rate = float(
        training["learning_rate"]
    )

    weight_decay = float(
        training["weight_decay"]
    )

    selection_metric = (
        training["selection_metric"]
    )

    print()
    print("TRAINING")
    print("-" * 100)

    print(
        "Epochs:",
        epochs,
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Learning rate:",
        learning_rate,
    )

    print(
        "Weight decay:",
        weight_decay,
    )

    print(
        "Selection metric:",
        selection_metric,
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    print()
    print("ENVIRONMENT")
    print("-" * 100)

    print(
        "Python:",
        sys.version.split()[0],
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "PennyLane:",
        qml.__version__,
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "Classical device:",
        CLASSICAL_DEVICE,
    )

    print(
        "VQC device:",
        VQC_DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------------
    # Load ASUS-20B artifact
    # ------------------------------------------------------------

    print()
    print("ASUS-20B INPUT")
    print("-" * 100)

    feature_file, data = load_input()

    print(
        "Input:",
        feature_file,
    )

    print(
        "Input SHA256:",
        sha256_file(
            feature_file
        ),
    )

    X_train = data[
        "train_features_6"
    ]

    y_train = data[
        "train_label"
    ]

    X_val = data[
        "val_features_6"
    ]

    y_val = data[
        "val_label"
    ]

    X_test = data[
        "test_features_6"
    ]

    y_test = data[
        "test_label"
    ]

    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
    )

    # ------------------------------------------------------------
    # Verify labels
    # ------------------------------------------------------------

    unique_train = np.unique(
        y_train
    )

    unique_val = np.unique(
        y_val
    )

    unique_test = np.unique(
        y_test
    )

    print(
        "Train labels:",
        unique_train.tolist(),
    )

    print(
        "Validation labels:",
        unique_val.tolist(),
    )

    print(
        "Test labels:",
        unique_test.tolist(),
    )

    # ------------------------------------------------------------
    # Output protection
    # ------------------------------------------------------------

    if OUTPUT_ROOT.exists():

        existing = list(
            OUTPUT_ROOT.rglob("*")
        )

        if existing:

            raise FileExistsError(
                "Output already exists. "
                "Overwrite protection is active:\n"
                f"{OUTPUT_ROOT}"
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------

    results = []

    # ============================================================
    # 1. SOFTMAX
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ARM 1/7 — SOFTMAX"
    )
    print("=" * 100)

    set_seed(
        PILOT_SEED
    )

    model = SoftmaxModel()

    parameter_count = count_parameters(
        model
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    if parameter_count != 7:

        raise RuntimeError(
            "Softmax must have 7 parameters."
        )

    model, training_info = (
        train_classical_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
        )
    )

    probability = predict_classical(
        model,
        X_test,
    )

    uncertainty = entropy(
        probability
    )

    metrics = compute_metrics(
        y_test,
        probability,
        uncertainty,
    )

    results.append({

        "arm":
            "softmax",

        "status":
            "EXECUTED",

        "trainable_parameters":
            parameter_count,

        "training":
            training_info,

        "metrics":
            metrics,
    })

    print(
        "Best validation ROC-AUC:",
        training_info[
            "best_validation_roc_auc"
        ],
    )

    print(
        "Test AURC:",
        metrics["AURC"],
    )

    # ============================================================
    # 2. TEMPERATURE SCALING
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ARM 2/7 — TEMPERATURE SCALED"
    )
    print("=" * 100)

    set_seed(
        PILOT_SEED
    )

    model = SoftmaxModel()

    model, training_info = (
        train_classical_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
        )
    )

    model.eval()

    val_x = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    val_y_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    temperature = nn.Parameter(
        torch.tensor(
            1.0,
            dtype=torch.float32,
            device=CLASSICAL_DEVICE,
        )
    )

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.1,
        max_iter=50,
    )

    criterion = nn.BCEWithLogitsLoss()

    def closure():

        optimizer.zero_grad()

        logits = model(
            val_x
        )

        scaled_logits = (
            logits
            / torch.clamp(
                temperature,
                min=0.05,
            )
        )

        loss = criterion(
            scaled_logits,
            val_y_tensor,
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    fitted_temperature = float(
        torch.clamp(
            temperature.detach(),
            min=0.05,
        )
        .cpu()
        .item()
    )

    test_x = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=CLASSICAL_DEVICE,
    )

    with torch.no_grad():

        logits = model(
            test_x
        )

        probability = (
            torch.sigmoid(
                logits
                / fitted_temperature
            )
            .cpu()
            .numpy()
        )

    uncertainty = entropy(
        probability
    )

    metrics = compute_metrics(
        y_test,
        probability,
        uncertainty,
    )

    results.append({

        "arm":
            "temperature_scaled",

        "status":
            "EXECUTED",

        "base_model_trainable_parameters":
            count_parameters(model),

        "temperature":
            fitted_temperature,

        "training":
            training_info,

        "metrics":
            metrics,
    })

    print(
        "Temperature:",
        fitted_temperature,
    )

    print(
        "Test AURC:",
        metrics["AURC"],
    )

    # ============================================================
    # 3. MC DROPOUT
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ARM 3/7 — MC DROPOUT"
    )
    print("=" * 100)

    set_seed(
        PILOT_SEED
    )

    model = MCDropoutModel()

    parameter_count = count_parameters(
        model
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    if parameter_count != 129:

        raise RuntimeError(
            "MC-Dropout must have 129 parameters."
        )

    model, training_info = (
        train_classical_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
        )
    )

    mc_probabilities = (
        predict_mc_dropout(
            model,
            X_test,
            30,
        )
    )

    probability = (
        mc_probabilities.mean(
            axis=0
        )
    )

    uncertainty = (
        ensemble_mutual_information(
            mc_probabilities
        )
    )

    metrics = compute_metrics(
        y_test,
        probability,
        uncertainty,
    )

    results.append({

        "arm":
            "mc_dropout",

        "status":
            "EXECUTED",

        "trainable_parameters":
            parameter_count,

        "passes":
            30,

        "training":
            training_info,

        "metrics":
            metrics,
    })

    print(
        "Passes:",
        30,
    )

    print(
        "Test AURC:",
        metrics["AURC"],
    )

    # ============================================================
    # 4. DEEP ENSEMBLE
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ARM 4/7 — DEEP ENSEMBLE"
    )
    print("=" * 100)

    ensemble_probabilities = []

    ensemble_training = []

    for member in range(5):

        member_seed = (
            PILOT_SEED
            + member
        )

        set_seed(
            member_seed
        )

        member_model = SoftmaxModel()

        parameter_count = count_parameters(
            member_model
        )

        if parameter_count != 7:

            raise RuntimeError(
                "Deep ensemble member must "
                "have 7 parameters."
            )

        member_model, info = (
            train_classical_model(
                member_model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
            )
        )

        member_probability = (
            predict_classical(
                member_model,
                X_test,
            )
        )

        ensemble_probabilities.append(
            member_probability
        )

        ensemble_training.append(
            info
        )

        print(
            f"Member {member + 1}/5 "
            f"| best val ROC-AUC="
            f"{info['best_validation_roc_auc']:.6f}"
        )

    ensemble_probabilities = np.stack(
        ensemble_probabilities,
        axis=0,
    )

    probability = (
        ensemble_probabilities.mean(
            axis=0
        )
    )

    uncertainty = (
        ensemble_mutual_information(
            ensemble_probabilities
        )
    )

    metrics = compute_metrics(
        y_test,
        probability,
        uncertainty,
    )

    results.append({

        "arm":
            "deep_ensemble",

        "status":
            "EXECUTED",

        "members":
            5,

        "member_trainable_parameters":
            7,

        "training":
            ensemble_training,

        "metrics":
            metrics,
    })

    print(
        "Members:",
        5,
    )

    print(
        "Test AURC:",
        metrics["AURC"],
    )

    # ============================================================
    # 5. DETERMINISTIC VQC
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ARM 5/7 — DETERMINISTIC VQC"
    )
    print("=" * 100)

    set_seed(
        PILOT_SEED
    )

    model = DeterministicVQC()

    quantum_parameter_count = int(
        model.theta.numel()
    )

    hybrid_parameter_count = (
        count_parameters(model)
    )

    print(
        "Quantum trainable parameters:",
        quantum_parameter_count,
    )

    print(
        "Hybrid model trainable parameters:",
        hybrid_parameter_count,
    )

    if quantum_parameter_count != 24:

        raise RuntimeError(
            "VQC quantum parameter count "
            "must be exactly 24."
        )

    if hybrid_parameter_count != 31:

        raise RuntimeError(
            "VQC hybrid model should contain "
            "24 quantum + 7 classifier parameters."
        )

    print(
        "VQC execution device:",
        VQC_DEVICE,
    )

    model, training_info = train_vqc_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
    )

    probability = predict_vqc(
        model,
        X_test,
    )

    uncertainty = entropy(
        probability
    )

    metrics = compute_metrics(
        y_test,
        probability,
        uncertainty,
    )

    results.append({

        "arm":
            "deterministic_vqc",

        "status":
            "EXECUTED",

        "quantum_trainable_parameters":
            quantum_parameter_count,

        "classifier_trainable_parameters":
            7,

        "hybrid_trainable_parameters":
            hybrid_parameter_count,

        "simulation":
            "exact statevector",

        "training":
            training_info,

        "metrics":
            metrics,
    })

    print(
        "Best validation ROC-AUC:",
        training_info[
            "best_validation_roc_auc"
        ],
    )

    print(
        "Test AURC:",
        metrics["AURC"],
    )

    # ============================================================
    # 6. LAPLACE VQC
    # ============================================================

    laplace_vqc_result = _run_laplace_arm(
        model=DeterministicVQC(),
        arm_name="laplace_vqc",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=PILOT_SEED,
        vqc=True,
    )

    results.append(laplace_vqc_result)

    # ============================================================
    # 7. LAPLACE MLP
    # ============================================================

    laplace_mlp_result = _run_laplace_arm(
        model=LaplaceMLP(),
        arm_name="laplace_mlp",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=PILOT_SEED,
        vqc=False,
    )

    results.append(laplace_mlp_result)

    # ============================================================
    # Save results
    # ============================================================

    results_file = (
        OUTPUT_ROOT
        / "PILOT_RESULTS.json"
    )

    payload = {

        "step":
            "ASUS-21",

        "pilot":
            True,

        "fold":
            PILOT_FOLD,

        "seed":
            PILOT_SEED,

        "configuration_sha256":
            config_hash,

        "input_feature_sha256":
            sha256_file(
                feature_file
            ),

        "classical_device":
            str(
                CLASSICAL_DEVICE
            ),

        "vqc_device":
            str(
                VQC_DEVICE
            ),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "executed_arms": [
            "softmax",
            "temperature_scaled",
            "mc_dropout",
            "deep_ensemble",
            "deterministic_vqc",
            "laplace_vqc",
            "laplace_mlp",
        ],

        "blocked_arms": [],

        "results":
            results,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    results_file.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Audit
    # ============================================================

    audit = {

        "step":
            "ASUS-21",

        "status":
            "PILOT_COMPLETED_ALL_7_ARMS",

        "project":
            "quantum-epistemic-reliability",

        "dataset":
            "BreaKHis",

        "fold":
            PILOT_FOLD,

        "seed":
            PILOT_SEED,

        "configuration_sha256":
            config_hash,

        "input_feature_sha256":
            sha256_file(
                feature_file
            ),

        "train_shape":
            list(
                X_train.shape
            ),

        "validation_shape":
            list(
                X_val.shape
            ),

        "test_shape":
            list(
                X_test.shape
            ),

        "training": {

            "epochs":
                epochs,

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "weight_decay":
                weight_decay,

            "selection_metric":
                selection_metric,

            "test_used_for_selection":
                False,
        },

        "executed_arms": [
            "softmax",
            "temperature_scaled",
            "mc_dropout",
            "deep_ensemble",
            "deterministic_vqc",
            "laplace_vqc",
            "laplace_mlp",
        ],

        "blocked_arms": [],

        "results_file":
            str(
                results_file.relative_to(
                    PROJECT
                )
            ),

        "results_sha256":
            sha256_file(
                results_file
            ),

        "full_experiment_allowed":
            True,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Final report
    # ============================================================

    print()
    print("=" * 100)
    print(
        "ASUS-21 PILOT COMPLETE"
    )
    print("=" * 100)

    print(
        "Executed arms: 7"
    )

    print(
        "Blocked arms: 0"
    )

    print()
    print(
        "Results:",
        results_file,
    )

    print(
        "Audit:",
        AUDIT_FILE,
    )

    print()
    print(
        "FULL EXPERIMENT ALLOWED: TRUE"
    )

    print(
        "All seven ASUS-21 experimental arms completed."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()