from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pennylane as qml
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# FIXED EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

LATENT_FILE = (
    ROOT
    / "experiments"
    / "cbis_core_vqc_pilot"
    / "SHARED_LATENTS.pt"
)

VQC_CHECKPOINT = (
    ROOT
    / "experiments"
    / "cbis_vqc_cpu_pilot"
    / "best_vqc.pt"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_phase7_quantum_uncertainty"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026

N_QUBITS = 6
DEPTH = 2
LATENT_DIM = 32

SHOT_COUNTS = (
    100,
    500,
    1000,
)

SHOT_BATCHES = 10

PARAMETER_ENSEMBLE_SIZE = 20

# Local perturbation scale.
# This is deliberately recorded as part of the protocol.
PARAMETER_SIGMA_FRACTION = 0.05


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def entropy_binary(p):

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


def safe_auc(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_auprc(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y,
            score,
        )
    )


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def make_device(
    shots=None,
    seed=None,
):

    kwargs = {
        "wires": N_QUBITS,
    }

    if shots is not None:
        kwargs["shots"] = shots

    if seed is not None:
        kwargs["seed"] = seed

    return qml.device(
        "default.qubit",
        **kwargs,
    )


def build_circuit(
    device,
):

    @qml.qnode(
        device,
        interface=None,
    )
    def circuit(
        angles,
        weights,
    ):

        for q in range(
            N_QUBITS
        ):

            qml.RY(
                float(angles[q]),
                wires=q,
            )

        for d in range(
            DEPTH
        ):

            for q in range(
                N_QUBITS
            ):

                qml.RY(
                    float(
                        weights[d, q, 0]
                    ),
                    wires=q,
                )

                qml.RZ(
                    float(
                        weights[d, q, 1]
                    ),
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

    return circuit


# ============================================================
# LOAD MODEL PARAMETERS
# ============================================================

def load_vqc():

    state = torch.load(
        VQC_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "quantum_weights",
        "angle_projection.weight",
        "angle_projection.bias",
        "readout.weight",
        "readout.bias",
    }

    missing = (
        required
        - set(state.keys())
    )

    if missing:
        raise RuntimeError(
            f"VQC checkpoint missing keys: {sorted(missing)}"
        )

    angle_weight = state[
        "angle_projection.weight"
    ].detach().cpu().numpy().astype(
        np.float64
    )

    angle_bias = state[
        "angle_projection.bias"
    ].detach().cpu().numpy().astype(
        np.float64
    )

    quantum_weights = state[
        "quantum_weights"
    ].detach().cpu().numpy().astype(
        np.float64
    )

    readout_weight = float(
        state[
            "readout.weight"
        ].detach().cpu().numpy().reshape(-1)[0]
    )

    readout_bias = float(
        state[
            "readout.bias"
        ].detach().cpu().numpy().reshape(-1)[0]
    )

    return {
        "angle_weight":
            angle_weight,

        "angle_bias":
            angle_bias,

        "quantum_weights":
            quantum_weights,

        "readout_weight":
            readout_weight,

        "readout_bias":
            readout_bias,
    }


# ============================================================
# CLASSICAL → QUANTUM ANGLES
# ============================================================

def latent_to_angles(
    z,
    angle_weight,
    angle_bias,
):

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    projected = (
        z
        @ angle_weight.T
        + angle_bias
    )

    return (
        np.pi
        * np.tanh(
            projected
        )
    )


# ============================================================
# QUANTUM EXPECTATION → PROBABILITY
# ============================================================

def expectation_to_probability(
    expectation,
    readout_weight,
    readout_bias,
):

    logit = (
        readout_weight
        * expectation
        + readout_bias
    )

    return float(
        sigmoid(logit)
    )


# ============================================================
# IDEAL PREDICTION
# ============================================================

def ideal_probability(
    angles,
    weights,
    readout_weight,
    readout_bias,
):

    device = make_device()

    circuit = build_circuit(
        device
    )

    expectation = float(
        circuit(
            angles,
            weights,
        )
    )

    probability = (
        expectation_to_probability(
            expectation,
            readout_weight,
            readout_bias,
        )
    )

    return (
        expectation,
        probability,
    )


# ============================================================
# SHOT-BATCH PREDICTIONS
# ============================================================

def shot_probability(
    angles,
    weights,
    readout_weight,
    readout_bias,
    shots,
    seed,
):

    device = make_device(
        shots=shots,
        seed=seed,
    )

    circuit = build_circuit(
        device
    )

    expectation = float(
        circuit(
            angles,
            weights,
        )
    )

    probability = (
        expectation_to_probability(
            expectation,
            readout_weight,
            readout_bias,
        )
    )

    return (
        expectation,
        probability,
    )


# ============================================================
# PARAMETER-PERTURBATION ENSEMBLE
# ============================================================

def make_parameter_perturbation(
    base,
    rng,
):

    std = float(
        np.std(base)
    )

    if (
        not np.isfinite(std)
        or std == 0.0
    ):
        std = 1.0

    sigma = (
        PARAMETER_SIGMA_FRACTION
        * std
    )

    return (
        base
        + rng.normal(
            0.0,
            sigma,
            size=base.shape,
        )
    )


def build_perturbed_parameters(
    base,
    rng,
):

    return {
        "angle_weight":
            make_parameter_perturbation(
                base["angle_weight"],
                rng,
            ),

        "angle_bias":
            make_parameter_perturbation(
                base["angle_bias"],
                rng,
            ),

        "quantum_weights":
            make_parameter_perturbation(
                base["quantum_weights"],
                rng,
            ),

        "readout_weight":
            float(
                make_parameter_perturbation(
                    np.asarray(
                        [base["readout_weight"]]
                    ),
                    rng,
                )[0]
            ),

        "readout_bias":
            float(
                make_parameter_perturbation(
                    np.asarray(
                        [base["readout_bias"]]
                    ),
                    rng,
                )[0]
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print(
        "CBIS-DDSM PHASE 7 — QUANTUM UNCERTAINTY"
    )
    print("=" * 80)

    data = torch.load(
        LATENT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    z_test = (
        data[
            "internal_test_z"
        ]
        .float()
        .numpy()
    )

    y_test = (
        data[
            "internal_test_y"
        ]
        .numpy()
        .astype(int)
    )

    patient_ids = list(
        data[
            "internal_test_patient_id"
        ]
    )

    n = len(z_test)

    if n != 602:
        raise RuntimeError(
            f"Expected 602 internal-test records, got {n}"
        )

    print(
        "Internal-test records:",
        n,
    )

    base = load_vqc()

    # --------------------------------------------------------
    # Parameter count audit
    # --------------------------------------------------------

    parameter_count = (
        base[
            "angle_weight"
        ].size
        + base[
            "angle_bias"
        ].size
        + base[
            "quantum_weights"
        ].size
        + 1
        + 1
    )

    print(
        "VQC trainable parameters:",
        parameter_count,
    )

    if parameter_count != 224:
        raise RuntimeError(
            f"Expected 224 VQC parameters, got {parameter_count}"
        )

    # --------------------------------------------------------
    # Prepare angles
    # --------------------------------------------------------

    angles = latent_to_angles(
        z_test,
        base["angle_weight"],
        base["angle_bias"],
    )

    predictions = []

    # --------------------------------------------------------
    # Process every internal-test record
    # --------------------------------------------------------

    for i in range(n):

        if (
            i % 25 == 0
            or i == n - 1
        ):

            print(
                f"Processing {i + 1}/{n}",
                flush=True,
            )

        angle = angles[i]

        # ----------------------------------------------------
        # Ideal prediction
        # ----------------------------------------------------

        ideal_exp, ideal_p = (
            ideal_probability(
                angle,
                base[
                    "quantum_weights"
                ],
                base[
                    "readout_weight"
                ],
                base[
                    "readout_bias"
                ],
            )
        )

        # ----------------------------------------------------
        # Shot distributions
        # ----------------------------------------------------

        shot_record = {}

        for shots in SHOT_COUNTS:

            probabilities = []

            expectations = []

            for batch in range(
                SHOT_BATCHES
            ):

                seed = (
                    SEED
                    + i * 1000
                    + shots
                    + batch
                )

                exp_value, probability = (
                    shot_probability(
                        angle,
                        base[
                            "quantum_weights"
                        ],
                        base[
                            "readout_weight"
                        ],
                        base[
                            "readout_bias"
                        ],
                        shots,
                        seed,
                    )
                )

                expectations.append(
                    exp_value
                )

                probabilities.append(
                    probability
                )

            probabilities = np.asarray(
                probabilities,
                dtype=np.float64,
            )

            expectations = np.asarray(
                expectations,
                dtype=np.float64,
            )

            shot_record[
                str(shots)
            ] = {
                "mean_probability":
                    float(
                        probabilities.mean()
                    ),

                "probability_variance":
                    float(
                        probabilities.var(
                            ddof=1
                        )
                    ),

                "probability_std":
                    float(
                        probabilities.std(
                            ddof=1
                        )
                    ),

                "mean_expectation":
                    float(
                        expectations.mean()
                    ),

                "expectation_variance":
                    float(
                        expectations.var(
                            ddof=1
                        )
                    ),

                "predictive_entropy":
                    float(
                        entropy_binary(
                            probabilities.mean()
                        )
                    ),
            }

        # ----------------------------------------------------
        # Parameter-perturbation ensemble
        # ----------------------------------------------------

        rng = np.random.default_rng(
            SEED + i
        )

        ensemble_probabilities = []

        for b in range(
            PARAMETER_ENSEMBLE_SIZE
        ):

            perturbed = (
                build_perturbed_parameters(
                    base,
                    rng,
                )
            )

            perturbed_angles = (
                latent_to_angles(
                    z_test[i:i + 1],
                    perturbed[
                        "angle_weight"
                    ],
                    perturbed[
                        "angle_bias"
                    ],
                )[0]
            )

            _, p = ideal_probability(
                perturbed_angles,
                perturbed[
                    "quantum_weights"
                ],
                perturbed[
                    "readout_weight"
                ],
                perturbed[
                    "readout_bias"
                ],
            )

            ensemble_probabilities.append(
                p
            )

        ensemble_probabilities = np.asarray(
            ensemble_probabilities,
            dtype=np.float64,
        )

        mean_p = float(
            ensemble_probabilities.mean()
        )

        ensemble_entropy = (
            entropy_binary(
                ensemble_probabilities
            )
        )

        predictive_entropy = float(
            entropy_binary(
                mean_p
            )
        )

        epistemic_information_gain = float(
            predictive_entropy
            - ensemble_entropy.mean()
        )

        parameter_variance = float(
            ensemble_probabilities.var(
                ddof=1
            )
        )

        parameter_std = float(
            ensemble_probabilities.std(
                ddof=1
            )
        )

        # ----------------------------------------------------
        # Final row
        # ----------------------------------------------------

        ideal_prediction = (
            int(ideal_p >= 0.5)
        )

        error = int(
            ideal_prediction
            != int(y_test[i])
        )

        row = {
            "patient_id":
                str(patient_ids[i]),

            "label":
                int(y_test[i]),

            "ideal_expectation":
                float(ideal_exp),

            "ideal_probability":
                float(ideal_p),

            "ideal_prediction":
                ideal_prediction,

            "error":
                error,

            "parameter_ensemble_mean_probability":
                mean_p,

            "parameter_ensemble_probability_variance":
                parameter_variance,

            "parameter_ensemble_probability_std":
                parameter_std,

            "parameter_ensemble_predictive_entropy":
                predictive_entropy,

            "parameter_ensemble_expected_entropy":
                float(
                    ensemble_entropy.mean()
                ),

            "epistemic_information_gain":
                epistemic_information_gain,
        }

        for shots in SHOT_COUNTS:

            item = shot_record[
                str(shots)
            ]

            row[
                f"shots_{shots}_mean_probability"
            ] = item[
                "mean_probability"
            ]

            row[
                f"shots_{shots}_probability_variance"
            ] = item[
                "probability_variance"
            ]

            row[
                f"shots_{shots}_probability_std"
            ] = item[
                "probability_std"
            ]

            row[
                f"shots_{shots}_entropy"
            ] = item[
                "predictive_entropy"
            ]

        predictions.append(
            row
        )

    # --------------------------------------------------------
    # Save per-record uncertainty
    # --------------------------------------------------------

    prediction_file = (
        OUT
        / "PHASE7_VQC_UNCERTAINTY_PER_RECORD.csv"
    )

    prediction_df = pd.DataFrame(
        predictions
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    # --------------------------------------------------------
    # Error-discrimination analysis
    # --------------------------------------------------------

    error = prediction_df[
        "error"
    ].to_numpy(
        dtype=int
    )

    uncertainty_columns = {
        "shot_variance_100":
            "shots_100_probability_variance",

        "shot_variance_500":
            "shots_500_probability_variance",

        "shot_variance_1000":
            "shots_1000_probability_variance",

        "parameter_variance":
            "parameter_ensemble_probability_variance",

        "epistemic_information_gain":
            "epistemic_information_gain",

        "parameter_predictive_entropy":
            "parameter_ensemble_predictive_entropy",

        "parameter_std":
            "parameter_ensemble_probability_std",
    }

    error_detection = {}

    for name, column in (
        uncertainty_columns.items()
    ):

        score = prediction_df[
            column
        ].to_numpy(
            dtype=float
        )

        error_detection[name] = {
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
                    score[error == 0].mean()
                )
                if np.any(error == 0)
                else float("nan"),

            "mean_uncertainty_incorrect":
                float(
                    score[error == 1].mean()
                )
                if np.any(error == 1)
                else float("nan"),
        }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            "CBIS-DDSM Phase 7 quantum uncertainty",

        "seed":
            SEED,

        "internal_test_records":
            n,

        "qubits":
            N_QUBITS,

        "depth":
            DEPTH,

        "vqc_trainable_parameters":
            parameter_count,

        "shot_counts":
            list(SHOT_COUNTS),

        "shot_batches":
            SHOT_BATCHES,

        "parameter_ensemble_size":
            PARAMETER_ENSEMBLE_SIZE,

        "parameter_sigma_fraction":
            PARAMETER_SIGMA_FRACTION,

        "interpretation": {
            "shot_variability":
                "finite-measurement/shot variability with fixed trained parameters",

            "parameter_ensemble":
                "epistemic-style local parameter perturbation analysis",

            "caution":
                "the parameter-ensemble information-gain quantity is an epistemic-style estimator under the explicitly defined stochastic perturbation model, not an absolute physical decomposition of uncertainty",
        },

        "error_detection":
            error_detection,

        "prediction_file":
            str(prediction_file),

        "status":
            "PHASE7_COMPLETE",
    }

    summary_file = (
        OUT
        / "PHASE7_VQC_UNCERTAINTY_RESULTS.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "PHASE 7 UNCERTAINTY SUMMARY"
    )
    print("=" * 80)

    for name, values in (
        error_detection.items()
    ):

        print()
        print(name)

        print(
            "  Error-detection AUROC:",
            values[
                "error_detection_auroc"
            ],
        )

        print(
            "  Error-detection AUPRC:",
            values[
                "error_detection_auprc"
            ],
        )

        print(
            "  Mean uncertainty correct:",
            values[
                "mean_uncertainty_correct"
            ],
        )

        print(
            "  Mean uncertainty incorrect:",
            values[
                "mean_uncertainty_incorrect"
            ],
        )

    print()
    print(
        "Per-record uncertainty:",
        prediction_file,
    )

    print(
        "Results:",
        summary_file,
    )

    print()
    print(
        "STATUS: PHASE7_COMPLETE"
    )


if __name__ == "__main__":
    main()