from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(r"E:\CBIS_DDSM_QUANTUM")

LATENT = (
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

CLASSICAL_CHECKPOINT = (
    ROOT
    / "experiments"
    / "cbis_matched_classical_224"
    / "best_matched_classical_224.pt"
)

MC_CHECKPOINT = (
    ROOT
    / "experiments"
    / "cbis_matched_mc_dropout_224"
    / "best_matched_mc_dropout_224.pt"
)

VQC_RESULTS = (
    ROOT
    / "experiments"
    / "cbis_vqc_cpu_pilot"
    / "VQC_PILOT_RESULTS.json"
)

CLASSICAL_RESULTS = (
    ROOT
    / "experiments"
    / "cbis_matched_classical_224"
    / "MATCHED_CLASSICAL_224_RESULTS.json"
)

PHASE7_RESULTS = (
    ROOT
    / "experiments"
    / "cbis_phase7_final_uncertainty_comparison"
    / "PHASE7_FINAL_UNCERTAINTY_COMPARISON.json"
)

PHASE89_RESULTS = (
    ROOT
    / "experiments"
    / "cbis_phase8_9_conformal_selective"
    / "PHASE8_9_FINAL_RESULTS.json"
)

CONFORMAL_FILE = (
    ROOT
    / "experiments"
    / "cbis_phase8_9_conformal_selective"
    / "FINAL_CONFORMAL_PREDICTIONS.csv"
)

SELECTIVE_FILE = (
    ROOT
    / "experiments"
    / "cbis_phase8_9_conformal_selective"
    / "FINAL_SELECTIVE_PATIENT_DATA.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_final_method_freeze"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


def required_file(path: Path):

    if not path.is_file():

        raise RuntimeError(
            f"Required artifact missing: {path}"
        )


def main():

    print("=" * 80)
    print(
        "CBIS FINAL METHOD FREEZE AUDIT"
    )
    print("=" * 80)

    files = [
        LATENT,
        VQC_CHECKPOINT,
        CLASSICAL_CHECKPOINT,
        MC_CHECKPOINT,
        VQC_RESULTS,
        CLASSICAL_RESULTS,
        PHASE7_RESULTS,
        PHASE89_RESULTS,
        CONFORMAL_FILE,
        SELECTIVE_FILE,
    ]

    print()
    print(
        "ARTIFACT EXISTENCE"
    )

    for path in files:

        required_file(path)

        print(
            "PASS:",
            path,
        )

    # --------------------------------------------------------
    # Latent audit
    # --------------------------------------------------------

    data = torch.load(
        LATENT,
        map_location="cpu",
        weights_only=False,
    )

    expected_shapes = {
        "train_z": (2427, 32),
        "calibration_z": (537, 32),
        "internal_test_z": (602, 32),
    }

    print()
    print(
        "LATENT AUDIT"
    )

    for key, shape in expected_shapes.items():

        actual = tuple(
            data[key].shape
        )

        print(
            key,
            "expected=",
            shape,
            "actual=",
            actual,
        )

        if actual != shape:

            raise RuntimeError(
                f"{key} shape mismatch."
            )

    if int(
        data["latent_dim"]
    ) != 32:

        raise RuntimeError(
            "Latent dimension is not 32."
        )

    if int(
        data["seed"]
    ) != 2026:

        raise RuntimeError(
            "Latent seed is not 2026."
        )

    # --------------------------------------------------------
    # Checkpoint audits
    # --------------------------------------------------------

    vqc_state = torch.load(
        VQC_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    classical_state = torch.load(
        CLASSICAL_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    mc_state = torch.load(
        MC_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    vqc_parameters = sum(
        tensor.numel()
        for tensor in vqc_state.values()
    )

    classical_parameters = sum(
        tensor.numel()
        for tensor in classical_state.values()
    )

    mc_parameters = sum(
        tensor.numel()
        for tensor in mc_state.values()
    )

    print()
    print(
        "PARAMETER AUDIT"
    )

    print(
        "VQC parameters:",
        vqc_parameters,
    )

    print(
        "Classical parameters:",
        classical_parameters,
    )

    print(
        "MC-Dropout parameters:",
        mc_parameters,
    )

    if vqc_parameters != 224:

        raise RuntimeError(
            "VQC parameter count is not 224."
        )

    if classical_parameters != 224:

        raise RuntimeError(
            "Classical parameter count is not 224."
        )

    if mc_parameters != 224:

        raise RuntimeError(
            "MC-Dropout parameter count is not 224."
        )

    # --------------------------------------------------------
    # Result-file audits
    # --------------------------------------------------------

    vqc_results = json.loads(
        VQC_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    classical_results = json.loads(
        CLASSICAL_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    phase7 = json.loads(
        PHASE7_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    phase89 = json.loads(
        PHASE89_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "RESULT AUDIT"
    )

    print(
        "VQC status:",
        vqc_results.get(
            "status"
        )
    )

    print(
        "Classical status:",
        classical_results.get(
            "status"
        )
    )

    print(
        "Phase 7 status:",
        phase7.get(
            "status"
        )
    )

    print(
        "Phase 8-9 status:",
        phase89.get(
            "status"
        )
    )

    # --------------------------------------------------------
    # Final frozen protocol
    # --------------------------------------------------------

    frozen_protocol = {

        "dataset":
            "CBIS-DDSM",

        "seed":
            2026,

        "latent_dim":
            32,

        "train_records":
            2427,

        "calibration_records":
            537,

        "internal_test_records":
            602,

        "calibration_patients":
            235,

        "internal_test_patients":
            235,

        "classical_comparator":
            {
                "parameters":
                    224,

                "architecture":
                    "32 -> 7 bias-free linear -> fixed mean",
            },

        "mc_dropout":
            {
                "parameters":
                    224,

                "dropout":
                    0.30,

                "mc_passes":
                    50,
            },

        "vqc":
            {
                "parameters":
                    224,

                "qubits":
                    6,

                "depth":
                    2,
            },

        "vqc_uncertainty":
            {
                "shot_counts":
                    [
                        100,
                        500,
                        1000,
                    ],

                "shot_batches":
                    10,

                "parameter_ensemble_size":
                    20,

                "parameter_sigma_fraction":
                    0.05,

                "primary_common_uncertainty":
                    "predictive entropy",

                "epistemic_style":
                    "parameter-perturbation information-gain estimator",
            },

        "conformal":
            {
                "alphas":
                    [
                        0.10,
                        0.05,
                    ],

                "primary_calibration_partition":
                    "calibration",

                "evaluation_partition":
                    "internal_test",
            },

        "selective_prediction":
            {
                "primary_uncertainty":
                    "predictive entropy",

                "policy":
                    "accept lowest-uncertainty cases first",
            },

        "external_validation":
            {
                "cmmd":
                    "held out for external distribution-shift evaluation",

                "cmmd_tuning":
                    False,
            },

        "method_freeze_status":
            "FROZEN_BEFORE_EXTERNAL_VALIDATION",
    }

    # --------------------------------------------------------
    # Artifact hashes
    # --------------------------------------------------------

    hashes = {}

    for path in files:

        hashes[
            str(path)
        ] = sha256(path)

    # --------------------------------------------------------
    # Save freeze manifest
    # --------------------------------------------------------

    freeze_manifest = {

        "experiment":
            "CBIS-DDSM reliability model frozen protocol",

        "protocol":
            frozen_protocol,

        "artifacts_sha256":
            hashes,

        "status":
            "FROZEN_BEFORE_EXTERNAL_VALIDATION",
    }

    output = (
        OUT
        / "CBIS_FINAL_METHOD_FREEZE.json"
    )

    output.write_text(
        json.dumps(
            freeze_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FINAL CBIS METHOD FREEZE"
    )
    print("=" * 80)

    print(
        "Latent:",
        "32-D",
    )

    print(
        "Classical:",
        "224 parameters",
    )

    print(
        "MC-Dropout:",
        "224 parameters / 50 passes",
    )

    print(
        "VQC:",
        "6 qubits / depth 2 / 224 parameters",
    )

    print(
        "Calibration:",
        "235 patients",
    )

    print(
        "Internal test:",
        "235 patients",
    )

    print(
        "Conformal alpha:",
        "0.10, 0.05",
    )

    print(
        "Primary selective uncertainty:",
        "predictive entropy",
    )

    print()
    print(
        "FREEZE MANIFEST:"
    )

    print(
        output
    )

    print()
    print(
        "STATUS: CBIS METHOD FROZEN"
    )


if __name__ == "__main__":
    main()