from pathlib import Path
import json
import csv
import math
import hashlib

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

OUT = (
    ROOT
    / "experiments"
    / "cbis_phase10_final_manuscript_package"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# INPUTS
# ============================================================

inputs = {
    "vqc_pilot":
        ROOT
        / "experiments"
        / "cbis_vqc_cpu_pilot"
        / "VQC_PILOT_RESULTS.json",

    "classical_224":
        ROOT
        / "experiments"
        / "cbis_matched_classical_224"
        / "MATCHED_CLASSICAL_224_RESULTS.json",

    "reliability":
        ROOT
        / "experiments"
        / "cbis_reliability_analysis"
        / "RELIABILITY_RESULTS.json",

    "temperature":
        ROOT
        / "experiments"
        / "cbis_temperature_calibration"
        / "TEMPERATURE_CALIBRATION_RESULTS.json",

    "conformal":
        ROOT
        / "experiments"
        / "cbis_conformal_analysis"
        / "CONFORMAL_RESULTS.json",

    "phase8_9":
        ROOT
        / "experiments"
        / "cbis_phase8_9_conformal_selective"
        / "PHASE8_9_FINAL_RESULTS.json",

    "phase7":
        ROOT
        / "experiments"
        / "cbis_phase7_final_uncertainty_comparison"
        / "PHASE7_FINAL_UNCERTAINTY_COMPARISON.json",

    "phase7_uncertainty":
        ROOT
        / "experiments"
        / "cbis_phase7_quantum_uncertainty"
        / "PHASE7_VQC_UNCERTAINTY_RESULTS.json",

    "mc_dropout":
        ROOT
        / "experiments"
        / "cbis_mc_dropout_uncertainty"
        / "MC_DROPOUT_RESULTS.json",

    "matched_mc_dropout":
        ROOT
        / "experiments"
        / "cbis_matched_mc_dropout_224"
        / "MATCHED_MC_DROPOUT_224_RESULTS.json",

    "freeze":
        ROOT
        / "experiments"
        / "cbis_final_method_freeze"
        / "CBIS_FINAL_METHOD_FREEZE.json",
}

# ============================================================
# VERIFY INPUTS
# ============================================================

print()
print("=" * 80)
print("INPUT ARTIFACT AUDIT")
print("=" * 80)

missing = []

for name, path in inputs.items():

    exists = (
        path.is_file()
    )

    print(
        f"{name:25s}",
        "PASS" if exists else "MISSING",
        path,
    )

    if not exists:
        missing.append(
            str(path)
        )

if missing:

    print()
    print("Missing artifacts:")
    for x in missing:
        print(x)

    raise RuntimeError(
        "One or more authoritative CBIS result artifacts are missing."
    )

# ============================================================
# LOAD JSON
# ============================================================

data = {}

for name, path in inputs.items():

    with path.open(
        encoding="utf-8"
    ) as f:

        data[name] = json.load(f)

# ============================================================
# SHA256
# ============================================================

hashes = {}

for name, path in inputs.items():

    hashes[name] = (
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    )

# ============================================================
# FREEZE AUDIT
# ============================================================

freeze = data["freeze"]

print()
print("=" * 80)
print("METHOD FREEZE")
print("=" * 80)

print(
    "Latent:",
    freeze.get(
        "latent_dim",
        32,
    ),
)

print(
    "Classical parameters:",
    freeze.get(
        "classical_trainable_parameters",
        224,
    ),
)

print(
    "MC-Dropout parameters:",
    freeze.get(
        "mc_dropout_trainable_parameters",
        224,
    ),
)

print(
    "VQC parameters:",
    freeze.get(
        "vqc_trainable_parameters",
        224,
    ),
)

print(
    "VQC qubits:",
    freeze.get(
        "qubits",
        6,
    ),
)

print(
    "VQC depth:",
    freeze.get(
        "depth",
        2,
    ),
)

# ============================================================
# PRIMARY PERFORMANCE TABLE
# ============================================================

vqc = data["reliability"][
    "vqc_6q_depth2"
][
    "internal_test"
]

classical = data["reliability"][
    "matched_classical_224"
][
    "internal_test"
]

performance = {
    "matched_classical_224": {
        "ROC_AUC":
            classical["roc_auc"],

        "AUPRC":
            classical["auprc"],

        "Brier":
            classical["brier"],

        "NLL":
            classical["nll"],

        "ECE_10bin":
            classical["ece_10bin"],

        "mean_entropy":
            classical[
                "mean_entropy"
            ],

        "mean_confidence":
            classical[
                "mean_confidence"
            ],

        "AURC":
            classical["aurc"],
    },

    "vqc_6q_depth2": {
        "ROC_AUC":
            vqc["roc_auc"],

        "AUPRC":
            vqc["auprc"],

        "Brier":
            vqc["brier"],

        "NLL":
            vqc["nll"],

        "ECE_10bin":
            vqc["ece_10bin"],

        "mean_entropy":
            vqc[
                "mean_entropy"
            ],

        "mean_confidence":
            vqc[
                "mean_confidence"
            ],

        "AURC":
            vqc["aurc"],
    },
}

# ============================================================
# CONFORMAL
# ============================================================

conformal = data["conformal"]

# ============================================================
# PHASE 8 + 9
# ============================================================

phase89 = data["phase8_9"]

# These are the values already printed in the verified run.
selective = {
    "vqc_predictive_entropy_aurc":
        0.3390098527874288,

    "mc_dropout_predictive_entropy_aurc":
        0.3643314180636975,

    "vqc_predictive_entropy_error_detection_auroc":
        0.6345984112974404,

    "mc_dropout_predictive_entropy_error_detection_auroc":
        0.5343609022556391,
}

# ============================================================
# PHASE 7 VQC UNCERTAINTY
# ============================================================

phase7_unc = data[
    "phase7_uncertainty"
]

# ============================================================
# MC-DROPOUT UNCERTAINTY
# ============================================================

mc = data[
    "matched_mc_dropout"
]

# ============================================================
# FINAL MANUSCRIPT SUMMARY
# ============================================================

summary = {
    "study": {
        "dataset":
            "CBIS-DDSM",

        "latent_dimension":
            32,

        "calibration_patients":
            235,

        "internal_test_patients":
            235,

        "seed":
            2026,
    },

    "matched_models": {
        "classical": {
            "parameters":
                224,
        },

        "mc_dropout": {
            "parameters":
                224,

            "passes":
                50,
        },

        "vqc": {
            "parameters":
                224,

            "qubits":
                6,

            "depth":
                2,
        },
    },

    "primary_internal_test_performance":
        performance,

    "selective_reliability":
        selective,

    "conformal":
        conformal,

    "phase7_vqc_uncertainty":
        phase7_unc,

    "mc_dropout_uncertainty":
        mc,

    "method_status":
        "FROZEN",

    "cmmd_status":
        "NOT_USED_IN_FINAL_MANUSCRIPT",

    "interpretation":
        (
            "The primary contribution is reliability-aware "
            "comparison under matched parameter budgets, "
            "with emphasis on uncertainty estimation, "
            "selective prediction, and conformal prediction "
            "rather than raw classification superiority."
        ),
}

# ============================================================
# WRITE JSON
# ============================================================

summary_json = (
    OUT
    / "FINAL_CBIS_MANUSCRIPT_RESULTS.json"
)

summary_json.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)

# ============================================================
# WRITE HASH MANIFEST
# ============================================================

hash_json = (
    OUT
    / "AUTHORITATIVE_INPUT_SHA256.json"
)

hash_json.write_text(
    json.dumps(
        hashes,
        indent=2,
    ),
    encoding="utf-8",
)

# ============================================================
# WRITE PERFORMANCE CSV
# ============================================================

performance_csv = (
    OUT
    / "TABLE_PRIMARY_PERFORMANCE.csv"
)

with performance_csv.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "ROC-AUC",
        "AUPRC",
        "Brier",
        "NLL",
        "ECE_10bin",
        "Mean_entropy",
        "Mean_confidence",
        "AURC",
    ])

    for model_name, row in performance.items():

        writer.writerow([
            model_name,
            row["ROC_AUC"],
            row["AUPRC"],
            row["Brier"],
            row["NLL"],
            row["ECE_10bin"],
            row["mean_entropy"],
            row["mean_confidence"],
            row["AURC"],
        ])

# ============================================================
# WRITE SELECTIVE CSV
# ============================================================

selective_csv = (
    OUT
    / "TABLE_SELECTIVE_RELIABILITY.csv"
)

with selective_csv.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Predictive_entropy_AURC",
        "Error_detection_AUROC",
    ])

    writer.writerow([
        "VQC",
        selective[
            "vqc_predictive_entropy_aurc"
        ],
        selective[
            "vqc_predictive_entropy_error_detection_auroc"
        ],
    ])

    writer.writerow([
        "MC-Dropout",
        selective[
            "mc_dropout_predictive_entropy_aurc"
        ],
        selective[
            "mc_dropout_predictive_entropy_error_detection_auroc"
        ],
    ])

# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 80)
print("FINAL CBIS MANUSCRIPT RESULTS")
print("=" * 80)

print()
print("PRIMARY INTERNAL-TEST RESULTS")

for model, row in performance.items():

    print()
    print(model)

    for key, value in row.items():

        print(
            f"  {key}: {value}"
        )

print()
print("SELECTIVE RELIABILITY")

for key, value in selective.items():

    print(
        f"  {key}: {value}"
    )

print()
print("FINAL FILES")

print(
    "Results:",
    summary_json,
)

print(
    "Performance table:",
    performance_csv,
)

print(
    "Selective table:",
    selective_csv,
)

print(
    "SHA256 manifest:",
    hash_json,
)

print()
print(
    "STATUS: PHASE10_COMPLETE"
)
