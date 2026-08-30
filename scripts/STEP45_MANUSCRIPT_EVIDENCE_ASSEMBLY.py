from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import sys

import pandas as pd


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP45_MANUSCRIPT_EVIDENCE"
)

TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
METRIC_DIR = OUT / "metrics"
CONFIG_DIR = OUT / "configuration"
CLAIM_DIR = OUT / "claim_map"

for d in [
    OUT,
    TABLE_DIR,
    FIG_DIR,
    SOURCE_DIR,
    METRIC_DIR,
    CONFIG_DIR,
    CLAIM_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


RUNS = {

    "34B":
        ROOT
        / "experiments"
        / "STEP34B_FINAL_MATCHED_MLP_VQC",

    "35A":
        ROOT
        / "experiments"
        / "STEP35A_UNCERTAINTY",

    "35B":
        ROOT
        / "experiments"
        / "STEP35B_PARAMETER_PERTURBATION",

    "36":
        ROOT
        / "experiments"
        / "STEP36_CALIBRATION",

    "37":
        ROOT
        / "experiments"
        / "STEP37_CONFORMAL",

    "38":
        ROOT
        / "experiments"
        / "STEP38_SELECTIVE_PREDICTION",

    "39":
        ROOT
        / "experiments"
        / "STEP39_CORRUPTION_ROBUSTNESS",

    "40":
        ROOT
        / "experiments"
        / "STEP40_QUANTUM_NOISE_ROBUSTNESS",

    "41B":
        ROOT
        / "experiments"
        / "STEP41B_BREAKHIS_EXTERNAL_INFERENCE",

    "41C":
        ROOT
        / "experiments"
        / "STEP41C_THERMOGRAPHY_EXTERNAL_INFERENCE",

    "42":
        ROOT
        / "experiments"
        / "STEP42_FINAL_STATISTICAL_VALIDATION",

    "43":
        ROOT
        / "experiments"
        / "STEP43_MASTER_PUBLICATION_CONSOLIDATION",

    "44":
        ROOT
        / "experiments"
        / "STEP44_PUBLICATION_TABLE_FIGURE_AUDIT",
}


def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def copy_file(
    source,
    destination,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )


print()
print("=" * 100)
print(
    "STEP 45 - MANUSCRIPT EVIDENCE ASSEMBLY"
)
print("=" * 100)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_inputs = [

    RUNS["43"]
    / "tables"
    / "MASTER_PUBLICATION_RESULTS.csv",

    RUNS["43"]
    / "tables"
    / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv",

    RUNS["43"]
    / "tables"
    / "MASTER_SUPPORTING_ANALYSES.csv",

    RUNS["44"]
    / "tables"
    / "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv",

    RUNS["44"]
    / "tables"
    / "TABLE_44_CBIS_PRIMARY_RESULTS.csv",

    RUNS["44"]
    / "tables"
    / "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv",

    RUNS["44"]
    / "tables"
    / "TABLE_44_SUPPORTING_EXPERIMENT_INDEX.csv",
]


missing = [
    str(p)
    for p in required_inputs
    if not p.is_file()
]

if missing:

    raise RuntimeError(
        "Required publication inputs missing:\n"
        +
        "\n".join(
            missing
        )
    )


# ============================================================
# COPY AUTHORITATIVE TABLES
# ============================================================

table_files = {

    "MASTER_PUBLICATION_RESULTS.csv":
        RUNS["43"]
        / "tables"
        / "MASTER_PUBLICATION_RESULTS.csv",

    "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv":
        RUNS["43"]
        / "tables"
        / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv",

    "MASTER_SUPPORTING_ANALYSES.csv":
        RUNS["43"]
        / "tables"
        / "MASTER_SUPPORTING_ANALYSES.csv",

    "MASTER_DATASET_SUMMARY.csv":
        RUNS["43"]
        / "tables"
        / "MASTER_DATASET_SUMMARY.csv",

    "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv",

    "TABLE_44_CBIS_PRIMARY_RESULTS.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_CBIS_PRIMARY_RESULTS.csv",

    "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv",

    "TABLE_44_SUPPORTING_EXPERIMENT_INDEX.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_SUPPORTING_EXPERIMENT_INDEX.csv",

    "TABLE_44_FIGURE_INVENTORY.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_FIGURE_INVENTORY.csv",

    "TABLE_44_FIGURE_COMPLETENESS.csv":
        RUNS["44"]
        / "tables"
        / "TABLE_44_FIGURE_COMPLETENESS.csv",
}


for name, source in table_files.items():

    copy_file(
        source,
        TABLE_DIR / name,
    )


# ============================================================
# COPY FINAL FIGURES
# ============================================================

figure_inventory = pd.read_csv(
    RUNS["44"]
    / "tables"
    / "TABLE_44_FIGURE_INVENTORY.csv"
)

figure_count = 0

for _, row in figure_inventory.iterrows():

    stage = str(
        row["stage"]
    )

    relative_path = str(
        row[
            "relative_path"
        ]
    )

    source = (
        RUNS[stage]
        / relative_path
    )

    if not source.is_file():

        continue

    destination = (
        FIG_DIR
        / stage
        / relative_path
    )

    copy_file(
        source,
        destination,
    )

    figure_count += 1


# ============================================================
# COPY CRITICAL SOURCE JSON
# ============================================================

json_sources = [

    RUNS["34B"]
    / "metrics"
    / "STEP34B_FINAL_RESULTS.json",

    RUNS["35A"]
    / "metrics"
    / "STEP35A_FINAL_RESULTS.json",

    RUNS["35B"]
    / "metrics"
    / "STEP35B_FINAL_RESULTS.json",

    RUNS["36"]
    / "metrics"
    / "STEP36_FINAL_RESULTS.json",

    RUNS["37"]
    / "metrics"
    / "STEP37_FINAL_RESULTS.json",

    RUNS["38"]
    / "metrics"
    / "STEP38_FINAL_RESULTS.json",

    RUNS["39"]
    / "metrics"
    / "STEP39_FINAL_RESULTS.json",

    RUNS["40"]
    / "metrics"
    / "STEP40_FINAL_RESULTS.json",

    RUNS["41B"]
    / "metrics"
    / "STEP41B_FINAL_RESULTS.json",

    RUNS["41C"]
    / "metrics"
    / "STEP41C_FINAL_RESULTS.json",

    RUNS["42"]
    / "metrics"
    / "STEP42_FINAL_RESULTS.json",

    RUNS["43"]
    / "metrics"
    / "STEP43_MASTER_RESULTS.json",

    RUNS["44"]
    / "metrics"
    / "STEP44_FINAL_RESULTS.json",

]


for source in json_sources:

    if not source.is_file():

        continue

    copy_file(
        source,
        METRIC_DIR
        /
        source.name,
    )


# ============================================================
# AUTHORITATIVE CLAIM MAP
# ============================================================

claim_rows = [

    {
        "claim_id":
            "C01",

        "claim":
            "CBIS-DDSM is the primary in-domain evaluation.",

        "evidence":
            str(
                RUNS["34B"]
                / "metrics"
                / "STEP34B_FINAL_RESULTS.json"
            ),

        "stage":
            "34B",
    },

    {
        "claim_id":
            "C02",

        "claim":
            "Matched MLP contains 25 trainable parameters.",

        "evidence":
            str(
                RUNS["34B"]
                / "metrics"
                / "STEP34B_FINAL_RESULTS.json"
            ),

        "stage":
            "34B",
    },

    {
        "claim_id":
            "C03",

        "claim":
            "VQC contains 6 qubits, depth 2, and 24 trainable parameters.",

        "evidence":
            str(
                RUNS["34B"]
                / "metrics"
                / "STEP34B_FINAL_RESULTS.json"
            ),

        "stage":
            "34B",
    },

    {
        "claim_id":
            "C04",

        "claim":
            "Patient-level CBIS VQC ROC-AUC is approximately 0.585.",

        "evidence":
            str(
                RUNS["42"]
                / "tables"
                / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv"
            ),

        "stage":
            "42",
    },

    {
        "claim_id":
            "C05",

        "claim":
            "Patient-level CBIS MLP ROC-AUC is approximately 0.728.",

        "evidence":
            str(
                RUNS["42"]
                / "tables"
                / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv"
            ),

        "stage":
            "42",
    },

    {
        "claim_id":
            "C06",

        "claim":
            "Finite-shot uncertainty was evaluated at 256, 1024, and 4096 shots.",

        "evidence":
            str(
                RUNS["35A"]
                / "metrics"
                / "STEP35A_FINAL_RESULTS.json"
            ),

        "stage":
            "35A",
    },

    {
        "claim_id":
            "C07",

        "claim":
            "Calibration was fitted on the frozen calibration split and evaluated on internal test.",

        "evidence":
            str(
                RUNS["36"]
                / "metrics"
                / "STEP36_FINAL_RESULTS.json"
            ),

        "stage":
            "36",
    },

    {
        "claim_id":
            "C08",

        "claim":
            "Conformal prediction was calibrated on the frozen calibration split.",

        "evidence":
            str(
                RUNS["37"]
                / "metrics"
                / "STEP37_FINAL_RESULTS.json"
            ),

        "stage":
            "37",
    },

    {
        "claim_id":
            "C09",

        "claim":
            "Selective prediction was evaluated at multiple target coverage levels.",

        "evidence":
            str(
                RUNS["38"]
                / "metrics"
                / "STEP38_FINAL_RESULTS.json"
            ),

        "stage":
            "38",
    },

    {
        "claim_id":
            "C10",

        "claim":
            "Image-corruption robustness was evaluated across five corruption families and three severity levels.",

        "evidence":
            str(
                RUNS["39"]
                / "metrics"
                / "STEP39_FINAL_RESULTS.json"
            ),

        "stage":
            "39",
    },

    {
        "claim_id":
            "C11",

        "claim":
            "Quantum-noise robustness was evaluated under controlled simulated noise.",

        "evidence":
            str(
                RUNS["40"]
                / "metrics"
                / "STEP40_FINAL_RESULTS.json"
            ),

        "stage":
            "40",
    },

    {
        "claim_id":
            "C12",

        "claim":
            "BreaKHis is treated as an external histopathology cross-domain transfer stress test.",

        "evidence":
            str(
                RUNS["41B"]
                / "metrics"
                / "STEP41B_FINAL_RESULTS.json"
            ),

        "stage":
            "41B",
    },

    {
        "claim_id":
            "C13",

        "claim":
            "DMR-IR Healthy/Sick and Mendeley Benign/Malignant are analyzed as separate thermography tasks.",

        "evidence":
            str(
                RUNS["41C"]
                / "metrics"
                / "STEP41C_FINAL_RESULTS.json"
            ),

        "stage":
            "41C",
    },

    {
        "claim_id":
            "C14",

        "claim":
            "Bootstrap confidence intervals quantify uncertainty around the reported evaluation metrics.",

        "evidence":
            str(
                RUNS["42"]
                / "metrics"
                / "STEP42_FINAL_RESULTS.json"
            ),

        "stage":
            "42",
    },

    {
        "claim_id":
            "C15",

        "claim":
            "All publication figures have PDF, SVG, and PNG representations.",

        "evidence":
            str(
                RUNS["44"]
                / "tables"
                / "TABLE_44_FIGURE_COMPLETENESS.csv"
            ),

        "stage":
            "44",
    },

]


claim_df = pd.DataFrame(
    claim_rows
)

claim_df.to_csv(
    CLAIM_DIR
    / "MASTER_CLAIM_TO_EVIDENCE_MAP.csv",
    index=False,
)


# ============================================================
# MANUSCRIPT FACT SHEET
# ============================================================

fact_sheet = {

    "primary_dataset":
        "CBIS-DDSM",

    "cbis_test_records":
        578,

    "cbis_test_patients":
        227,

    "cbis_train_records":
        2321,

    "cbis_calibration_records":
        502,

    "breakhis_images":
        7909,

    "dmr_ir_images":
        2394,

    "dmr_ir_healthy":
        1263,

    "dmr_ir_sick":
        1131,

    "mendeley_thermography_images":
        357,

    "mendeley_benign":
        252,

    "mendeley_malignant":
        105,

    "vqc_qubits":
        6,

    "vqc_depth":
        2,

    "vqc_parameters":
        24,

    "mlp_parameters":
        25,

    "pca_dimensions":
        6,

    "bootstrap_replicates":
        2000,

    "confidence_level":
        0.95,

    "external_training":
        False,

    "external_calibration":
        False,

    "posthoc_model_tuning":
        False,

    "original_data_modified":
        False,

    "interpretation":
        "Primary CBIS in-domain evaluation with external "
        "cross-domain/cross-modal robustness stress tests.",
}


(
    SOURCE_DIR
    / "MANUSCRIPT_FACT_SHEET.json"
).write_text(
    json.dumps(
        fact_sheet,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL INVENTORY
# ============================================================

inventory_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
        continue

    inventory_rows.append({

        "relative_path":
            str(
                path.relative_to(
                    OUT
                )
            ),

        "bytes":
            int(
                path.stat().st_size
            ),

        "sha256":
            sha256_file(
                path
            ),
    })


pd.DataFrame(
    inventory_rows
).to_csv(
    OUT
    / "SHA256_INVENTORY.csv",
    index=False,
)


# ============================================================
# RESULT
# ============================================================

result = {

    "status":
        "STEP45_COMPLETE",

    "figure_files_assembled":
        int(
            figure_count
        ),

    "claim_count":
        int(
            len(claim_df)
        ),

    "fact_sheet":
        str(
            SOURCE_DIR
            / "MANUSCRIPT_FACT_SHEET.json"
        ),

    "claim_map":
        str(
            CLAIM_DIR
            / "MASTER_CLAIM_TO_EVIDENCE_MAP.csv"
        ),

    "master_tables":
        [
            str(
                p
            )
            for p in (
                TABLE_DIR.glob(
                    "*.csv"
                )
            )
        ],

    "figures":
        str(
            FIG_DIR
        ),

    "metrics":
        str(
            METRIC_DIR
        ),

    "no_training":
        True,

    "no_model_changes":
        True,

    "status_note":
        "Evidence assembly only; no scientific computation or "
        "model fitting was performed.",
}


(
    METRIC_DIR
    / "STEP45_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CONFIG
# ============================================================

config = {

    "purpose":
        "single manuscript evidence package",

    "training":
        False,

    "model_fitting":
        False,

    "calibration":
        False,

    "source_stage":
        "43-44",

    "claim_map":
        True,

    "required_figure_formats":
        [
            "PDF",
            "SVG",
            "PNG_400_DPI",
        ],
}


(
    CONFIG_DIR
    / "STEP45_CONFIGURATION.json"
).write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


environment = {

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "python":
        sys.version,

    "platform":
        platform.platform(),

    "pandas":
        pd.__version__,
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


print()
print("=" * 100)
print(
    "STEP 45 COMPLETE"
)
print("=" * 100)

print()
print(
    "Claim map:",
    CLAIM_DIR
    / "MASTER_CLAIM_TO_EVIDENCE_MAP.csv",
)

print(
    "Fact sheet:",
    SOURCE_DIR
    / "MANUSCRIPT_FACT_SHEET.json",
)

print(
    "Tables:",
    TABLE_DIR,
)

print(
    "Figures:",
    FIG_DIR,
)

print(
    "Metrics:",
    METRIC_DIR,
)

print()
print(
    "STATUS: STEP45_COMPLETE"
)