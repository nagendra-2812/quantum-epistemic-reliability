from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

RUNS = {
    "34B": ROOT / "experiments" / "STEP34B_FINAL_MATCHED_MLP_VQC",
    "35A": ROOT / "experiments" / "STEP35A_UNCERTAINTY",
    "35B": ROOT / "experiments" / "STEP35B_PARAMETER_PERTURBATION",
    "36": ROOT / "experiments" / "STEP36_CALIBRATION",
    "37": ROOT / "experiments" / "STEP37_CONFORMAL",
    "38": ROOT / "experiments" / "STEP38_SELECTIVE_PREDICTION",
    "39": ROOT / "experiments" / "STEP39_CORRUPTION_ROBUSTNESS",
    "40": ROOT / "experiments" / "STEP40_QUANTUM_NOISE_ROBUSTNESS",
    "41B": ROOT / "experiments" / "STEP41B_BREAKHIS_EXTERNAL_INFERENCE",
    "41C": ROOT / "experiments" / "STEP41C_THERMOGRAPHY_EXTERNAL_INFERENCE",
    "42": ROOT / "experiments" / "STEP42_FINAL_STATISTICAL_VALIDATION",
    "43": ROOT / "experiments" / "STEP43_MASTER_PUBLICATION_CONSOLIDATION",
}

OUT = (
    ROOT
    / "experiments"
    / "STEP44_PUBLICATION_TABLE_FIGURE_AUDIT"
)

TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_data"
METRIC_DIR = OUT / "metrics"
CONFIG_DIR = OUT / "configuration"

for d in [
    OUT,
    TABLE_DIR,
    SOURCE_DIR,
    METRIC_DIR,
    CONFIG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================

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


def require_file(path):

    if not path.is_file():

        raise RuntimeError(
            f"Required file missing: {path}"
        )

    return path


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print(
    "STEP 44 - PUBLICATION TABLE / FIGURE AUDIT"
)
print("=" * 100)


# ============================================================
# REQUIRE MASTER RESULTS
# ============================================================

MASTER = require_file(
    RUNS["43"]
    / "tables"
    / "MASTER_PUBLICATION_RESULTS.csv"
)

BOOTSTRAP = require_file(
    RUNS["43"]
    / "tables"
    / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv"
)

SUPPORTING = require_file(
    RUNS["43"]
    / "tables"
    / "MASTER_SUPPORTING_ANALYSES.csv"
)

DATASET_SUMMARY = require_file(
    RUNS["43"]
    / "tables"
    / "MASTER_DATASET_SUMMARY.csv"
)

master = pd.read_csv(
    MASTER
)

bootstrap = pd.read_csv(
    BOOTSTRAP
)

supporting = pd.read_csv(
    SUPPORTING
)

dataset_summary = pd.read_csv(
    DATASET_SUMMARY
)


# ============================================================
# MASTER DATA QUALITY
# ============================================================

required_master = [
    "dataset",
    "level",
    "model",
    "metric",
    "value",
]

missing_master = [
    c
    for c in required_master
    if c not in master.columns
]

if missing_master:

    raise RuntimeError(
        f"MASTER_PUBLICATION_RESULTS missing: {missing_master}"
    )


expected_datasets = {
    "CBIS-DDSM",
    "BreaKHis",
    "DMR-IR Healthy/Sick",
    "Mendeley Benign/Malignant",
}

found_datasets = set(
    master[
        "dataset"
    ].astype(str)
)

missing_datasets = (
    expected_datasets
    -
    found_datasets
)

if missing_datasets:

    raise RuntimeError(
        "Expected datasets absent from master table: "
        + str(
            sorted(
                missing_datasets
            )
        )
    )


# ============================================================
# DUPLICATE CLAIM CHECK
# ============================================================

duplicate_claims = (
    master
    .groupby(
        [
            "dataset",
            "level",
            "model",
            "metric",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

duplicate_claims = duplicate_claims[
    duplicate_claims[
        "count"
    ] > 1
]

if len(
    duplicate_claims
):

    raise RuntimeError(
        "Duplicate publication claims found:\n"
        +
        duplicate_claims.to_string(
            index=False
        )
    )


# ============================================================
# CREATE PRIMARY TABLE
# ============================================================

primary = bootstrap[
    bootstrap[
        "level"
    ]
    .astype(str)
    .eq("patient")
].copy()

primary[
    "estimate"
] = primary[
    "estimate"
].astype(float)

primary[
    "ci95_low"
] = primary[
    "ci95_low"
].astype(float)

primary[
    "ci95_high"
] = primary[
    "ci95_high"
].astype(float)

primary[
    "estimate_with_ci"
] = (
    primary[
        "estimate"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    " ["
    +
    primary[
        "ci95_low"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    ", "
    +
    primary[
        "ci95_high"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    "]"
)

primary_table = primary[
    [
        "dataset",
        "model",
        "metric",
        "n",
        "estimate_with_ci",
    ]
].copy()

primary_table.to_csv(
    TABLE_DIR
    / "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv",
    index=False,
)


# ============================================================
# PRIMARY CBIS TABLE
# ============================================================

cbis = bootstrap[
    bootstrap[
        "dataset"
    ]
    .astype(str)
    .eq("CBIS-DDSM")
].copy()

cbis_primary = cbis[
    cbis[
        "level"
    ].astype(str).eq("patient")
].copy()

cbis_primary[
    "estimate_ci"
] = (
    cbis_primary[
        "estimate"
    ].map(
        lambda x:
            f"{x:.4f}"
    )
    +
    " ["
    +
    cbis_primary[
        "ci95_low"
    ].map(
        lambda x:
            f"{x:.4f}"
    )
    +
    ", "
    +
    cbis_primary[
        "ci95_high"
    ].map(
        lambda x:
            f"{x:.4f}"
    )
    +
    "]"
)

cbis_primary.to_csv(
    TABLE_DIR
    / "TABLE_44_CBIS_PRIMARY_RESULTS.csv",
    index=False,
)


# ============================================================
# CROSS-DATASET TABLE
# ============================================================

cross_dataset = bootstrap[
    bootstrap[
        "level"
    ].astype(str).eq("record")
].copy()

cross_dataset[
    "estimate_ci"
] = (
    cross_dataset[
        "estimate"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    " ["
    +
    cross_dataset[
        "ci95_low"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    ", "
    +
    cross_dataset[
        "ci95_high"
    ].map(
        lambda x:
            f"{x:.3f}"
    )
    +
    "]"
)

cross_dataset.to_csv(
    TABLE_DIR
    / "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv",
    index=False,
)


# ============================================================
# SUPPORTING EXPERIMENT INDEX
# ============================================================

supporting[
    [
        "stage",
        "description",
        "status",
    ]
].to_csv(
    TABLE_DIR
    / "TABLE_44_SUPPORTING_EXPERIMENT_INDEX.csv",
    index=False,
)


# ============================================================
# ARTIFACT INVENTORY
# ============================================================

artifact_rows = []

for name, folder in RUNS.items():

    if not folder.is_dir():
        continue

    for path in sorted(
        folder.rglob("*")
    ):

        if not path.is_file():
            continue

        artifact_rows.append({

            "stage":
                name,

            "relative_path":
                str(
                    path.relative_to(
                        folder
                    )
                ),

            "bytes":
                int(
                    path.stat().st_size
                ),

            "extension":
                path.suffix.lower(),

        })


artifact_df = pd.DataFrame(
    artifact_rows
)

artifact_df.to_csv(
    SOURCE_DIR
    / "STEP44_ALL_PUBLICATION_ARTIFACTS.csv",
    index=False,
)


# ============================================================
# FIGURE AUDIT
# ============================================================

figure_rows = []

for name, folder in RUNS.items():

    if not folder.is_dir():
        continue

    for path in sorted(
        folder.rglob("*")
    ):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix not in [
            ".pdf",
            ".svg",
            ".png",
        ]:
            continue

        figure_rows.append({

            "stage":
                name,

            "relative_path":
                str(
                    path.relative_to(
                        folder
                    )
                ),

            "type":
                suffix[1:].upper(),

            "bytes":
                int(
                    path.stat().st_size
                ),

        })


figure_df = pd.DataFrame(
    figure_rows
)

figure_df.to_csv(
    TABLE_DIR
    / "TABLE_44_FIGURE_INVENTORY.csv",
    index=False,
)


# ============================================================
# PUBLICATION FIGURE COMPLETENESS
# ============================================================

figure_groups = {}

for _, row in figure_df.iterrows():

    relative = row[
        "relative_path"
    ]

    stem = Path(
        relative
    ).stem

    if stem.endswith(
        "_400DPI"
    ):

        stem = stem[
            :-
            len(
                "_400DPI"
            )
        ]

    figure_groups.setdefault(
        (
            row[
                "stage"
            ],
            stem,
        ),
        set(),
    ).add(
        row[
            "type"
        ]
    )


completeness_rows = []

for (
    key,
    types,
) in sorted(
    figure_groups.items()
):

    stage, stem = key

    completeness_rows.append({

        "stage":
            stage,

        "figure_stem":
            stem,

        "PDF":
            "PDF" in types,

        "SVG":
            "SVG" in types,

        "PNG":
            "PNG" in types,

        "complete":
            (
                "PDF" in types
                and
                "SVG" in types
                and
                "PNG" in types
            ),
    })


completeness_df = pd.DataFrame(
    completeness_rows
)

completeness_df.to_csv(
    TABLE_DIR
    / "TABLE_44_FIGURE_COMPLETENESS.csv",
    index=False,
)


# ============================================================
# CHECK PUBLICATION FIGURES
# ============================================================

incomplete_figures = completeness_df[
    ~completeness_df[
        "complete"
    ]
]

if len(
    incomplete_figures
):

    print()
    print(
        "WARNING: figure groups without PDF+SVG+PNG:"
    )

    print(
        incomplete_figures.to_string(
            index=False
        )
    )


# ============================================================
# CLAIM GUARDRAILS
# ============================================================

guardrails = {

    "CBIS_is_primary_in_domain_evaluation":
        True,

    "BreaKHis_is_cross_domain_stress_test":
        True,

    "DMR_IR_is_cross_modal_stress_test":
        True,

    "Mendeley_is_separate_thermography_task":
        True,

    "uncertainty_claims_require_error_detection_or_calibration_evidence":
        True,

    "no_claim_that_VQC_is_globally_superior":
        True,

    "no_posthoc_model_tuning":
        True,

    "no_external_training":
        True,

    "no_external_calibration":
        True,

    "all_figure_formats_verified":
        bool(
            (
                len(
                    incomplete_figures
                )
                ==
                0
            )
        ),
}


(
    METRIC_DIR
    / "STEP44_CLAIM_GUARDRAILS.json"
).write_text(
    json.dumps(
        guardrails,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# MASTER JSON
# ============================================================

result = {

    "status":
        "STEP44_COMPLETE",

    "master_table":
        str(
            MASTER
        ),

    "bootstrap_table":
        str(
            BOOTSTRAP
        ),

    "publication_tables":
        [
            str(
                TABLE_DIR
                / "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv"
            ),

            str(
                TABLE_DIR
                / "TABLE_44_CBIS_PRIMARY_RESULTS.csv"
            ),

            str(
                TABLE_DIR
                / "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv"
            ),

            str(
                TABLE_DIR
                / "TABLE_44_SUPPORTING_EXPERIMENT_INDEX.csv"
            ),
        ],

    "figure_inventory":
        str(
            TABLE_DIR
            / "TABLE_44_FIGURE_INVENTORY.csv"
        ),

    "figure_completeness":
        str(
            TABLE_DIR
            / "TABLE_44_FIGURE_COMPLETENESS.csv"
        ),

    "incomplete_figure_count":
        int(
            len(
                incomplete_figures
            )
        ),

    "guardrails":
        guardrails,
}


(
    METRIC_DIR
    / "STEP44_FINAL_RESULTS.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CONFIGURATION
# ============================================================

config = {

    "purpose":
        "publication table and figure audit",

    "new_training":
        False,

    "new_model_fitting":
        False,

    "new_calibration":
        False,

    "primary_dataset":
        "CBIS-DDSM",

    "external_datasets":
        [
            "BreaKHis",
            "DMR-IR Healthy/Sick",
            "Mendeley Benign/Malignant",
        ],

    "required_figure_formats":
        [
            "PDF",
            "SVG",
            "PNG_400_DPI",
        ],

    "source_master_results":
        str(
            MASTER
        ),

    "source_statistical_results":
        str(
            BOOTSTRAP
        ),
}


(
    CONFIG_DIR
    / "STEP44_CONFIGURATION.json"
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

    "numpy":
        np.__version__,

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


# ============================================================
# SHA256
# ============================================================

hash_rows = []

for path in sorted(
    OUT.rglob("*")
):

    if not path.is_file():
        continue

    if path.name == "SHA256_INVENTORY.csv":
        continue

    hash_rows.append({

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
    hash_rows
).to_csv(
    OUT
    / "SHA256_INVENTORY.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print(
    "STEP 44 COMPLETE"
)
print("=" * 100)

print()
print(
    "Primary table:",
    TABLE_DIR
    / "TABLE_44_PRIMARY_PATIENT_LEVEL_RESULTS.csv",
)

print(
    "CBIS table:",
    TABLE_DIR
    / "TABLE_44_CBIS_PRIMARY_RESULTS.csv",
)

print(
    "Cross-dataset table:",
    TABLE_DIR
    / "TABLE_44_CROSS_DATASET_TRANSFER_RESULTS.csv",
)

print(
    "Figure inventory:",
    TABLE_DIR
    / "TABLE_44_FIGURE_INVENTORY.csv",
)

print(
    "Figure completeness:",
    TABLE_DIR
    / "TABLE_44_FIGURE_COMPLETENESS.csv",
)

print(
    "Results:",
    METRIC_DIR
    / "STEP44_FINAL_RESULTS.json",
)

print()
print(
    "STATUS: STEP44_COMPLETE"
)