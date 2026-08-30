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

OUT = (
    ROOT
    / "experiments"
    / "STEP43_MASTER_PUBLICATION_CONSOLIDATION"
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
}


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

            h.update(
                block
            )

    return h.hexdigest()


def load_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def find_json(
    folder,
    name,
):

    path = (
        folder
        / name
    )

    if path.is_file():

        return path

    return None


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print(
    "STEP 43 - MASTER PUBLICATION CONSOLIDATION"
)
print("=" * 100)


# ============================================================
# VERIFY COMPLETED RUNS
# ============================================================

run_status = {}

for name, folder in RUNS.items():

    exists = folder.is_dir()

    run_status[
        name
    ] = {

        "folder":
            str(folder),

        "exists":
            exists,
    }

    print(
        f"{name}:",
        "FOUND"
        if exists
        else
        "MISSING",
    )

if not all(
    x[
        "exists"
    ]
    for x in run_status.values()
):

    missing = [
        k
        for k, v
        in run_status.items()
        if not v[
            "exists"
        ]
    ]

    raise RuntimeError(
        "Required completed experiment folders missing: "
        + str(missing)
    )


# ============================================================
# PRIMARY CBIS RESULTS
# ============================================================

# CBIS numerical results are taken from the authoritative
# Step 42 bootstrap/statistical table. This avoids assuming
# undocumented JSON field names in the Step 34B result file.

cbis_rows = []


# ============================================================
# STEP 42 BOOTSTRAP RESULTS
# ============================================================

stat_table = (
    RUNS["42"]
    / "tables"
    / "TABLE_42_BOOTSTRAP_STATISTICAL_VALIDATION.csv"
)

if not stat_table.is_file():

    raise RuntimeError(
        f"Step 42 statistical table missing: {stat_table}"
    )

stats = pd.read_csv(
    stat_table
)

stats[
    "metric"
] = (
    stats[
        "metric"
    ]
    .astype(str)
    .str.upper()
)


# ============================================================
# PRIMARY STATISTICAL TABLE
# ============================================================

primary_stats = stats[
    stats[
        "dataset"
    ].isin(
        [
            "CBIS-DDSM",
            "BreaKHis",
            "DMR-IR Healthy/Sick",
            "Mendeley Benign/Malignant",
        ]
    )
].copy()


primary_stats[
    "ci95"
] = (
    primary_stats[
        "ci95_low"
    ].map(
        lambda x:
            f"{x:.4f}"
    )
    +
    "â€“"
    +
    primary_stats[
        "ci95_high"
    ].map(
        lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# MASTER RESULTS
# ============================================================

master_rows = []

for row in cbis_rows:

    master_rows.append(
        row
    )


for _, row in primary_stats.iterrows():

    master_rows.append({

        "dataset":
            row[
                "dataset"
            ],

        "level":
            row[
                "level"
            ],

        "model":
            row[
                "model"
            ],

        "metric":
            row[
                "metric"
            ],

        "value":
            float(
                row[
                    "estimate"
                ]
            ),

        "ci95_low":
            float(
                row[
                    "ci95_low"
                ]
            ),

        "ci95_high":
            float(
                row[
                    "ci95_high"
                ]
            ),

        "ci95":
            row[
                "ci95"
            ],

        "source":
            str(
                stat_table
            ),
    })


master = pd.DataFrame(
    master_rows
)


# ============================================================
# UNIQUE PRIMARY ROWS
# ============================================================

master = master.drop_duplicates(
    subset=[
        "dataset",
        "level",
        "model",
        "metric",
    ],
    keep="first",
).reset_index(
    drop=True
)


# ============================================================
# LOAD SUPPORTING ANALYSES
# ============================================================

supporting = []


# 35A
path35a = (
    RUNS["35A"]
    / "metrics"
    / "STEP35A_FINAL_RESULTS.json"
)

if path35a.is_file():

    data35a = load_json(
        path35a
    )

    supporting.append({

        "stage":
            "35A",

        "description":
            "finite-shot uncertainty",

        "source":
            str(
                path35a
            ),

        "status":
            data35a.get(
                "status",
                "",
            ),
    })


# 35B
path35b = (
    RUNS["35B"]
    / "metrics"
    / "STEP35B_FINAL_RESULTS.json"
)

if path35b.is_file():

    data35b = load_json(
        path35b
    )

    supporting.append({

        "stage":
            "35B",

        "description":
            "parameter perturbation",

        "source":
            str(
                path35b
            ),

        "status":
            data35b.get(
                "status",
                "",
            ),
    })


# 36
path36 = (
    RUNS["36"]
    / "metrics"
    / "STEP36_FINAL_RESULTS.json"
)

if path36.is_file():

    data36 = load_json(
        path36
    )

    supporting.append({

        "stage":
            "36",

        "description":
            "calibration and reliability",

        "source":
            str(
                path36
            ),

        "status":
            data36.get(
                "status",
                "",
            ),
    })


# 37
path37 = (
    RUNS["37"]
    / "metrics"
    / "STEP37_FINAL_RESULTS.json"
)

if path37.is_file():

    data37 = load_json(
        path37
    )

    supporting.append({

        "stage":
            "37",

        "description":
            "conformal prediction",

        "source":
            str(
                path37
            ),

        "status":
            data37.get(
                "status",
                "",
            ),
    })


# 38
path38 = (
    RUNS["38"]
    / "metrics"
    / "STEP38_FINAL_RESULTS.json"
)

if path38.is_file():

    data38 = load_json(
        path38
    )

    supporting.append({

        "stage":
            "38",

        "description":
            "selective prediction",

        "source":
            str(
                path38
            ),

        "status":
            data38.get(
                "status",
                "",
            ),
    })


# 39
path39 = (
    RUNS["39"]
    / "metrics"
    / "STEP39_FINAL_RESULTS.json"
)

if path39.is_file():

    data39 = load_json(
        path39
    )

    supporting.append({

        "stage":
            "39",

        "description":
            "image corruption robustness",

        "source":
            str(
                path39
            ),

        "status":
            data39.get(
                "status",
                "",
            ),
    })


# 40
path40 = (
    RUNS["40"]
    / "metrics"
    / "STEP40_FINAL_RESULTS.json"
)

if path40.is_file():

    data40 = load_json(
        path40
    )

    supporting.append({

        "stage":
            "40",

        "description":
            "quantum noise robustness",

        "source":
            str(
                path40
            ),

        "status":
            data40.get(
                "status",
                "",
            ),
    })


# 41B
path41b = (
    RUNS["41B"]
    / "metrics"
    / "STEP41B_FINAL_RESULTS.json"
)

if path41b.is_file():

    data41b = load_json(
        path41b
    )

    supporting.append({

        "stage":
            "41B",

        "description":
            "BreaKHis external transfer",

        "source":
            str(
                path41b
            ),

        "status":
            data41b.get(
                "status",
                "",
            ),
    })


# 41C
path41c = (
    RUNS["41C"]
    / "metrics"
    / "STEP41C_FINAL_RESULTS.json"
)

if path41c.is_file():

    data41c = load_json(
        path41c
    )

    supporting.append({

        "stage":
            "41C",

        "description":
            "thermography external transfer",

        "source":
            str(
                path41c
            ),

        "status":
            data41c.get(
                "status",
                "",
            ),
    })


# ============================================================
# SAVE TABLES
# ============================================================

master.to_csv(
    TABLE_DIR
    / "MASTER_PUBLICATION_RESULTS.csv",
    index=False,
)

primary_stats.to_csv(
    TABLE_DIR
    / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv",
    index=False,
)

pd.DataFrame(
    supporting
).to_csv(
    TABLE_DIR
    / "MASTER_SUPPORTING_ANALYSES.csv",
    index=False,
)


# ============================================================
# DATASET SUMMARY
# ============================================================

dataset_summary = []

for dataset in [
    "CBIS-DDSM",
    "BreaKHis",
    "DMR-IR Healthy/Sick",
    "Mendeley Benign/Malignant",
]:

    subset = master[
        master[
            "dataset"
        ]
        ==
        dataset
    ]

    dataset_summary.append({

        "dataset":
            dataset,

        "rows":
            len(subset),

        "models":
            ", ".join(
                sorted(
                    subset[
                        "model"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),

        "levels":
            ", ".join(
                sorted(
                    subset[
                        "level"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),

    })


pd.DataFrame(
    dataset_summary
).to_csv(
    TABLE_DIR
    / "MASTER_DATASET_SUMMARY.csv",
    index=False,
)


# ============================================================
# RESEARCH-CLAIM GUARDRAILS
# ============================================================

guardrails = {

    "no_posthoc_model_selection":
        True,

    "no_external_training":
        True,

    "no_external_calibration":
        True,

    "cbis_primary_evaluation_frozen":
        True,

    "external_datasets_as_transfer_stress_tests":
        True,

    "dmr_ir_and_mendeley_kept_separate":
        True,

    "breakhis_as_histopathology_cross_domain_test":
        True,

    "thermography_as_cross_modal_test":
        True,

    "uncertainty_not_equated_with_bayesian_posterior":
        True,

    "quantum_noise_not_claimed_to_improve_accuracy":
        True,

    "publication_numbers_should_trace_to_saved_artifacts":
        True,
}


# ============================================================
# JSON
# ============================================================

result = {

    "status":
        "STEP43_COMPLETE",

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "completed_runs":
        run_status,

    "master_results_rows":
        int(
            len(master)
        ),

    "statistical_rows":
        int(
            len(primary_stats)
        ),

    "master_results":
        master.to_dict(
            orient="records"
        ),

    "supporting_analyses":
        supporting,

    "guardrails":
        guardrails,

    "artifacts":
        {

            "master_results":
                str(
                    TABLE_DIR
                    / "MASTER_PUBLICATION_RESULTS.csv"
                ),

            "bootstrap":
                str(
                    TABLE_DIR
                    / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv"
                ),

            "supporting":
                str(
                    TABLE_DIR
                    / "MASTER_SUPPORTING_ANALYSES.csv"
                ),

            "dataset_summary":
                str(
                    TABLE_DIR
                    / "MASTER_DATASET_SUMMARY.csv"
                ),
        },
}


(
    METRIC_DIR
    / "STEP43_MASTER_RESULTS.json"
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
        "single authoritative numerical source for manuscript",

    "training_performed":
        False,

    "model_changes":
        False,

    "datasets":
        [
            "CBIS-DDSM",
            "BreaKHis",
            "DMR-IR Healthy/Sick",
            "Mendeley Benign/Malignant",
        ],

    "source_runs":
        {
            k:
                str(v)
            for k, v
            in RUNS.items()
        },

    "bootstrap_replicates":
        2000,

    "confidence_level":
        0.95,
}


(
    CONFIG_DIR
    / "STEP43_CONFIGURATION.json"
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
            path.stat().st_size,

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
    "STEP 43 COMPLETE"
)
print("=" * 100)

print()

print(
    "Master results:",
    TABLE_DIR
    / "MASTER_PUBLICATION_RESULTS.csv",
)

print(
    "Bootstrap results:",
    TABLE_DIR
    / "MASTER_BOOTSTRAP_STATISTICAL_RESULTS.csv",
)

print(
    "Supporting analyses:",
    TABLE_DIR
    / "MASTER_SUPPORTING_ANALYSES.csv",
)

print(
    "Dataset summary:",
    TABLE_DIR
    / "MASTER_DATASET_SUMMARY.csv",
)

print(
    "JSON:",
    METRIC_DIR
    / "STEP43_MASTER_RESULTS.json",
)

print()
print(
    "STATUS: STEP43_COMPLETE"
)