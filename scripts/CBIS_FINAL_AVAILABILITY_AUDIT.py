from pathlib import Path
from collections import Counter

import json
import pandas as pd


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

CANONICAL = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv"
)

MAPPED = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING.csv"
)

FAILURES = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING_FAILURES.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = (
    OUT
    / "CBIS_FINAL_AVAILABILITY_AUDIT.json"
)

AVAILABLE = (
    OUT
    / "CBIS_PUBLICATION_AVAILABLE_RECORDS.csv"
)

UNAVAILABLE = (
    OUT
    / "CBIS_PUBLICATION_UNAVAILABLE_RECORDS.csv"
)

PATIENT_SUMMARY = (
    OUT
    / "CBIS_UNAVAILABLE_PATIENT_SUMMARY.csv"
)


def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


print()
print("=" * 100)
print("CBIS FINAL AVAILABILITY / SPLIT-IMPACT AUDIT")
print("=" * 100)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

for path in [
    CANONICAL,
    MAPPED,
    FAILURES,
]:

    if not path.is_file():

        raise RuntimeError(
            f"Required file not found: {path}"
        )

canonical = pd.read_csv(
    CANONICAL
)

mapped = pd.read_csv(
    MAPPED
)

failures = pd.read_csv(
    FAILURES
)

print()
print(
    "Canonical rows:",
    len(canonical)
)

print(
    "Mapped rows:",
    len(mapped)
)

print(
    "Unavailable rows:",
    len(failures)
)

# ------------------------------------------------------------
# Add availability status
# ------------------------------------------------------------

canonical_keys = set(
    canonical.index
)

mapped_rows = mapped.copy()
failure_rows = failures.copy()

# Use patient + source + abnormality + view as stable identity.
def record_key(row):

    return (
        clean(
            row.get(
                "patient_id",
                ""
            )
        ),
        clean(
            row.get(
                "source_table",
                ""
            )
        ),
        clean(
            row.get(
                "abnormality_id",
                ""
            )
        ),
        clean(
            row.get(
                "image_view",
                ""
            )
        ),
        clean(
            row.get(
                "laterality",
                ""
            )
        ),
    )


mapped_keys = {
    record_key(row)
    for _, row in mapped.iterrows()
}

failure_keys = {
    record_key(row)
    for _, row in failures.iterrows()
}

canonical_rows = []

for _, row in canonical.iterrows():

    key = record_key(
        row
    )

    item = row.to_dict()

    if key in mapped_keys:

        item[
            "physical_availability"
        ] = "AVAILABLE"

    elif key in failure_keys:

        item[
            "physical_availability"
        ] = "UNAVAILABLE"

    else:

        item[
            "physical_availability"
        ] = "UNCLASSIFIED"

    canonical_rows.append(
        item
    )

availability = pd.DataFrame(
    canonical_rows
)

# ------------------------------------------------------------
# Sanity
# ------------------------------------------------------------

print()
print("=" * 100)
print("AVAILABILITY COUNTS")
print("=" * 100)

print(
    availability[
        "physical_availability"
    ]
    .value_counts()
    .to_dict()
)

if (
    availability[
        "physical_availability"
    ] == "UNCLASSIFIED"
).any():

    raise RuntimeError(
        "Some canonical records are neither mapped nor "
        "listed as unavailable."
    )

# ------------------------------------------------------------
# Split impact
# ------------------------------------------------------------

print()
print("=" * 100)
print("SPLIT IMPACT")
print("=" * 100)

split_summary = {}

for split in sorted(
    availability[
        "experimental_split"
    ]
    .dropna()
    .unique()
):

    subset = availability[
        availability[
            "experimental_split"
        ]
        == split
    ]

    available = subset[
        subset[
            "physical_availability"
        ]
        == "AVAILABLE"
    ]

    unavailable = subset[
        subset[
            "physical_availability"
        ]
        == "UNAVAILABLE"
    ]

    split_summary[
        str(split)
    ] = {

        "canonical_records":
            int(len(subset)),

        "available_records":
            int(len(available)),

        "unavailable_records":
            int(len(unavailable)),

        "canonical_patients":
            int(
                subset[
                    "patient_id"
                ]
                .nunique()
            ),

        "available_patients":
            int(
                available[
                    "patient_id"
                ]
                .nunique()
            ),

        "unavailable_patients":
            int(
                unavailable[
                    "patient_id"
                ]
                .nunique()
            ),
    }

    print()
    print(
        split
    )

    print(
        "  canonical records:",
        len(subset)
    )

    print(
        "  available records:",
        len(available)
    )

    print(
        "  unavailable records:",
        len(unavailable)
    )

    print(
        "  canonical patients:",
        subset[
            "patient_id"
        ].nunique()
    )

    print(
        "  unavailable patients:",
        unavailable[
            "patient_id"
        ].nunique()
    )

# ------------------------------------------------------------
# Unavailable record details
# ------------------------------------------------------------

unavailable = availability[
    availability[
        "physical_availability"
    ]
    == "UNAVAILABLE"
].copy()

unavailable.to_csv(
    UNAVAILABLE,
    index=False,
)

available = availability[
    availability[
        "physical_availability"
    ]
    == "AVAILABLE"
].copy()

available.to_csv(
    AVAILABLE,
    index=False,
)

# ------------------------------------------------------------
# Failure classifications
# ------------------------------------------------------------

print()
print("=" * 100)
print("UNAVAILABLE RECORD CLASSIFICATIONS")
print("=" * 100)

if "classification" in failures.columns:

    print(
        failures[
            "classification"
        ]
        .value_counts()
        .to_string()
    )

# ------------------------------------------------------------
# Patient-level summary
# ------------------------------------------------------------

group_cols = [
    "patient_id",
    "experimental_split",
    "source_table",
]

patient_rows = []

for keys, group in unavailable.groupby(
    group_cols,
    dropna=False,
):

    patient, split, source = keys

    patient_rows.append({

        "patient_id":
            patient,

        "experimental_split":
            split,

        "source_table":
            source,

        "unavailable_records":
            len(group),

        "views":
            "|".join(
                sorted(
                    set(
                        group[
                            "image_view"
                        ]
                        .astype(str)
                    )
                )
            ),

        "lateralities":
            "|".join(
                sorted(
                    set(
                        group[
                            "laterality"
                        ]
                        .astype(str)
                    )
                )
            ),
    })

patient_summary = pd.DataFrame(
    patient_rows
)

patient_summary.to_csv(
    PATIENT_SUMMARY,
    index=False,
)

print()
print(
    "Unavailable patients:",
    len(patient_summary)
)

# ------------------------------------------------------------
# Label balance impact
# ------------------------------------------------------------

print()
print("=" * 100)
print("LABEL IMPACT")
print("=" * 100)

label_summary = {}

for split in sorted(
    availability[
        "experimental_split"
    ]
    .dropna()
    .unique()
):

    subset = availability[
        availability[
            "experimental_split"
        ]
        == split
    ]

    available_subset = subset[
        subset[
            "physical_availability"
        ]
        == "AVAILABLE"
    ]

    unavailable_subset = subset[
        subset[
            "physical_availability"
        ]
        == "UNAVAILABLE"
    ]

    print()
    print(
        split
    )

    print(
        "  canonical labels:",
        Counter(
            subset[
                "binary_label"
            ]
        )
    )

    print(
        "  available labels:",
        Counter(
            available_subset[
                "binary_label"
            ]
        )
    )

    print(
        "  unavailable labels:",
        Counter(
            unavailable_subset[
                "binary_label"
            ]
        )
    )

    label_summary[
        str(split)
    ] = {

        "canonical":
            {
                str(k):
                    int(v)
                for k, v
                in Counter(
                    subset[
                        "binary_label"
                    ]
                ).items()
            },

        "available":
            {
                str(k):
                    int(v)
                for k, v
                in Counter(
                    available_subset[
                        "binary_label"
                    ]
                ).items()
            },

        "unavailable":
            {
                str(k):
                    int(v)
                for k, v
                in Counter(
                    unavailable_subset[
                        "binary_label"
                    ]
                ).items()
            },
    }

# ------------------------------------------------------------
# Patient-disjointness of the AVAILABLE cohort
# ------------------------------------------------------------

available_train = set(
    available[
        available[
            "experimental_split"
        ]
        == "train"
    ][
        "patient_id"
    ]
)

available_cal = set(
    available[
        available[
            "experimental_split"
        ]
        == "calibration"
    ][
        "patient_id"
    ]
)

available_test = set(
    available[
        available[
            "experimental_split"
        ]
        == "internal_test"
    ][
        "patient_id"
    ]
)

overlaps = {

    "train_calibration":
        sorted(
            available_train
            &
            available_cal
        ),

    "train_test":
        sorted(
            available_train
            &
            available_test
        ),

    "calibration_test":
        sorted(
            available_cal
            &
            available_test
        ),
}

print()
print("=" * 100)
print("AVAILABLE-COHORT PATIENT DISJOINTNESS")
print("=" * 100)

print(
    "Train ∩ calibration:",
    len(
        overlaps[
            "train_calibration"
        ]
    )
)

print(
    "Train ∩ test:",
    len(
        overlaps[
            "train_test"
        ]
    )
)

print(
    "Calibration ∩ test:",
    len(
        overlaps[
            "calibration_test"
        ]
    )
)

# ------------------------------------------------------------
# Publication decision logic
# ------------------------------------------------------------

test_unavailable = split_summary.get(
    "internal_test",
    {}
).get(
    "unavailable_records",
    0,
)

train_unavailable = split_summary.get(
    "train",
    {}
).get(
    "unavailable_records",
    0,
)

cal_unavailable = split_summary.get(
    "calibration",
    {}
).get(
    "unavailable_records",
    0,
)

if (
    len(overlaps["train_calibration"]) == 0
    and
    len(overlaps["train_test"]) == 0
    and
    len(overlaps["calibration_test"]) == 0
):

    disjoint = True

else:

    disjoint = False


print()
print("=" * 100)
print("PUBLICATION IMPACT")
print("=" * 100)

print(
    "Unavailable train records:",
    train_unavailable
)

print(
    "Unavailable calibration records:",
    cal_unavailable
)

print(
    "Unavailable internal-test records:",
    test_unavailable
)

print(
    "Available cohort patient-disjoint:",
    disjoint
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

report = {

    "canonical_records":
        int(len(canonical)),

    "available_records":
        int(len(available)),

    "unavailable_records":
        int(len(unavailable)),

    "unavailable_patients":
        int(len(patient_summary)),

    "split_summary":
        split_summary,

    "label_summary":
        label_summary,

    "available_patient_overlap":
        {
            k:
                int(len(v))
            for k, v
            in overlaps.items()
        },

    "files":
        {
            "available":
                str(AVAILABLE),

            "unavailable":
                str(UNAVAILABLE),

            "patient_summary":
                str(PATIENT_SUMMARY),
        },

    "status":
        "FINAL_AVAILABILITY_AUDIT_COMPLETE",
}

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 100)
print(
    "FINAL AVAILABILITY AUDIT COMPLETE"
)
print("=" * 100)

print()
print(
    "Available records:",
    AVAILABLE
)

print(
    "Unavailable records:",
    UNAVAILABLE
)

print(
    "Patient summary:",
    PATIENT_SUMMARY
)

print(
    "Report:",
    REPORT
)

print()
print(
    "NO TRAINING PERFORMED."
)

print(
    "NO DATA MODIFIED."
)