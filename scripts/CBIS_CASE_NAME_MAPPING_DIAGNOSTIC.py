from pathlib import Path
from collections import defaultdict
import json
import re

import pandas as pd


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv"
)

INDEX = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_DICOM_INDEX.csv"
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
    / "CBIS_CASE_NAME_MAPPING_DIAGNOSTIC.json"
)

SAMPLES = (
    OUT
    / "CBIS_CASE_NAME_MAPPING_SAMPLES.csv"
)


def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def norm(x):

    return clean(x).lower()


def normalize_patient(x):

    s = clean(x).lower()

    m = re.search(
        r"p[_-]?(\d+)",
        s,
    )

    if m:

        return (
            "p_"
            + m.group(1).zfill(5)
        )

    return s


def extract_case_info(text):

    s = clean(
        text
    )

    low = s.lower()

    patient = ""

    m = re.search(
        r"p[_-]?(\d+)",
        low,
    )

    if m:

        patient = (
            "p_"
            + m.group(1).zfill(5)
        )

    laterality = ""

    if "right" in low:
        laterality = "right"

    elif "left" in low:
        laterality = "left"

    view = ""

    # Prefer explicit view tokens.
    if re.search(
        r"(^|[_-])cc([_-]|$)",
        low,
    ):

        view = "cc"

    elif re.search(
        r"(^|[_-])mlo([_-]|$)",
        low,
    ):

        view = "mlo"

    return {
        "patient":
            patient,

        "laterality":
            laterality,

        "view":
            view,
    }


print()
print("=" * 100)
print("CBIS CASE-NAME MAPPING DIAGNOSTIC")
print("=" * 100)

manifest = pd.read_csv(
    MANIFEST
)

index = pd.read_csv(
    INDEX
)

print()
print(
    "Manifest rows:",
    len(manifest),
)

print(
    "DICOM index rows:",
    len(index),
)


# ============================================================
# MANIFEST CASE SIGNATURES
# ============================================================

manifest_rows = []

for i, row in manifest.iterrows():

    path = clean(
        row[
            "image_file_path_metadata"
        ]
    )

    info = extract_case_info(
        path
    )

    patient_manifest = normalize_patient(
        row[
            "patient_id"
        ]
    )

    manifest_rows.append({

        "manifest_row":
            int(i),

        "patient_id":
            clean(
                row[
                    "patient_id"
                ]
            ),

        "patient_norm":
            patient_manifest,

        "source_table":
            clean(
                row[
                    "source_table"
                ]
            ),

        "lesion_type":
            clean(
                row[
                    "lesion_type"
                ]
            ),

        "image_view_manifest":
            clean(
                row[
                    "image_view"
                ]
            ),

        "laterality_manifest":
            clean(
                row[
                    "laterality"
                ]
            ),

        "abnormality_id":
            clean(
                row[
                    "abnormality_id"
                ]
            ),

        "metadata_path":
            path,

        "path_patient":
            info[
                "patient"
            ],

        "path_laterality":
            info[
                "laterality"
            ],

        "path_view":
            info[
                "view"
            ],
    })


m = pd.DataFrame(
    manifest_rows
)


# ============================================================
# LOCAL DICOM CASE SIGNATURES
# ============================================================

local_rows = []

for i, row in index.iterrows():

    physical = clean(
        row[
            "physical_path"
        ]
    )

    physical_lower = (
        physical
        .lower()
        .replace(
            "\\",
            "/",
        )
    )

    # Take path component immediately above
    # the numeric folders.
    parts = [
        p
        for p in physical_lower.split("/")
        if p
    ]

    case_component = ""

    for part in parts:

        if (
            "training_p_" in part
            or "test_p_" in part
        ):

            case_component = part

    patient_header = normalize_patient(
        row[
            "patient_id_dicom"
        ]
    )

    info = extract_case_info(
        case_component
    )

    source = ""

    if (
        "calc-training" in case_component
    ):
        source = "calc_train"

    elif (
        "calc-test" in case_component
    ):
        source = "calc_test"

    elif (
        "mass-training" in case_component
    ):
        source = "mass_train"

    elif (
        "mass-test" in case_component
    ):
        source = "mass_test"

    local_rows.append({

        "index_row":
            int(i),

        "physical_path":
            physical,

        "case_component":
            case_component,

        "source_table":
            source,

        "patient_header":
            patient_header,

        "case_patient":
            info[
                "patient"
            ],

        "case_laterality":
            info[
                "laterality"
            ],

        "case_view":
            info[
                "view"
            ],

        "view_position":
            clean(
                row[
                    "view_position"
                ]
            ),
    })


l = pd.DataFrame(
    local_rows
)


# ============================================================
# NORMALIZATION
# ============================================================

for df in [
    m,
    l,
]:

    for c in df.columns:

        if c.endswith(
            "_view"
        ):

            df[c] = (
                df[c]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        if c.endswith(
            "_laterality"
        ):

            df[c] = (
                df[c]
                .astype(str)
                .str.lower()
                .str.strip()
            )


# ============================================================
# SIGNATURE COUNTS
# ============================================================

print()
print("=" * 100)
print("MANIFEST SIGNATURE EXAMPLES")
print("=" * 100)

print(
    m[
        [
            "patient_id",
            "source_table",
            "lesion_type",
            "image_view_manifest",
            "laterality_manifest",
            "abnormality_id",
            "metadata_path",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("LOCAL CASE SIGNATURE EXAMPLES")
print("=" * 100)

print(
    l[
        [
            "source_table",
            "case_component",
            "patient_header",
            "case_laterality",
            "case_view",
            "view_position",
            "physical_path",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# MATCH PATIENT + LATERALITY + VIEW + SOURCE
# ============================================================

print()
print("=" * 100)
print(
    "CASE-SIGNATURE MATCH TEST"
)
print("=" * 100)

sample_matches = []

match_counts = defaultdict(
    int
)

for _, mr in m.head(100).iterrows():

    patient = (
        mr[
            "patient_norm"
        ]
    )

    source = norm(
        mr[
            "source_table"
        ]
    )

    laterality = norm(
        mr[
            "laterality_manifest"
        ]
    )

    view = norm(
        mr[
            "image_view_manifest"
        ]
    )

    candidates = l[
        (
            l[
                "source_table"
            ]
            == source
        )
        &
        (
            l[
                "patient_header"
            ]
            == patient
        )
        &
        (
            (
                l[
                    "case_laterality"
                ]
                == laterality
            )
            |
            (
                l[
                    "case_laterality"
                ]
                == ""
            )
        )
        &
        (
            (
                l[
                    "case_view"
                ]
                == view
            )
            |
            (
                l[
                    "view_position"
                ]
                .astype(str)
                .str.lower()
                == view
            )
        )
    ]

    n = len(
        candidates
    )

    match_counts[
        str(n)
    ] += 1

    if len(
        sample_matches
    ) < 50:

        sample_matches.append({

            "manifest_row":
                mr[
                    "manifest_row"
                ],

            "patient":
                mr[
                    "patient_id"
                ],

            "source":
                source,

            "laterality":
                laterality,

            "view":
                view,

            "candidate_count":
                n,

            "candidate_examples":
                "|".join(
                    candidates[
                        "physical_path"
                    ]
                    .head(5)
                    .tolist()
                ),
        })


print()
print(
    "First-100 candidate-count distribution:"
)

for k, v in sorted(
    match_counts.items(),
    key=lambda x:
        int(x[0]),
):

    print(
        "  candidates =",
        k,
        ":",
        v,
    )


print()
print("=" * 100)
print("SAMPLE MATCHES")
print("=" * 100)

sample_df = pd.DataFrame(
    sample_matches
)

print(
    sample_df.to_string(
        index=False
    )
)

sample_df.to_csv(
    SAMPLES,
    index=False,
)


# ============================================================
# UNIQUE PATIENT CASE COUNTS
# ============================================================

print()
print("=" * 100)
print("PATIENT COVERAGE")
print("=" * 100)

manifest_patients = set(
    m[
        "patient_norm"
    ]
)

local_patients = set(
    l[
        "patient_header"
    ]
)

print(
    "Manifest normalized patients:",
    len(manifest_patients)
)

print(
    "Local normalized patients:",
    len(local_patients)
)

print(
    "Patient intersection:",
    len(
        manifest_patients
        &
        local_patients
    )
)

print(
    "Manifest patients not in local:",
    len(
        manifest_patients
        -
        local_patients
    )
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "manifest_rows":
        int(len(manifest)),

    "local_dicom_rows":
        int(len(index)),

    "manifest_unique_normalized_patients":
        int(len(manifest_patients)),

    "local_unique_normalized_patients":
        int(len(local_patients)),

    "patient_intersection":
        int(
            len(
                manifest_patients
                &
                local_patients
            )
        ),

    "manifest_patients_not_local":
        int(
            len(
                manifest_patients
                -
                local_patients
            )
        ),

    "first_100_candidate_count_distribution":
        dict(
            match_counts
        ),

    "sample_csv":
        str(
            SAMPLES
        ),

    "status":
        "CASE_NAME_DIAGNOSTIC_COMPLETE",
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
print("CASE-NAME DIAGNOSTIC COMPLETE")
print("=" * 100)

print()
print(
    "Samples:",
    SAMPLES,
)

print(
    "Report:",
    REPORT,
)

print()
print(
    "NO TRAINING PERFORMED."
)

print(
    "NO DATA MODIFIED."
)