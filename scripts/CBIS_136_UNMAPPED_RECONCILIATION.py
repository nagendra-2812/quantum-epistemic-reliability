from pathlib import Path
from collections import defaultdict
import json
import re

import pandas as pd


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

FAILURES = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING_FAILURES.csv"
)

CASE_INDEX = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_LOCAL_CASE_INDEX.csv"
)

MAPPED = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
)

REPORT = (
    OUT
    / "CBIS_136_UNMAPPED_RECONCILIATION_REPORT.json"
)

DETAIL = (
    OUT
    / "CBIS_136_UNMAPPED_RECONCILIATION.csv"
)


def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def norm(x):

    return clean(x).lower().strip()


def patient_norm(x):

    s = norm(x)

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


print()
print("=" * 100)
print("CBIS 136 UNMAPPED RECORD RECONCILIATION")
print("=" * 100)

if not FAILURES.is_file():
    raise RuntimeError(
        f"Failure file not found: {FAILURES}"
    )

if not CASE_INDEX.is_file():
    raise RuntimeError(
        f"Case index not found: {CASE_INDEX}"
    )

fail = pd.read_csv(
    FAILURES
)

cases = pd.read_csv(
    CASE_INDEX
)

mapped = pd.read_csv(
    MAPPED
)

print()
print(
    "Unmapped manifest rows:",
    len(fail),
)

print(
    "Local case objects:",
    len(cases),
)

print(
    "Already mapped rows:",
    len(mapped),
)


# ============================================================
# NORMALIZE
# ============================================================

for df in [
    fail,
    cases,
]:

    if "patient_id" in df.columns:
        df[
            "patient_norm"
        ] = (
            df[
                "patient_id"
            ]
            .map(patient_norm)
        )

    elif "patient" in df.columns:
        df[
            "patient_norm"
        ] = (
            df[
                "patient"
            ]
            .map(patient_norm)
        )

    if "source_table" in df.columns:
        df[
            "source_norm"
        ] = (
            df[
                "source_table"
            ]
            .map(norm)
        )

    if "laterality" in df.columns:
        df[
            "laterality_norm"
        ] = (
            df[
                "laterality"
            ]
            .map(norm)
        )

    if "view" in df.columns:
        df[
            "view_norm"
        ] = (
            df[
                "view"
            ]
            .map(norm)
        )

    if "image_view" in df.columns:
        df[
            "view_norm"
        ] = (
            df[
                "image_view"
            ]
            .map(norm)
        )


# ============================================================
# DIAGNOSTIC LOOKUP
# ============================================================

by_patient = defaultdict(list)
by_patient_source = defaultdict(list)
by_patient_source_view = defaultdict(list)
by_patient_source_side = defaultdict(list)
by_patient_source_side_view = defaultdict(list)

for _, row in cases.iterrows():

    patient = norm(
        row[
            "patient"
        ]
    )

    source = norm(
        row[
            "source_table"
        ]
    )

    side = norm(
        row[
            "laterality"
        ]
    )

    view = norm(
        row[
            "view"
        ]
    )

    item = row.to_dict()

    by_patient[
        patient
    ].append(
        item
    )

    by_patient_source[
        (
            patient,
            source,
        )
    ].append(
        item
    )

    by_patient_source_view[
        (
            patient,
            source,
            view,
        )
    ].append(
        item
    )

    by_patient_source_side[
        (
            patient,
            source,
            side,
        )
    ].append(
        item
    )

    by_patient_source_side_view[
        (
            patient,
            source,
            side,
            view,
        )
    ].append(
        item
    )


# ============================================================
# CLASSIFY EACH FAILURE
# ============================================================

details = []

classification_counts = defaultdict(int)

for _, row in fail.iterrows():

    patient = norm(
        row[
            "patient_norm"
        ]
    )

    source = norm(
        row[
            "source_norm"
        ]
    )

    side = norm(
        row[
            "laterality_norm"
        ]
    )

    view = norm(
        row[
            "view_norm"
        ]
    )

    exact = by_patient_source_side_view.get(
        (
            patient,
            source,
            side,
            view,
        ),
        []
    )

    patient_source = by_patient_source.get(
        (
            patient,
            source,
        ),
        []
    )

    patient_only = by_patient.get(
        patient,
        []
    )

    patient_side = by_patient_source_side.get(
        (
            patient,
            source,
            side,
        ),
        []
    )

    patient_view = by_patient_source_view.get(
        (
            patient,
            source,
            view,
        ),
        []
    )

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    if exact:

        classification = (
            "EXACT_MATCH_SHOULD_NOT_HAVE_FAILED"
        )

        candidate_group = exact

    elif patient_side:

        classification = (
            "PATIENT_SOURCE_SIDE_EXISTS_VIEW_DIFFERENCE"
        )

        candidate_group = patient_side

    elif patient_view:

        classification = (
            "PATIENT_SOURCE_VIEW_EXISTS_SIDE_DIFFERENCE"
        )

        candidate_group = patient_view

    elif patient_source:

        classification = (
            "PATIENT_SOURCE_EXISTS_NO_SIDE_VIEW_MATCH"
        )

        candidate_group = patient_source

    elif patient_only:

        classification = (
            "PATIENT_EXISTS_DIFFERENT_SOURCE"
        )

        candidate_group = patient_only

    else:

        classification = (
            "PATIENT_NOT_PRESENT_LOCALLY"
        )

        candidate_group = []

    classification_counts[
        classification
    ] += 1

    candidate_paths = []

    candidate_names = []

    candidate_signatures = []

    for c in candidate_group:

        candidate_paths.append(
            clean(
                c[
                    "case_directory"
                ]
            )
        )

        candidate_names.append(
            clean(
                c[
                    "case_name"
                ]
            )
        )

        candidate_signatures.append(
            (
                f"{c['source_table']}"
                f"|{c['patient']}"
                f"|{c['laterality']}"
                f"|{c['view']}"
            )
        )

    details.append({

        "patient_id":
            clean(
                row[
                    "patient_id"
                ]
            ),

        "source_table":
            source,

        "lesion_type":
            clean(
                row.get(
                    "lesion_type",
                    "",
                )
            ),

        "laterality":
            side,

        "image_view":
            view,

        "abnormality_id":
            clean(
                row.get(
                    "abnormality_id",
                    "",
                )
            ),

        "metadata_path":
            clean(
                row.get(
                    "image_file_path_metadata",
                    "",
                )
            ),

        "classification":
            classification,

        "patient_local_case_count":
            len(
                patient_only
            ),

        "patient_source_case_count":
            len(
                patient_source
            ),

        "patient_source_side_case_count":
            len(
                patient_side
            ),

        "patient_source_view_case_count":
            len(
                patient_view
            ),

        "candidate_case_names":
            "|".join(
                candidate_names
            ),

        "candidate_case_signatures":
            "|".join(
                candidate_signatures
            ),

        "candidate_directories":
            "|".join(
                candidate_paths
            ),
    })


# ============================================================
# OUTPUT
# ============================================================

detail_df = pd.DataFrame(
    details
)

detail_df.to_csv(
    DETAIL,
    index=False,
)

print()
print("=" * 100)
print("CLASSIFICATION OF 136 UNMAPPED RECORDS")
print("=" * 100)

for classification, count in sorted(
    classification_counts.items()
):

    print(
        f"{classification}: {count}"
    )


# ============================================================
# SHOW ALL PATIENTS WHERE SOME LOCAL CASE EXISTS
# ============================================================

print()
print("=" * 100)
print("UNMAPPED CASES WITH LOCAL PATIENT INFORMATION")
print("=" * 100)

has_local = detail_df[
    detail_df[
        "classification"
    ]
    != "PATIENT_NOT_PRESENT_LOCALLY"
]

print(
    has_local[
        [
            "patient_id",
            "source_table",
            "laterality",
            "image_view",
            "abnormality_id",
            "classification",
            "candidate_case_signatures",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
)


# ============================================================
# CALC-TEST SPECIFIC SUMMARY
# ============================================================

calc_test = detail_df[
    detail_df[
        "source_table"
    ]
    == "calc_test"
]

print()
print("=" * 100)
print("CALC-TEST UNMAPPED SUMMARY")
print("=" * 100)

print(
    "Calc-test unmapped rows:",
    len(calc_test)
)

print()
print(
    "Distinct calc-test patients:",
    calc_test[
        "patient_id"
    ].nunique()
)

print()
print(
    "Calc-test classifications:"
)

print(
    calc_test[
        "classification"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# VIEW / LATERALITY PATTERN
# ============================================================

print()
print("=" * 100)
print("UNMAPPED VIEW/LATERALITY DISTRIBUTION")
print("=" * 100)

print(
    detail_df[
        [
            "source_table",
            "laterality",
            "image_view",
        ]
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "unmapped_rows":
        int(len(fail)),

    "mapped_rows":
        int(len(mapped)),

    "local_case_objects":
        int(len(cases)),

    "classification_counts":
        dict(
            classification_counts
        ),

    "calc_test_unmapped_rows":
        int(len(calc_test)),

    "calc_test_unmapped_patients":
        int(
            calc_test[
                "patient_id"
            ].nunique()
        ),

    "status":
        "136_RECORD_DIAGNOSTIC_COMPLETE",

    "detail_file":
        str(DETAIL),
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
print("136-RECORD DIAGNOSTIC COMPLETE")
print("=" * 100)

print()
print(
    "Detailed CSV:",
    DETAIL
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