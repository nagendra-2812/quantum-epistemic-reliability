from pathlib import Path
from collections import defaultdict, Counter
import json
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

CANONICAL = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv"
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

MAPPED = (
    OUT
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING.csv"
)

FAILURES = (
    OUT
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING_FAILURES.csv"
)

CASE_INDEX = (
    OUT
    / "CBIS_LOCAL_CASE_INDEX.csv"
)

CONFLICTS = (
    OUT
    / "CBIS_PHYSICAL_LABEL_CONFLICTS.csv"
)

REPORT = (
    OUT
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING_REPORT.json"
)


SOURCE_ROOTS = {
    "calc_train":
        ROOT / "calc_train",

    "calc_test":
        ROOT / "calc_test",

    "mass_train":
        ROOT / "mass_train",

    "mass_test":
        ROOT / "mass_test",
}


# ============================================================
# HELPERS
# ============================================================

def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def norm(x):

    return clean(x).lower()


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


def extract_case_name(name):

    s = clean(name)

    low = s.lower()

    # Examples:
    # Calc-Training_P_00005_RIGHT_CC_1
    # Mass-Test_P_00016_LEFT_MLO_2

    pattern = (
        r"^(calc|mass)-"
        r"(training|test)_"
        r"(p[_-]?\d+)_"
        r"(left|right)_"
        r"(cc|mlo)"
        r"(?:_(\d+))?$"
    )

    m = re.match(
        pattern,
        low,
    )

    if not m:
        return None

    lesion = m.group(1)
    split = m.group(2)
    patient = patient_norm(
        m.group(3)
    )
    laterality = m.group(4)
    view = m.group(5)
    case_number = (
        m.group(6)
        or ""
    )

    return {

        "lesion":
            lesion,

        "split":
            split,

        "patient":
            patient,

        "laterality":
            laterality,

        "view":
            view,

        "case_number":
            case_number,

        "case_name":
            s,

    }


def get_manifest_signature(row):

    return {

        "source_table":
            norm(
                row[
                    "source_table"
                ]
            ),

        "patient":
            patient_norm(
                row[
                    "patient_id"
                ]
            ),

        "laterality":
            norm(
                row[
                    "laterality"
                ]
            ),

        "view":
            norm(
                row[
                    "image_view"
                ]
            ),

        "lesion_type":
            norm(
                row[
                    "lesion_type"
                ]
            ),

        "abnormality_id":
            clean(
                row[
                    "abnormality_id"
                ]
            ),

    }


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print(
    "CBIS FINAL ROBUST PHYSICAL MAPPING"
)
print("=" * 100)

if not CANONICAL.is_file():

    raise RuntimeError(
        f"Canonical manifest not found: {CANONICAL}"
    )

df = pd.read_csv(
    CANONICAL
)

print()
print(
    "Canonical rows:",
    len(df)
)


# ============================================================
# INDEX LOCAL CASE DIRECTORIES
# ============================================================

print()
print("=" * 100)
print(
    "INDEXING LOCAL CASE DIRECTORIES"
)
print("=" * 100)

local_cases = []

for source_table, root in (
    SOURCE_ROOTS.items()
):

    if not root.is_dir():

        raise RuntimeError(
            f"Missing source root: {root}"
        )

    all_dirs = [
        p
        for p in root.rglob("*")
        if p.is_dir()
    ]

    valid_count = 0

    for case_dir in all_dirs:

        info = extract_case_name(
            case_dir.name
        )

        if info is None:
            continue

        dcm_files = sorted(
            p
            for p in case_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            == ".dcm"
        )

        if not dcm_files:
            continue

        local_cases.append({

            "source_table":
                source_table,

            "case_directory":
                str(case_dir),

            "case_name":
                info[
                    "case_name"
                ],

            "lesion":
                info[
                    "lesion"
                ],

            "split":
                info[
                    "split"
                ],

            "patient":
                info[
                    "patient"
                ],

            "laterality":
                info[
                    "laterality"
                ],

            "view":
                info[
                    "view"
                ],

            "case_number":
                info[
                    "case_number"
                ],

            "dicom_files":
                [
                    str(p)
                    for p
                    in dcm_files
                ],

            "dicom_count":
                len(
                    dcm_files
                ),
        })

        valid_count += 1

    print()
    print(
        source_table,
        "case objects:",
        valid_count
    )


local_df = pd.DataFrame(
    local_cases
)

if len(local_df) == 0:

    raise RuntimeError(
        "No valid local CBIS case directories found."
    )

local_df.to_csv(
    CASE_INDEX,
    index=False,
)

print()
print(
    "TOTAL LOCAL CASE OBJECTS:",
    len(local_df)
)


# ============================================================
# LOOKUP
# ============================================================

lookup = defaultdict(
    list
)

for _, row in local_df.iterrows():

    key = (

        row[
            "source_table"
        ],

        row[
            "patient"
        ],

        row[
            "laterality"
        ],

        row[
            "view"
        ],

    )

    lookup[
        key
    ].append(
        row.to_dict()
    )


# ============================================================
# MAP MANIFEST
# ============================================================

print()
print("=" * 100)
print(
    "ROBUST MANIFEST MAPPING"
)
print("=" * 100)

mapped_rows = []
failure_rows = []

mapping_counts = Counter()

for idx, row in df.iterrows():

    sig = get_manifest_signature(
        row
    )

    key = (

        sig[
            "source_table"
        ],

        sig[
            "patient"
        ],

        sig[
            "laterality"
        ],

        sig[
            "view"
        ],

    )

    candidates = lookup.get(
        key,
        []
    )

    # --------------------------------------------------------
    # Never crash on zero candidates.
    # --------------------------------------------------------

    if len(candidates) == 0:

        failure = row.to_dict()

        failure[
            "failure_reason"
        ] = "NO_PATIENT_LATERALITY_VIEW_MATCH"

        failure[
            "candidate_count"
        ] = 0

        failure[
            "candidate_case_directories"
        ] = ""

        failure_rows.append(
            failure
        )

        if (
            (idx + 1) % 250
            == 0
        ):

            print(
                f"Processed {idx + 1}/{len(df)}...",
                flush=True,
            )

        continue

    # --------------------------------------------------------
    # If multiple case objects exist, preserve ALL of them.
    #
    # We do NOT arbitrarily choose one.
    # --------------------------------------------------------

    case_dirs = [
        c[
            "case_directory"
        ]
        for c in candidates
    ]

    case_names = [
        c[
            "case_name"
        ]
        for c in candidates
    ]

    all_dcms = []

    for c in candidates:

        all_dcms.extend(
            c[
                "dicom_files"
            ]
        )

    # Remove duplicate files
    all_dcms = sorted(
        set(
            all_dcms
        )
    )

    output = row.to_dict()

    output[
        "physical_case_count"
    ] = len(
        candidates
    )

    output[
        "physical_case_directories"
    ] = "|".join(
        case_dirs
    )

    output[
        "physical_case_names"
    ] = "|".join(
        case_names
    )

    output[
        "physical_dicom_count"
    ] = len(
        all_dcms
    )

    output[
        "physical_dicom_paths"
    ] = "|".join(
        all_dcms
    )

    output[
        "physical_mapping_status"
    ] = (
        "MATCHED_SINGLE_CASE"
        if len(candidates) == 1
        else
        "MATCHED_MULTIPLE_CASES"
    )

    mapped_rows.append(
        output
    )

    mapping_counts[
        output[
            "physical_mapping_status"
        ]
    ] += 1

    if (
        (idx + 1) % 250
        == 0
    ):

        print(
            f"Processed {idx + 1}/{len(df)}...",
            flush=True,
        )


mapped_df = pd.DataFrame(
    mapped_rows
)

failures_df = pd.DataFrame(
    failure_rows
)


# ============================================================
# EXISTENCE CHECK
# ============================================================

print()
print("=" * 100)
print(
    "PHYSICAL FILE EXISTENCE CHECK"
)
print("=" * 100)

missing_targets = []

if len(mapped_df):

    for _, row in mapped_df.iterrows():

        paths = [
            p
            for p
            in clean(
                row[
                    "physical_dicom_paths"
                ]
            ).split("|")
            if p
        ]

        for p in paths:

            if not Path(
                p
            ).is_file():

                missing_targets.append({
                    "manifest_row":
                        row[
                            "patient_id"
                        ],

                    "physical_path":
                        p,
                })


# ============================================================
# PHYSICAL CASE DUPLICATION IS NOT AN ERROR
# ============================================================

print()
print("=" * 100)
print(
    "MAPPING SUMMARY"
)
print("=" * 100)

print(
    "Canonical records:",
    len(df)
)

print(
    "Mapped records:",
    len(mapped_df)
)

print(
    "Unmapped records:",
    len(failures_df)
)

print(
    "Mapping categories:",
    dict(
        mapping_counts
    )
)

print(
    "Missing physical DICOM targets:",
    len(missing_targets)
)


# ============================================================
# FAILURE BREAKDOWN
# ============================================================

if len(failures_df):

    print()
    print(
        "FAILURE REASONS:"
    )

    print(
        failures_df[
            "failure_reason"
        ]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "FIRST 30 UNMAPPED RECORDS:"
    )

    show_columns = [
        c
        for c in [
            "patient_id",
            "source_table",
            "lesion_type",
            "laterality",
            "image_view",
            "abnormality_id",
            "image_file_path_metadata",
            "failure_reason",
        ]
        if c in failures_df.columns
    ]

    print(
        failures_df[
            show_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# LABEL / PHYSICAL-CASE CONFLICT AUDIT
#
# Same physical case may occur in multiple manifest rows.
# That is acceptable only if labels are consistent.
# ============================================================

print()
print("=" * 100)
print(
    "PHYSICAL CASE LABEL CONSISTENCY AUDIT"
)
print("=" * 100)

case_labels = defaultdict(
    set
)

case_manifest_rows = defaultdict(
    list
)

for _, row in mapped_df.iterrows():

    case_paths = [
        x
        for x
        in clean(
            row[
                "physical_case_directories"
            ]
        ).split("|")
        if x
    ]

    label = clean(
        row[
            "binary_label"
        ]
    )

    for case_path in case_paths:

        case_labels[
            case_path
        ].add(
            label
        )

        case_manifest_rows[
            case_path
        ].append(
            int(
                row.get(
                    "abnormality_id",
                    -1
                )
                if str(
                    row.get(
                        "abnormality_id",
                        ""
                    )
                ).isdigit()
                else -1
            )
        )


conflict_rows = []

for case_path, labels in (
    case_labels.items()
):

    if len(labels) > 1:

        conflict_rows.append({

            "physical_case_directory":
                case_path,

            "labels":
                "|".join(
                    sorted(
                        labels
                    )
                ),

            "manifest_abnormality_ids":
                "|".join(
                    str(x)
                    for x
                    in sorted(
                        set(
                            case_manifest_rows[
                                case_path
                            ]
                        )
                    )
                ),
        })


conflicts_df = pd.DataFrame(
    conflict_rows
)

conflicts_df.to_csv(
    CONFLICTS,
    index=False,
)

print(
    "Physical cases with conflicting binary labels:",
    len(
        conflicts_df
    )
)

if len(conflicts_df):

    print()
    print(
        conflicts_df.head(30).to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

mapped_df.to_csv(
    MAPPED,
    index=False,
)

failures_df.to_csv(
    FAILURES,
    index=False,
)


# ============================================================
# FINAL STATUS
# ============================================================

status = (
    "FULL_MAPPING_PASS"
    if (
        len(mapped_df)
        == len(df)
        and
        len(missing_targets)
        == 0
    )
    else
    "MAPPING_REQUIRES_REVIEW"
)

report = {

    "canonical_records":
        int(
            len(df)
        ),

    "mapped_records":
        int(
            len(mapped_df)
        ),

    "unmapped_records":
        int(
            len(failures_df)
        ),

    "local_case_objects":
        int(
            len(local_df)
        ),

    "mapping_categories":
        dict(
            mapping_counts
        ),

    "missing_physical_targets":
        int(
            len(missing_targets)
        ),

    "physical_case_label_conflicts":
        int(
            len(conflicts_df)
        ),

    "status":
        status,

    "mapped_file":
        str(MAPPED),

    "failure_file":
        str(FAILURES),

    "case_index":
        str(CASE_INDEX),

    "label_conflicts":
        str(CONFLICTS),
}

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL CONSOLE
# ============================================================

print()
print("=" * 100)
print(
    "CBIS FINAL ROBUST MAPPING COMPLETE"
)
print("=" * 100)

print()
print(
    "Mapped:",
    MAPPED
)

print(
    "Failures:",
    FAILURES
)

print(
    "Case index:",
    CASE_INDEX
)

print(
    "Label conflicts:",
    CONFLICTS
)

print(
    "Report:",
    REPORT
)

print()
print(
    "STATUS:",
    status
)

if status == "FULL_MAPPING_PASS":

    print()
    print(
        "ALL 3568 MANIFEST RECORDS HAVE PHYSICAL DICOM MAPPINGS."
    )

    print(
        "READY FOR FINAL CBIS IMAGE-INPUT AUDIT."
    )

else:

    print()
    print(
        "TRAINING NOT STARTED."
    )

    print(
        "UNMAPPED RECORDS WERE PRESERVED FOR REVIEW."
    )
