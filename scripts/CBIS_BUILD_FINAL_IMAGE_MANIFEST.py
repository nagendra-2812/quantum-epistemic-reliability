from pathlib import Path
import json
import re
from collections import defaultdict

import pandas as pd


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

FINAL_MANIFEST = (
    OUT
    / "CBIS_FINAL_PHYSICAL_IMAGE_MANIFEST.csv"
)

FAILURES = (
    OUT
    / "CBIS_FINAL_PHYSICAL_IMAGE_MANIFEST_FAILURES.csv"
)

REPORT = (
    OUT
    / "CBIS_FINAL_PHYSICAL_IMAGE_MANIFEST_REPORT.json"
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


def extract_case(text):

    s = clean(
        text
    ).lower()

    m = re.search(
        r"(calc|mass)-"
        r"(training|test)_"
        r"(p[_-]?\d+)"
        r"_"
        r"(left|right)"
        r"_"
        r"(cc|mlo)"
        r"(?:_(\d+))?",
        s,
    )

    if not m:
        return None

    lesion = m.group(1)
    split = m.group(2)
    patient = m.group(3)
    side = m.group(4)
    view = m.group(5)
    case_num = m.group(6) or ""

    return {

        "lesion":
            lesion,

        "split":
            split,

        "patient":
            patient_norm(
                patient
            ),

        "laterality":
            side,

        "view":
            view,

        "case_num":
            case_num,

        "case_name":
            s,
    }


# ============================================================
# LOAD CANONICAL MANIFEST
# ============================================================

print()
print("=" * 100)
print("BUILD FINAL CBIS PHYSICAL IMAGE MANIFEST")
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
    len(df),
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = {
    "patient_id",
    "source_table",
    "image_view",
    "laterality",
    "lesion_type",
    "abnormality_id",
    "experimental_split",
    "binary_label",
}

missing = (
    required
    -
    set(df.columns)
)

if missing:

    raise RuntimeError(
        "Missing canonical columns: "
        + str(
            sorted(missing)
        )
    )


# ============================================================
# INDEX LOCAL CASE DIRECTORIES
# ============================================================

print()
print("=" * 100)
print("INDEXING LOCAL CASE DIRECTORIES")
print("=" * 100)

local_cases = []

for source_table, root in (
    SOURCE_ROOTS.items()
):

    if not root.is_dir():

        raise RuntimeError(
            f"Missing local source directory: {root}"
        )

    # A case directory is any directory whose name contains
    # Calc-/Mass- Training/Test and P_....
    dirs = [
        p
        for p in root.rglob("*")
        if p.is_dir()
    ]

    count = 0

    for case_dir in dirs:

        info = extract_case(
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

        info.update({

            "source_table":
                source_table,

            "case_dir":
                case_dir,

            "dicom_files":
                dcm_files,

        })

        local_cases.append(
            info
        )

        count += 1

    print()
    print(
        source_table,
        "case directories:",
        count
    )


print()
print(
    "Total local case objects:",
    len(local_cases)
)


# ============================================================
# BUILD LOOKUP
# ============================================================

lookup = defaultdict(
    list
)

for case in local_cases:

    key = (

        case[
            "source_table"
        ],

        case[
            "patient"
        ],

        case[
            "laterality"
        ],

        case[
            "view"
        ],

    )

    lookup[
        key
    ].append(
        case
    )


# ============================================================
# MAP CANONICAL RECORDS
# ============================================================

print()
print("=" * 100)
print("MAPPING CANONICAL RECORDS")
print("=" * 100)

mapped = []
failures = []

for i, row in df.iterrows():

    source = norm(
        row[
            "source_table"
        ]
    )

    patient = patient_norm(
        row[
            "patient_id"
        ]
    )

    side = norm(
        row[
            "laterality"
        ]
    )

    view = norm(
        row[
            "image_view"
        ]
    )

    lesion = norm(
        row[
            "lesion_type"
        ]
    )

    abnormality_id = clean(
        row[
            "abnormality_id"
        ]
    )

    candidates = lookup.get(
        (
            source,
            patient,
            side,
            view,
        ),
        []
    )

    # --------------------------------------------------------
    # Narrow candidates by lesion type
    # --------------------------------------------------------

    if lesion:

        lesion_candidates = [
            c
            for c in candidates
            if c[
                "lesion"
            ] == lesion
        ]

        if lesion_candidates:

            candidates = (
                lesion_candidates
            )

    # --------------------------------------------------------
    # Candidate diagnosis
    # --------------------------------------------------------

    if len(candidates) == 0:

        failures.append({

            **row.to_dict(),

            "failure_reason":
                "NO_CASE_MATCH",

            "candidate_count":
                0,

        })

    elif len(candidates) > 1:

        # We must not silently choose one.
        #
        # Compare abnormality/case numbering when available.

        case_num = ""

        m = re.search(
            r"_(\d+)$",
            abnormality_id,
        )

        if m:
            case_num = m.group(1)

        narrowed = [
            c
            for c in candidates
            if (
                case_num
                and
                c[
                    "case_num"
                ]
                == case_num
            )
        ]

        if len(
            narrowed
        ) == 1:

            candidates = narrowed

        else:

            failures.append({

                **row.to_dict(),

                "failure_reason":
                    "AMBIGUOUS_CASE_MATCH",

                "candidate_count":
                    len(
                        candidates
                    ),

                "candidate_directories":
                    "|".join(
                        str(
                            c[
                                "case_dir"
                            ]
                        )
                        for c
                        in candidates
                    ),

            })

            continue

    # --------------------------------------------------------
    # Unique match
    # --------------------------------------------------------

    case = candidates[0]

    dcm_files = case[
        "dicom_files"
    ]

    row_dict = row.to_dict()

    row_dict.update({

        "physical_case_directory":
            str(
                case[
                    "case_dir"
                ]
            ),

        "physical_dicom_count":
            len(
                dcm_files
            ),

        "physical_dicom_paths":
            "|".join(
                str(x)
                for x in dcm_files
            ),

        "physical_mapping_status":
            "MATCHED_UNIQUE_CASE",

        "physical_case_name":
            case[
                "case_name"
            ],

        "physical_case_number":
            case[
                "case_num"
            ],

    })

    mapped.append(
        row_dict
    )

    if (
        (i + 1) % 250
        == 0
    ):

        print(
            f"Processed {i + 1}/{len(df)}...",
            flush=True,
        )


mapped_df = pd.DataFrame(
    mapped
)

failure_df = pd.DataFrame(
    failures
)


# ============================================================
# CHECKS
# ============================================================

print()
print("=" * 100)
print("FINAL MAPPING CHECK")
print("=" * 100)

print(
    "Canonical records:",
    len(df)
)

print(
    "Matched records:",
    len(mapped_df)
)

print(
    "Failures:",
    len(failure_df)
)

# Every mapped path must exist.
missing_physical = []

if len(mapped_df):

    for _, row in (
        mapped_df.iterrows()
    ):

        paths = [
            x
            for x
            in clean(
                row[
                    "physical_dicom_paths"
                ]
            ).split("|")
            if x
        ]

        for path in paths:

            if not Path(
                path
            ).is_file():

                missing_physical.append(
                    path
                )


print(
    "Missing physical DICOM targets:",
    len(
        missing_physical
    )
)

# ------------------------------------------------------------
# Duplicate case assignment check
# ------------------------------------------------------------

case_assignment_counts = (
    mapped_df[
        "physical_case_directory"
    ]
    .value_counts()
    if len(mapped_df)
    else pd.Series(
        dtype=int
    )
)

duplicate_case_assignments = {
    str(k):
        int(v)
    for k, v
    in case_assignment_counts.items()
    if v > 1
}

print(
    "Case directories assigned to multiple manifest records:",
    len(
        duplicate_case_assignments
    )
)


# ============================================================
# SAVE
# ============================================================

mapped_df.to_csv(
    FINAL_MANIFEST,
    index=False,
)

failure_df.to_csv(
    FAILURES,
    index=False,
)

status = (
    "FINAL_MAPPING_PASS"
    if (
        len(mapped_df)
        == len(df)
        and
        len(failure_df)
        == 0
        and
        len(missing_physical)
        == 0
        and
        len(
            duplicate_case_assignments
        )
        == 0
    )
    else
    "FINAL_MAPPING_REQUIRES_REVIEW"
)

report = {

    "canonical_rows":
        int(len(df)),

    "matched_rows":
        int(len(mapped_df)),

    "failure_rows":
        int(len(failure_df)),

    "missing_physical_targets":
        int(len(missing_physical)),

    "duplicate_case_assignments":
        duplicate_case_assignments,

    "status":
        status,

    "final_manifest":
        str(FINAL_MANIFEST),

    "failure_manifest":
        str(FAILURES),
}

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("FINAL CBIS PHYSICAL IMAGE MANIFEST")
print("=" * 100)

print()
print(
    "Manifest:",
    FINAL_MANIFEST
)

print(
    "Failures:",
    FAILURES
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

if status == "FINAL_MAPPING_PASS":

    print()
    print(
        "ALL CANONICAL RECORDS HAVE VERIFIED LOCAL"
    )

    print(
        "CASE-LEVEL DICOM OBJECTS."
    )

    print()
    print(
        "READY FOR STEP 34A-v2 TRAINING."
    )

else:

    print()
    print(
        "DO NOT START TRAINING."
    )
