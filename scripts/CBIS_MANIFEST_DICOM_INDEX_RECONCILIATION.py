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

MAPPED = (
    OUT
    / "CBIS_MANIFEST_DICOM_INDEX_RECONCILIATION.csv"
)

FAILURES = (
    OUT
    / "CBIS_MANIFEST_DICOM_INDEX_RECONCILIATION_FAILURES.csv"
)

REPORT = (
    OUT
    / "CBIS_MANIFEST_DICOM_INDEX_RECONCILIATION.json"
)


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm(x):
    return clean(x).lower()


def uid_list(path_text):

    text = (
        clean(path_text)
        .replace("\\", "/")
    )

    parts = [
        p.strip()
        for p in text.split("/")
        if p.strip()
    ]

    return [
        p
        for p in parts
        if p.startswith("1.3.6.1.")
    ]


def patient_tokens(path_text):

    text = (
        clean(path_text)
        .replace("\\", "/")
    )

    parts = [
        p
        for p in text.split("/")
        if p
    ]

    values = []

    for p in parts:

        low = p.lower()

        matches = re.findall(
            r"p[_-]?\d+",
            low,
        )

        values.extend(
            matches
        )

    return values


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 100)
print("CBIS MANIFEST ↔ DICOM INDEX RECONCILIATION")
print("=" * 100)

if not MANIFEST.is_file():
    raise RuntimeError(
        f"Canonical manifest missing: {MANIFEST}"
    )

if not INDEX.is_file():
    raise RuntimeError(
        f"DICOM index missing: {INDEX}"
    )

manifest = pd.read_csv(
    MANIFEST
)

index = pd.read_csv(
    INDEX
)

print()
print("Manifest rows:", len(manifest))
print("DICOM index rows:", len(index))

print()
print("Manifest columns:")
print(list(manifest.columns))

print()
print("Index columns:")
print(list(index.columns))


# ============================================================
# NORMALIZE INDEX
# ============================================================

required_manifest = {
    "patient_id",
    "source_table",
    "image_file_path_metadata",
    "binary_label",
}

missing_manifest = (
    required_manifest
    -
    set(manifest.columns)
)

if missing_manifest:
    raise RuntimeError(
        "Manifest missing: "
        + str(sorted(missing_manifest))
    )

required_index = {
    "physical_path",
    "filename",
    "patient_id_dicom",
    "study_uid",
    "series_uid",
    "sop_uid",
}

missing_index = (
    required_index
    -
    set(index.columns)
)

if missing_index:
    raise RuntimeError(
        "DICOM index missing: "
        + str(sorted(missing_index))
    )


for c in [
    "patient_id",
    "source_table",
    "image_file_path_metadata",
]:
    manifest[c] = (
        manifest[c]
        .map(clean)
    )

for c in [
    "physical_path",
    "filename",
    "patient_id_dicom",
    "study_uid",
    "series_uid",
    "sop_uid",
]:
    index[c] = (
        index[c]
        .map(clean)
    )


# ============================================================
# INDEX STATISTICS
# ============================================================

print()
print("=" * 100)
print("DICOM INDEX STATISTICS")
print("=" * 100)

print(
    "Unique DICOM PatientIDs:",
    index[
        "patient_id_dicom"
    ].nunique(),
)

print(
    "Unique Study UIDs:",
    index[
        "study_uid"
    ].nunique(),
)

print(
    "Unique Series UIDs:",
    index[
        "series_uid"
    ].nunique(),
)

print(
    "Unique SOP UIDs:",
    index[
        "sop_uid"
    ].nunique(),
)

print()
print(
    "First 20 DICOM PatientIDs:"
)

for x in (
    index[
        "patient_id_dicom"
    ]
    .drop_duplicates()
    .head(20)
):
    print(
        "  ",
        repr(x)
    )


# ============================================================
# MANIFEST UID EXTRACTION
# ============================================================

expected_rows = []

for i, row in manifest.iterrows():

    path = clean(
        row[
            "image_file_path_metadata"
        ]
    )

    uids = uid_list(
        path
    )

    expected_rows.append({

        "manifest_row":
            int(i),

        "patient_id":
            clean(
                row[
                    "patient_id"
                ]
            ),

        "source_table":
            clean(
                row[
                    "source_table"
                ]
            ),

        "binary_label":
            clean(
                row[
                    "binary_label"
                ]
            ),

        "metadata_path":
            path,

        "manifest_uid_1":
            uids[-2]
            if len(uids) >= 2
            else "",

        "manifest_uid_2":
            uids[-1]
            if len(uids) >= 2
            else "",

        "patient_tokens":
            "|".join(
                patient_tokens(
                    path
                )
            ),
    })

expected = pd.DataFrame(
    expected_rows
)


# ============================================================
# MATCH STRATEGY
# ============================================================

# Build direct indexes for fast matching.

index_by_study_series = defaultdict(list)
index_by_series = defaultdict(list)
index_by_filename = defaultdict(list)
index_by_patient = defaultdict(list)

for i, row in index.iterrows():

    item = row.to_dict()

    study = norm(
        row[
            "study_uid"
        ]
    )

    series = norm(
        row[
            "series_uid"
        ]
    )

    filename = norm(
        row[
            "filename"
        ]
    )

    patient = norm(
        row[
            "patient_id_dicom"
        ]
    )

    index_by_study_series[
        (
            study,
            series,
        )
    ].append(
        item
    )

    index_by_series[
        series
    ].append(
        item
    )

    index_by_filename[
        filename
    ].append(
        item
    )

    index_by_patient[
        patient
    ].append(
        item
    )


# ============================================================
# MATCH
# ============================================================

mapped = []
failures = []

method_counts = defaultdict(int)

for n, row in expected.iterrows():

    study = norm(
        row[
            "manifest_uid_1"
        ]
    )

    series = norm(
        row[
            "manifest_uid_2"
        ]
    )

    patient = norm(
        row[
            "patient_id"
        ]
    )

    path = clean(
        row[
            "metadata_path"
        ]
    )

    filename = (
        Path(path).name.lower()
        if path
        else ""
    )

    candidates = []

    method = ""

    # --------------------------------------------------------
    # METHOD A: study + series
    #
    # This is the most important test because the manifest
    # path contains explicit UID hierarchy.
    # --------------------------------------------------------

    if study and series:

        candidates = (
            index_by_study_series.get(
                (
                    study,
                    series,
                ),
                []
            )
        )

        if len(candidates) == 1:

            method = (
                "study_series_exact"
            )

    # --------------------------------------------------------
    # METHOD B: series only
    # --------------------------------------------------------

    if not method and series:

        candidates = (
            index_by_series.get(
                series,
                []
            )
        )

        if len(candidates) == 1:

            method = (
                "series_unique"
            )

    # --------------------------------------------------------
    # METHOD C: filename only
    # --------------------------------------------------------

    if not method and filename:

        candidates = (
            index_by_filename.get(
                filename,
                []
            )
        )

        if len(candidates) == 1:

            method = (
                "filename_unique"
            )

    # --------------------------------------------------------
    # METHOD D: patient + source-path token
    # diagnostic only
    # --------------------------------------------------------

    if not method:

        patient_candidates = (
            index_by_patient.get(
                patient,
                []
            )
        )

        if len(
            patient_candidates
        ) == 1:

            candidates = (
                patient_candidates
            )

            method = (
                "patient_unique"
            )

    # --------------------------------------------------------
    # Resolve only unique candidates
    # --------------------------------------------------------

    if method and len(candidates) == 1:

        item = candidates[0]

        result = {
            **row.to_dict(),

            "resolved_image_path":
                item[
                    "physical_path"
                ],

            "mapping_method":
                method,

            "dicom_patient_id":
                clean(
                    item[
                        "patient_id_dicom"
                    ]
                ),

            "dicom_study_uid":
                clean(
                    item[
                        "study_uid"
                    ]
                ),

            "dicom_series_uid":
                clean(
                    item[
                        "series_uid"
                    ]
                ),

            "dicom_sop_uid":
                clean(
                    item[
                        "sop_uid"
                    ]
                ),

            "dicom_filename":
                clean(
                    item[
                        "filename"
                    ]
                ),
        }

        mapped.append(
            result
        )

        method_counts[
            method
        ] += 1

    else:

        failures.append({

            **row.to_dict(),

            "failure_reason":
                (
                    "NO_MATCH"
                    if len(candidates) == 0
                    else
                    "AMBIGUOUS_MATCH"
                ),

            "candidate_count":
                len(candidates),

            "candidate_paths":
                "|".join(
                    clean(
                        x[
                            "physical_path"
                        ]
                    )
                    for x in candidates[
                        :20
                    ]
                ),
        })

    if (
        (n + 1) % 250
        == 0
    ):

        print(
            f"Processed {n + 1}/{len(expected)}...",
            flush=True
        )


mapped_df = pd.DataFrame(
    mapped
)

failure_df = pd.DataFrame(
    failures
)


# ============================================================
# VALIDATE MAPPINGS
# ============================================================

print()
print("=" * 100)
print("MAPPING SUMMARY")
print("=" * 100)

print(
    "Manifest records:",
    len(expected)
)

print(
    "Mapped:",
    len(mapped_df)
)

print(
    "Unmapped:",
    len(failure_df)
)

print()
print("Methods:")

for method, count in sorted(
    method_counts.items()
):

    print(
        "  ",
        method,
        ":",
        count
    )


# ============================================================
# CRITICAL UNIQUE FILE CHECK
# ============================================================

duplicate_physical_targets = {}

if len(mapped_df):

    counts = (
        mapped_df[
            "resolved_image_path"
        ]
        .value_counts()
    )

    duplicate_physical_targets = {
        str(path):
            int(count)
        for path, count
        in counts.items()
        if count > 1
    }


print()
print(
    "Physical DICOMs mapped to multiple manifest rows:",
    len(
        duplicate_physical_targets
    )
)


# ============================================================
# SOURCE-TABLE MATCH CHECK
# ============================================================

source_mismatches = 0

for _, row in mapped_df.iterrows():

    path = norm(
        row[
            "resolved_image_path"
        ]
    )

    source = norm(
        row[
            "source_table"
        ]
    )

    if source == "calc_train":
        expected_root = "/calc_train/"
    elif source == "calc_test":
        expected_root = "/calc_test/"
    elif source == "mass_train":
        expected_root = "/mass_train/"
    elif source == "mass_test":
        expected_root = "/mass_test/"
    else:
        expected_root = ""

    if expected_root and expected_root not in path.replace("\\", "/"):

        # Windows path normalization
        source_root_name = (
            source
        )

        if source_root_name not in path:

            source_mismatches += 1


# ============================================================
# SAVE
# ============================================================

mapped_df.to_csv(
    MAPPED,
    index=False,
)

failure_df.to_csv(
    FAILURES,
    index=False,
)

report = {

    "manifest":
        str(MANIFEST),

    "dicom_index":
        str(INDEX),

    "manifest_rows":
        int(len(expected)),

    "mapped_rows":
        int(len(mapped_df)),

    "unmapped_rows":
        int(len(failure_df)),

    "method_counts":
        dict(
            method_counts
        ),

    "duplicate_physical_targets":
        duplicate_physical_targets,

    "source_mismatches":
        int(source_mismatches),

    "status":
        (
            "MAPPING_PASS"
            if (
                len(mapped_df)
                == len(expected)
                and
                len(failure_df)
                == 0
                and
                len(
                    duplicate_physical_targets
                )
                == 0
            )
            else
            "MAPPING_INCOMPLETE"
        ),
}


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("CBIS MANIFEST ↔ DICOM INDEX RECONCILIATION COMPLETE")
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
    "Report:",
    REPORT
)

print()
print(
    "STATUS:",
    report[
        "status"
    ]
)

if report[
    "status"
] == "MAPPING_PASS":

    print()
    print(
        "ALL MANIFEST RECORDS HAVE A UNIQUE VERIFIED"
    )

    print(
        "LOCAL DICOM OBJECT."
    )

    print()
    print(
        "READY FOR STEP 34A-v2."
    )

else:

    print()
    print(
        "DO NOT TRAIN YET."
    )