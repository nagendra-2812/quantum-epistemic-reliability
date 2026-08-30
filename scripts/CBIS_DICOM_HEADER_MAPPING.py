from pathlib import Path
from collections import defaultdict
import json
import hashlib
import time

import pandas as pd
import pydicom


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
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
    / "CBIS_HEADER_BASED_PHYSICAL_MAPPING.csv"
)

FAILURES = (
    OUT
    / "CBIS_HEADER_BASED_MAPPING_FAILURES.csv"
)

REPORT = (
    OUT
    / "CBIS_HEADER_BASED_MAPPING_REPORT.json"
)

# Actual local roots found in your audit
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

def clean(value):

    if value is None:
        return ""

    return str(
        value
    ).strip()


def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(1024 * 1024),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def uid_from_ds(ds, name):

    value = getattr(
        ds,
        name,
        "",
    )

    return clean(
        value
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print("CBIS DICOM-HEADER PHYSICAL MAPPING")
print("=" * 100)

if not MANIFEST.is_file():

    raise RuntimeError(
        f"Manifest not found: {MANIFEST}"
    )

df = pd.read_csv(
    MANIFEST
)

print()
print(
    "Manifest:",
    MANIFEST
)

print(
    "Manifest rows:",
    len(df)
)

required = {
    "patient_id",
    "source_table",
    "image_file_path_metadata",
}

missing = (
    required
    -
    set(df.columns)
)

if missing:

    raise RuntimeError(
        "Missing manifest columns: "
        + str(
            sorted(missing)
        )
    )


# ============================================================
# EXTRACT EXPECTED STUDY / SERIES UIDS FROM MANIFEST PATH
# ============================================================

def manifest_uids(path_text):

    parts = [
        x.strip()
        for x in clean(
            path_text
        )
        .replace(
            "\\",
            "/",
        )
        .split("/")
        if x.strip()
    ]

    uid_parts = [
        x
        for x in parts
        if x.startswith(
            "1.3.6.1."
        )
    ]

    return uid_parts


expected_records = []

for idx, row in df.iterrows():

    metadata_path = clean(
        row[
            "image_file_path_metadata"
        ]
    )

    uids = manifest_uids(
        metadata_path
    )

    study_uid = (
        uids[-2]
        if len(uids) >= 2
        else ""
    )

    series_uid = (
        uids[-1]
        if len(uids) >= 2
        else ""
    )

    expected_records.append({
        "manifest_row":
            int(idx),

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

        "metadata_path":
            metadata_path,

        "expected_study_uid":
            study_uid,

        "expected_series_uid":
            series_uid,
    })

expected = pd.DataFrame(
    expected_records
)

print()
print(
    "Manifest records with extracted UID pairs:",
    int(
        (
            expected[
                "expected_study_uid"
            ].astype(str).str.len()
            > 0
        ).sum()
    )
)

# ============================================================
# INDEX LOCAL DICOM FILES BY HEADER
# ============================================================

print()
print("=" * 100)
print("INDEXING LOCAL DICOM HEADERS")
print("=" * 100)

# Key:
# (PatientID, StudyInstanceUID, SeriesInstanceUID)

header_index = defaultdict(
    list
)

read_failures = []

local_counts = {}

start = time.time()

for source_table, root in (
    SOURCE_ROOTS.items()
):

    if not root.is_dir():

        raise RuntimeError(
            f"Local source root missing: {root}"
        )

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        == ".dcm"
    )

    local_counts[
        source_table
    ] = len(files)

    print()
    print(
        source_table,
        "DICOM files:",
        len(files)
    )

    for n, path in enumerate(
        files,
        1,
    ):

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            patient = uid_from_ds(
                ds,
                "PatientID",
            )

            study = uid_from_ds(
                ds,
                "StudyInstanceUID",
            )

            series = uid_from_ds(
                ds,
                "SeriesInstanceUID",
            )

            sop = uid_from_ds(
                ds,
                "SOPInstanceUID",
            )

            modality = uid_from_ds(
                ds,
                "Modality",
            )

            view = uid_from_ds(
                ds,
                "ViewPosition",
            )

            rows = getattr(
                ds,
                "Rows",
                "",
            )

            columns = getattr(
                ds,
                "Columns",
                "",
            )

            key = (
                patient,
                study,
                series,
            )

            header_index[
                key
            ].append({

                "path":
                    str(path),

                "source_table":
                    source_table,

                "patient_id_dicom":
                    patient,

                "study_uid_dicom":
                    study,

                "series_uid_dicom":
                    series,

                "sop_uid_dicom":
                    sop,

                "modality":
                    modality,

                "view_position":
                    view,

                "rows":
                    clean(rows),

                "columns":
                    clean(columns),
            })

        except Exception as exc:

            read_failures.append({

                "path":
                    str(path),

                "source_table":
                    source_table,

                "error":
                    repr(exc),
            })

        if n % 500 == 0:

            print(
                f"  indexed {n}/{len(files)}...",
                flush=True,
            )


print()
print(
    "Unique DICOM header keys:",
    len(header_index)
)

print(
    "DICOM read failures:",
    len(read_failures)
)

# ============================================================
# MATCH MANIFEST → LOCAL DICOM
# ============================================================

print()
print("=" * 100)
print("HEADER-BASED MATCHING")
print("=" * 100)

mapped_rows = []
failure_rows = []

method_counter = defaultdict(
    int
)

for n, row in expected.iterrows():

    patient = clean(
        row[
            "patient_id"
        ]
    )

    source_table = clean(
        row[
            "source_table"
        ]
    )

    study = clean(
        row[
            "expected_study_uid"
        ]
    )

    series = clean(
        row[
            "expected_series_uid"
        ]
    )

    key = (
        patient,
        study,
        series,
    )

    candidates = (
        header_index.get(
            key,
            []
        )
    )

    # --------------------------------------------------------
    # Match 1: exact patient + study + series
    # --------------------------------------------------------

    if len(candidates) == 1:

        result = candidates[0]

        mapped_rows.append({

            **row.to_dict(),

            "resolved_image_path":
                result[
                    "path"
                ],

            "mapping_method":
                "patient_study_series_exact",

            "dicom_patient_id":
                result[
                    "patient_id_dicom"
                ],

            "dicom_study_uid":
                result[
                    "study_uid_dicom"
                ],

            "dicom_series_uid":
                result[
                    "series_uid_dicom"
                ],

            "dicom_sop_uid":
                result[
                    "sop_uid_dicom"
                ],

            "dicom_modality":
                result[
                    "modality"
                ],

            "dicom_view_position":
                result[
                    "view_position"
                ],

            "dicom_rows":
                result[
                    "rows"
                ],

            "dicom_columns":
                result[
                    "columns"
                ],

        })

        method_counter[
            "patient_study_series_exact"
        ] += 1

        continue

    # --------------------------------------------------------
    # Multiple local objects in same series
    #
    # The manifest row represents a full-mammogram object.
    # We need to identify the actual image object rather than
    # selecting arbitrarily.
    # --------------------------------------------------------

    if len(candidates) > 1:

        # Prefer a DICOM with a normal image payload.
        # Do not select randomly.

        image_candidates = []

        for c in candidates:

            try:

                ds = pydicom.dcmread(
                    c["path"],
                    stop_before_pixels=False,
                    force=True,
                )

                if hasattr(
                    ds,
                    "PixelData",
                ):

                    image_candidates.append(
                        c
                    )

            except Exception:
                pass

        if len(
            image_candidates
        ) == 1:

            result = image_candidates[0]

            mapped_rows.append({

                **row.to_dict(),

                "resolved_image_path":
                    result[
                        "path"
                    ],

                "mapping_method":
                    "patient_study_series_exact_pixeldata",

                "dicom_patient_id":
                    result[
                        "patient_id_dicom"
                    ],

                "dicom_study_uid":
                    result[
                        "study_uid_dicom"
                    ],

                "dicom_series_uid":
                    result[
                        "series_uid_dicom"
                    ],

                "dicom_sop_uid":
                    result[
                        "sop_uid_dicom"
                    ],

                "dicom_modality":
                    result[
                        "modality"
                    ],

                "dicom_view_position":
                    result[
                        "view_position"
                    ],

                "dicom_rows":
                    result[
                        "rows"
                    ],

                "dicom_columns":
                    result[
                        "columns"
                    ],

            })

            method_counter[
                "patient_study_series_exact_pixeldata"
            ] += 1

            continue

        # ----------------------------------------------------
        # Preserve ambiguity instead of guessing.
        # ----------------------------------------------------

        failure_rows.append({

            **row.to_dict(),

            "failure_reason":
                "AMBIGUOUS_MULTIPLE_LOCAL_DICOMS",

            "candidate_count":
                len(candidates),

            "candidate_paths":
                "|".join(
                    c["path"]
                    for c in candidates
                ),
        })

        continue

    # --------------------------------------------------------
    # No exact patient/study/series match
    #
    # Diagnostic fallback: study + series only
    # --------------------------------------------------------

    study_series_matches = []

    for (
        key2,
        values,
    ) in header_index.items():

        p2, s2, ser2 = key2

        if (
            s2 == study
            and
            ser2 == series
        ):

            for c in values:

                study_series_matches.append(
                    c
                )

    if len(
        study_series_matches
    ) == 1:

        result = (
            study_series_matches[0]
        )

        mapped_rows.append({

            **row.to_dict(),

            "resolved_image_path":
                result[
                    "path"
                ],

            "mapping_method":
                "study_series_unique",

            "dicom_patient_id":
                result[
                    "patient_id_dicom"
                ],

            "dicom_study_uid":
                result[
                    "study_uid_dicom"
                ],

            "dicom_series_uid":
                result[
                    "series_uid_dicom"
                ],

            "dicom_sop_uid":
                result[
                    "sop_uid_dicom"
                ],

            "dicom_modality":
                result[
                    "modality"
                ],

            "dicom_view_position":
                result[
                    "view_position"
                ],

            "dicom_rows":
                result[
                    "rows"
                ],

            "dicom_columns":
                result[
                    "columns"
                ],

        })

        method_counter[
            "study_series_unique"
        ] += 1

        continue

    failure_rows.append({

        **row.to_dict(),

        "failure_reason":
            (
                "NO_HEADER_MATCH"
                if len(
                    study_series_matches
                ) == 0
                else
                "AMBIGUOUS_STUDY_SERIES"
            ),

        "candidate_count":
            len(
                study_series_matches
            ),

        "candidate_paths":
            "|".join(
                c["path"]
                for c in study_series_matches[
                    :20
                ]
            ),
    })

    if (
        (n + 1) % 250
        == 0
    ):

        print(
            f"  matched {n + 1}/{len(expected)}...",
            flush=True,
        )


# ============================================================
# RESULTS
# ============================================================

mapped_df = pd.DataFrame(
    mapped_rows
)

failure_df = pd.DataFrame(
    failure_rows
)

resolved_count = len(
    mapped_df
)

failure_count = len(
    failure_df
)

print()
print("=" * 100)
print("FINAL HEADER-MAPPING RESULT")
print("=" * 100)

print(
    "Manifest records:",
    len(df)
)

print(
    "Resolved:",
    resolved_count
)

print(
    "Failures / ambiguities:",
    failure_count
)

print()
print(
    "Mapping methods:"
)

for method, count in sorted(
    method_counter.items()
):

    print(
        "  ",
        method,
        ":",
        count
    )


# ============================================================
# CRITICAL VALIDATION
# ============================================================

print()
print("=" * 100)
print("CRITICAL VALIDATION")
print("=" * 100)

bad_patient = 0
bad_study = 0
bad_series = 0
missing_files = 0

for _, row in mapped_df.iterrows():

    path = Path(
        str(
            row[
                "resolved_image_path"
            ]
        )
    )

    if not path.is_file():

        missing_files += 1
        continue

    if (
        clean(
            row[
                "dicom_patient_id"
            ]
        )
        !=
        clean(
            row[
                "patient_id"
            ]
        )
    ):

        bad_patient += 1

    if (
        clean(
            row[
                "dicom_study_uid"
            ]
        )
        !=
        clean(
            row[
                "expected_study_uid"
            ]
        )
    ):

        bad_study += 1

    if (
        clean(
            row[
                "dicom_series_uid"
            ]
        )
        !=
        clean(
            row[
                "expected_series_uid"
            ]
        )
    ):

        bad_series += 1


print(
    "Resolved paths missing on disk:",
    missing_files
)

print(
    "Patient mismatches:",
    bad_patient
)

print(
    "Study mismatches:",
    bad_study
)

print(
    "Series mismatches:",
    bad_series
)

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


REPORT_DATA = {

    "manifest":
        str(MANIFEST),

    "manifest_rows":
        int(len(df)),

    "local_dicom_counts":
        local_counts,

    "local_unique_header_keys":
        int(len(header_index)),

    "dicom_read_failures":
        int(len(read_failures)),

    "resolved":
        int(resolved_count),

    "failures":
        int(failure_count),

    "mapping_methods":
        dict(
            method_counter
        ),

    "validation":
        {
            "missing_resolved_files":
                int(missing_files),

            "patient_mismatches":
                int(bad_patient),

            "study_mismatches":
                int(bad_study),

            "series_mismatches":
                int(bad_series),
        },

    "status":
        (
            "FULL_MAPPING_PASS"
            if (
                resolved_count == len(df)
                and
                failure_count == 0
                and
                missing_files == 0
                and
                bad_patient == 0
                and
                bad_study == 0
                and
                bad_series == 0
            )
            else
            "MAPPING_REQUIRES_FURTHER_RECONCILIATION"
        ),

    "runtime_seconds":
        time.time() - start,
}

REPORT.write_text(
    json.dumps(
        REPORT_DATA,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SAVE HEADER READ FAILURES
# ============================================================

if read_failures:

    pd.DataFrame(
        read_failures
    ).to_csv(
        OUT
        / "LOCAL_DICOM_HEADER_READ_FAILURES.csv",
        index=False,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("CBIS DICOM-HEADER MAPPING COMPLETE")
print("=" * 100)

print()
print(
    "Mapped file:",
    MAPPED
)

print(
    "Failure file:",
    FAILURES
)

print(
    "Report:",
    REPORT
)

print()
print(
    "STATUS:",
    REPORT_DATA[
        "status"
    ]
)

if REPORT_DATA[
    "status"
] == "FULL_MAPPING_PASS":

    print()
    print(
        "ALL 3568 MANIFEST RECORDS HAVE VERIFIED"
    )

    print(
        "PATIENT/STUDY/SERIES-LEVEL PHYSICAL DICOM MAPPINGS."
    )

    print()
    print(
        "READY FOR STEP 34A-v2 TRAINING."
    )

else:

    print()
    print(
        "TRAINING MUST NOT START YET."
    )
