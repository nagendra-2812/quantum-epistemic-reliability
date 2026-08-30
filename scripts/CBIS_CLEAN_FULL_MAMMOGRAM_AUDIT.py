from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import pydicom


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

RECON = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
)

AVAILABLE = (
    RECON
    / "CBIS_PUBLICATION_AVAILABLE_RECORDS.csv"
)

MAPPING = (
    RECON
    / "CBIS_FINAL_ROBUST_PHYSICAL_MAPPING.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "STEP34A_V2_ASUS_CBIS_PUBLICATION_FREEZE"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL = (
    OUT
    / "CBIS_V2_FINAL_PHYSICAL_INPUT_MANIFEST.csv"
)

RECORD_FAILURES = (
    OUT
    / "CBIS_V2_RECORD_SELECTION_FAILURES.csv"
)

CANDIDATE_READ_FAILURES = (
    OUT
    / "CBIS_V2_CANDIDATE_DICOM_READ_FAILURES.csv"
)

REPORT = (
    OUT
    / "CBIS_V2_CLEAN_INPUT_SELECTION_AUDIT.json"
)


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def record_key(row):
    return (
        clean(row.get("patient_id", "")),
        clean(row.get("source_table", "")),
        clean(row.get("abnormality_id", "")),
        clean(row.get("image_view", "")),
        clean(row.get("laterality", "")),
    )


def inspect_dicom(path):

    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=False,
        force=True,
    )

    if not hasattr(
        ds,
        "PixelData",
    ):
        raise RuntimeError(
            "No PixelData"
        )

    pixels = ds.pixel_array

    if pixels.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D pixels, got ndim={pixels.ndim}"
        )

    rows = int(
        getattr(ds, "Rows", 0) or 0
    )

    columns = int(
        getattr(ds, "Columns", 0) or 0
    )

    if rows <= 0 or columns <= 0:
        raise RuntimeError(
            "Invalid Rows/Columns"
        )

    modality = clean(
        getattr(
            ds,
            "Modality",
            "",
        )
    )

    view = clean(
        getattr(
            ds,
            "ViewPosition",
            "",
        )
    )

    series_description = clean(
        getattr(
            ds,
            "SeriesDescription",
            "",
        )
    )

    image_type = clean(
        getattr(
            ds,
            "ImageType",
            "",
        )
    )

    desc = (
        series_description
        + " "
        + image_type
    ).lower()

    score = float(
        np.log1p(
            rows * columns
        )
    )

    # Positive evidence for a full mammogram.
    if "full" in desc:
        score += 20.0

    if "mammo" in desc:
        score += 10.0

    # Negative evidence for derived/ROI images.
    if "roi" in desc:
        score -= 30.0

    if "crop" in desc:
        score -= 30.0

    if "thumbnail" in desc:
        score -= 20.0

    if "mask" in desc:
        score -= 40.0

    return {
        "rows":
            rows,

        "columns":
            columns,

        "pixel_count":
            rows * columns,

        "modality":
            modality,

        "view_position":
            view,

        "series_description":
            series_description,

        "image_type":
            image_type,

        "score":
            score,
    }


print()
print("=" * 100)
print(
    "STEP 34A-v2 — CLEAN FULL-MAMMOGRAM SELECTION AUDIT"
)
print("=" * 100)

for path in [
    AVAILABLE,
    MAPPING,
]:

    if not path.is_file():
        raise RuntimeError(
            f"Required file not found: {path}"
        )

available = pd.read_csv(
    AVAILABLE
)

mapping = pd.read_csv(
    MAPPING
)

print()
print(
    "Available canonical records:",
    len(available),
)

# ------------------------------------------------------------
# Build mapping lookup
# ------------------------------------------------------------

mapping_lookup = {}

for _, row in mapping.iterrows():

    mapping_lookup[
        record_key(row)
    ] = row


selected = []
record_failures = []
candidate_failures = []

stats = Counter()

for n, (_, row) in enumerate(
    available.iterrows(),
    1,
):

    key = record_key(
        row
    )

    if key not in mapping_lookup:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "NO_MAPPING_RECORD",

        })

        stats[
            "no_mapping_record"
        ] += 1

        continue

    mapped = mapping_lookup[
        key
    ]

    raw_paths = clean(
        mapped.get(
            "physical_dicom_paths",
            "",
        )
    )

    paths = [
        Path(x)
        for x in raw_paths.split("|")
        if clean(x)
    ]

    existing_paths = [
        p
        for p in paths
        if p.is_file()
    ]

    if not existing_paths:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "NO_EXISTING_PHYSICAL_DICOM",

        })

        stats[
            "no_existing_dicom"
        ] += 1

        continue

    valid_candidates = []

    candidate_error_count = 0

    for p in existing_paths:

        try:

            info = inspect_dicom(
                p
            )

            valid_candidates.append({
                "path":
                    str(p),

                **info,
            })

        except Exception as exc:

            candidate_error_count += 1

            candidate_failures.append({

                **row.to_dict(),

                "physical_path":
                    str(p),

                "error":
                    repr(exc),

            })

    if candidate_error_count > 0:
        stats[
            "records_with_candidate_read_errors"
        ] += 1

    if not valid_candidates:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "NO_READABLE_PIXEL_DICOM",

            "candidate_count":
                len(existing_paths),

            "candidate_read_failures":
                candidate_error_count,

        })

        stats[
            "no_readable_candidate"
        ] += 1

        continue

    # --------------------------------------------------------
    # Choose highest-scoring candidate.
    # --------------------------------------------------------

    best_score = max(
        x[
            "score"
        ]
        for x in valid_candidates
    )

    best = [
        x
        for x in valid_candidates
        if (
            x[
                "score"
            ]
            == best_score
        )
    ]

    # If scores tie, use largest image.
    if len(best) > 1:

        max_pixels = max(
            x[
                "pixel_count"
            ]
            for x in best
        )

        best = [
            x
            for x in best
            if (
                x[
                    "pixel_count"
                ]
                == max_pixels
            )
        ]

    # Still ambiguous.
    if len(best) != 1:

        record_failures.append({

            **row.to_dict(),

            "failure_reason":
                "AMBIGUOUS_FULL_MAMMOGRAM_SELECTION",

            "candidate_count":
                len(valid_candidates),

            "candidate_paths":
                "|".join(
                    x["path"]
                    for x
                    in valid_candidates
                ),

            "candidate_dimensions":
                "|".join(
                    f"{x['rows']}x{x['columns']}"
                    for x
                    in valid_candidates
                ),

        })

        stats[
            "ambiguous_selection"
        ] += 1

        continue

    chosen = best[0]

    output = row.to_dict()

    output.update({

        "resolved_full_mammogram_dicom":
            chosen[
                "path"
            ],

        "resolved_rows":
            chosen[
                "rows"
            ],

        "resolved_columns":
            chosen[
                "columns"
            ],

        "resolved_pixel_count":
            chosen[
                "pixel_count"
            ],

        "resolved_modality":
            chosen[
                "modality"
            ],

        "resolved_view_position":
            chosen[
                "view_position"
            ],

        "resolved_series_description":
            chosen[
                "series_description"
            ],

        "resolved_image_type":
            chosen[
                "image_type"
            ],

        "resolved_candidate_count":
            len(valid_candidates),

        "candidate_read_failure_count":
            candidate_error_count,

        "selection_rule":
            (
                "highest reproducible full-image "
                "score; pixel-count tie-break"
            ),

        "selection_status":
            "VERIFIED",

    })

    selected.append(
        output
    )

    stats[
        "selected"
    ] += 1

    if (
        (n + 1) % 250
        == 0
    ):

        print(
            f"Processed {n + 1}/{len(available)}...",
            flush=True,
        )


selected_df = pd.DataFrame(
    selected
)

record_failure_df = pd.DataFrame(
    record_failures
)

candidate_failure_df = pd.DataFrame(
    candidate_failures
)


# ------------------------------------------------------------
# Split summaries
# ------------------------------------------------------------

print()
print("=" * 100)
print(
    "CLEAN SELECTION SUMMARY"
)
print("=" * 100)

print(
    "Available canonical records:",
    len(available)
)

print(
    "Successfully selected:",
    len(selected_df)
)

print(
    "Canonical records with selection failure:",
    len(record_failure_df)
)

print(
    "Individual candidate DICOM read failures:",
    len(candidate_failure_df)
)

print()
print(
    "Important: candidate DICOM read failures are NOT"
)

print(
    "counted as canonical-record failures when another"
)

print(
    "valid candidate exists."
)

# ------------------------------------------------------------
# Per split
# ------------------------------------------------------------

if len(selected_df):

    print()
    print(
        "SELECTED RECORDS BY SPLIT:"
    )

    print(
        selected_df[
            "experimental_split"
        ]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "SELECTED PATIENTS BY SPLIT:"
    )

    print(
        selected_df.groupby(
            "experimental_split"
        )[
            "patient_id"
        ]
        .nunique()
        .to_dict()
    )

if len(record_failure_df):

    print()
    print(
        "RECORD-LEVEL FAILURE REASONS:"
    )

    print(
        record_failure_df[
            "failure_reason"
        ]
        .value_counts()
        .to_dict()
    )


# ------------------------------------------------------------
# Patient disjointness
# ------------------------------------------------------------

def patients_for(
    frame,
    split,
):

    if len(frame) == 0:
        return set()

    return set(
        frame[
            frame[
                "experimental_split"
            ]
            == split
        ][
            "patient_id"
        ]
        .astype(str)
    )


train_patients = patients_for(
    selected_df,
    "train",
)

cal_patients = patients_for(
    selected_df,
    "calibration",
)

test_patients = patients_for(
    selected_df,
    "internal_test",
)

overlap = {

    "train_calibration":
        len(
            train_patients
            &
            cal_patients
        ),

    "train_test":
        len(
            train_patients
            &
            test_patients
        ),

    "calibration_test":
        len(
            cal_patients
            &
            test_patients
        ),
}

print()
print(
    "SELECTED-COHORT PATIENT OVERLAP:"
)

print(
    overlap
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

selected_df.to_csv(
    FINAL,
    index=False,
)

record_failure_df.to_csv(
    RECORD_FAILURES,
    index=False,
)

candidate_failure_df.to_csv(
    CANDIDATE_READ_FAILURES,
    index=False,
)

report = {

    "available_canonical_records":
        int(len(available)),

    "selected_records":
        int(len(selected_df)),

    "record_level_failures":
        int(len(record_failure_df)),

    "individual_candidate_read_failures":
        int(len(candidate_failure_df)),

    "record_failure_reasons":
        (
            record_failure_df[
                "failure_reason"
            ]
            .value_counts()
            .to_dict()
            if len(record_failure_df)
            else {}
        ),

    "selected_split_counts":
        (
            selected_df[
                "experimental_split"
            ]
            .value_counts()
            .to_dict()
            if len(selected_df)
            else {}
        ),

    "selected_patient_counts":
        (
            selected_df.groupby(
                "experimental_split"
            )[
                "patient_id"
            ]
            .nunique()
            .to_dict()
            if len(selected_df)
            else {}
        ),

    "patient_overlap":
        overlap,

    "status":
        (
            "CLEAN_FULL_MAMMOGRAM_SELECTION_PASS"
            if (
                len(selected_df)
                == len(available)
                and
                len(record_failure_df)
                == 0
                and
                all(
                    x == 0
                    for x
                    in overlap.values()
                )
            )
            else
            "CLEAN_SELECTION_REQUIRES_REVIEW"
        ),

    "final_manifest":
        str(FINAL),

    "record_failures":
        str(RECORD_FAILURES),

    "candidate_read_failures":
        str(CANDIDATE_READ_FAILURES),
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
    "CLEAN FULL-MAMMOGRAM AUDIT COMPLETE"
)
print("=" * 100)

print()
print(
    "Final manifest:",
    FINAL
)

print(
    "Record failures:",
    RECORD_FAILURES
)

print(
    "Candidate read failures:",
    CANDIDATE_READ_FAILURES
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

print()
print(
    "NO TRAINING PERFORMED."
)

print(
    "NO MEDICAL IMAGE FILES MODIFIED."
)