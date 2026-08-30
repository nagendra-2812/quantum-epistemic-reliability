from pathlib import Path
import json
import re
import hashlib

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

FAIL = (
    OUT
    / "CBIS_V2_FULL_MAMMOGRAM_SELECTION_FAILURES.csv"
)

REPORT = (
    OUT
    / "CBIS_V2_FULL_MAMMOGRAM_FREEZE_AUDIT.json"
)


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def read_dicom_info(path):

    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=False,
        force=True,
    )

    rows = int(
        getattr(ds, "Rows", 0)
        or 0
    )

    cols = int(
        getattr(ds, "Columns", 0)
        or 0
    )

    modality = clean(
        getattr(ds, "Modality", "")
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

    photometric = clean(
        getattr(
            ds,
            "PhotometricInterpretation",
            "",
        )
    )

    view_position = clean(
        getattr(
            ds,
            "ViewPosition",
            "",
        )
    )

    patient_id = clean(
        getattr(
            ds,
            "PatientID",
            "",
        )
    )

    study_uid = clean(
        getattr(
            ds,
            "StudyInstanceUID",
            "",
        )
    )

    series_uid = clean(
        getattr(
            ds,
            "SeriesInstanceUID",
            "",
        )
    )

    sop_uid = clean(
        getattr(
            ds,
            "SOPInstanceUID",
            "",
        )
    )

    pixel_count = (
        rows * cols
    )

    # Full mammograms are expected to be substantially larger
    # than small crop/ROI objects.
    #
    # This is a ranking feature, not a silent assumption.
    #
    score = 0.0

    score += (
        np.log1p(
            pixel_count
        )
    )

    desc = (
        series_description
        + " "
        + image_type
    ).lower()

    if "full" in desc:
        score += 20.0

    if "mammo" in desc:
        score += 10.0

    if "roi" in desc:
        score -= 30.0

    if "crop" in desc:
        score -= 30.0

    return {
        "path":
            str(path),

        "rows":
            rows,

        "columns":
            cols,

        "pixel_count":
            pixel_count,

        "modality":
            modality,

        "series_description":
            series_description,

        "image_type":
            image_type,

        "photometric":
            photometric,

        "view_position":
            view_position,

        "patient_id":
            patient_id,

        "study_uid":
            study_uid,

        "series_uid":
            series_uid,

        "sop_uid":
            sop_uid,

        "selection_score":
            float(score),
    }


print()
print("=" * 100)
print("STEP 34A-v2 — FINAL FULL-MAMMOGRAM INPUT FREEZE")
print("=" * 100)

if not AVAILABLE.is_file():
    raise RuntimeError(
        f"Available manifest missing: {AVAILABLE}"
    )

if not MAPPING.is_file():
    raise RuntimeError(
        f"Physical mapping missing: {MAPPING}"
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

print(
    "Physical mapping rows:",
    len(mapping),
)


# ------------------------------------------------------------
# Stable record identity
# ------------------------------------------------------------

def key(row):

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


available_keys = {
    key(row)
    for _, row
    in available.iterrows()
}


mapping_lookup = {}

for _, row in mapping.iterrows():

    k = key(row)

    mapping_lookup[
        k
    ] = row


# ------------------------------------------------------------
# Select full-mammogram DICOM
# ------------------------------------------------------------

selected = []
failures = []

candidate_count_distribution = {}

for n, (_, row) in enumerate(
    available.iterrows(),
    1,
):

    k = key(row)

    if k not in mapping_lookup:

        failures.append({
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

            "abnormality_id":
                clean(
                    row[
                        "abnormality_id"
                    ]
                ),

            "image_view":
                clean(
                    row[
                        "image_view"
                    ]
                ),

            "failure":
                "MAPPING_RECORD_NOT_FOUND",
        })

        continue

    mapped = mapping_lookup[
        k
    ]

    raw_paths = clean(
        mapped.get(
            "physical_dicom_paths",
            ""
        )
    )

    paths = [
        Path(x)
        for x in raw_paths.split("|")
        if clean(x)
    ]

    valid_paths = [
        p
        for p in paths
        if p.is_file()
    ]

    if not valid_paths:

        failures.append({
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

            "abnormality_id":
                clean(
                    row[
                        "abnormality_id"
                    ]
                ),

            "image_view":
                clean(
                    row[
                        "image_view"
                    ]
                ),

            "failure":
                "NO_PHYSICAL_DICOM",
        })

        continue

    candidates = []

    for p in valid_paths:

        try:

            info = read_dicom_info(
                p
            )

            candidates.append(
                info
            )

        except Exception as exc:

            failures.append({
                "patient_id":
                    clean(
                        row[
                            "patient_id"
                        ]
                    ),

                "physical_path":
                    str(p),

                "failure":
                    "DICOM_READ_FAILURE",

                "error":
                    repr(exc),
            })

    if not candidates:

        continue

    candidate_count_distribution[
        str(len(candidates))
    ] = (
        candidate_count_distribution.get(
            str(len(candidates)),
            0
        )
        + 1
    )

    # --------------------------------------------------------
    # Prefer the largest valid image by pixel count.
    #
    # This avoids selecting a small ROI/crop object.
    #
    # When multiple candidates have exactly the same maximal
    # dimensions, preserve the ambiguity instead of guessing.
    # --------------------------------------------------------

    max_pixels = max(
        x[
            "pixel_count"
        ]
        for x in candidates
    )

    top = [
        x
        for x in candidates
        if x[
            "pixel_count"
        ]
        == max_pixels
    ]

    if len(top) != 1:

        # Try metadata score as secondary discriminator.
        max_score = max(
            x[
                "selection_score"
            ]
            for x in top
        )

        top_score = [
            x
            for x in top
            if x[
                "selection_score"
            ]
            == max_score
        ]

        if len(
            top_score
        ) != 1:

            failures.append({

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

                "abnormality_id":
                    clean(
                        row[
                            "abnormality_id"
                        ]
                    ),

                "image_view":
                    clean(
                        row[
                            "image_view"
                        ]
                    ),

                "failure":
                    "AMBIGUOUS_FULL_MAMMOGRAM_SELECTION",

                "candidate_paths":
                    "|".join(
                        x[
                            "path"
                        ]
                        for x
                        in candidates
                    ),

                "candidate_dimensions":
                    "|".join(
                        (
                            f"{x['rows']}x{x['columns']}"
                        )
                        for x
                        in candidates
                    ),
            })

            continue

        selected_info = top_score[0]

    else:

        selected_info = top[0]

    out = row.to_dict()

    out[
        "resolved_full_mammogram_dicom"
    ] = selected_info[
        "path"
    ]

    out[
        "resolved_rows"
    ] = selected_info[
        "rows"
    ]

    out[
        "resolved_columns"
    ] = selected_info[
        "columns"
    ]

    out[
        "resolved_pixel_count"
    ] = selected_info[
        "pixel_count"
    ]

    out[
        "resolved_modality"
    ] = selected_info[
        "modality"
    ]

    out[
        "resolved_series_description"
    ] = selected_info[
        "series_description"
    ]

    out[
        "resolved_image_type"
    ] = selected_info[
        "image_type"
    ]

    out[
        "resolved_photometric"
    ] = selected_info[
        "photometric"
    ]

    out[
        "resolved_view_position"
    ] = selected_info[
        "view_position"
    ]

    out[
        "resolved_patient_id_dicom"
    ] = selected_info[
        "patient_id"
    ]

    out[
        "resolved_study_uid"
    ] = selected_info[
        "study_uid"
    ]

    out[
        "resolved_series_uid"
    ] = selected_info[
        "series_uid"
    ]

    out[
        "resolved_sop_uid"
    ] = selected_info[
        "sop_uid"
    ]

    out[
        "resolved_candidate_count"
    ] = len(
        candidates
    )

    out[
        "full_mammogram_selection_rule"
    ] = (
        "largest_pixel_count_then_metadata_score"
    )

    out[
        "full_mammogram_input_status"
    ] = (
        "VERIFIED_LOCAL_DICOM"
    )

    selected.append(
        out
    )

    if (
        n % 250
        == 0
    ):

        print(
            f"Processed {n}/{len(available)}...",
            flush=True,
        )


final_df = pd.DataFrame(
    selected
)

failure_df = pd.DataFrame(
    failures
)


# ------------------------------------------------------------
# Patient-disjointness
# ------------------------------------------------------------

def patient_set(
    frame,
    split,
):

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


train_patients = patient_set(
    final_df,
    "train",
)

cal_patients = patient_set(
    final_df,
    "calibration",
)

test_patients = patient_set(
    final_df,
    "internal_test",
)

overlaps = {

    "train_calibration":
        sorted(
            train_patients
            &
            cal_patients
        ),

    "train_test":
        sorted(
            train_patients
            &
            test_patients
        ),

    "calibration_test":
        sorted(
            cal_patients
            &
            test_patients
        ),
}


# ------------------------------------------------------------
# Counts
# ------------------------------------------------------------

print()
print("=" * 100)
print("FULL-MAMMOGRAM SELECTION RESULT")
print("=" * 100)

print(
    "Available records:",
    len(available)
)

print(
    "Final selected:",
    len(final_df)
)

print(
    "Selection failures:",
    len(failure_df)
)

print()
print(
    "Candidate-count distribution:",
    candidate_count_distribution
)

print()
print(
    "Patient overlaps:"
)

print(
    "  train/calibration:",
    len(
        overlaps[
            "train_calibration"
        ]
    )
)

print(
    "  train/test:",
    len(
        overlaps[
            "train_test"
        ]
    )
)

print(
    "  calibration/test:",
    len(
        overlaps[
            "calibration_test"
        ]
    )
)


# ------------------------------------------------------------
# Split counts
# ------------------------------------------------------------

print()
print(
    "FINAL SPLIT COUNTS:"
)

if len(final_df):

    print(
        final_df[
            "experimental_split"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        "FINAL PATIENT COUNTS:"
    )

    print(
        final_df.groupby(
            "experimental_split"
        )[
            "patient_id"
        ]
        .nunique()
        .to_dict()
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

final_df.to_csv(
    FINAL,
    index=False,
)

failure_df.to_csv(
    FAIL,
    index=False,
)

status = (
    "FULL_MAMMOGRAM_FREEZE_PASS"
    if (
        len(failure_df) == 0
        and
        len(final_df) == len(available)
        and
        len(
            overlaps[
                "train_calibration"
            ]
        ) == 0
        and
        len(
            overlaps[
                "train_test"
            ]
        ) == 0
        and
        len(
            overlaps[
                "calibration_test"
            ]
        ) == 0
    )
    else
    "FULL_MAMMOGRAM_FREEZE_REQUIRES_REVIEW"
)

report = {

    "available_records":
        int(
            len(available)
        ),

    "selected_records":
        int(
            len(final_df)
        ),

    "selection_failures":
        int(
            len(failure_df)
        ),

    "candidate_count_distribution":
        candidate_count_distribution,

    "train_records":
        int(
            (
                final_df[
                    "experimental_split"
                ]
                == "train"
            ).sum()
        ),

    "calibration_records":
        int(
            (
                final_df[
                    "experimental_split"
                ]
                == "calibration"
            ).sum()
        ),

    "internal_test_records":
        int(
            (
                final_df[
                    "experimental_split"
                ]
                == "internal_test"
            ).sum()
        ),

    "train_patients":
        int(
            len(train_patients)
        ),

    "calibration_patients":
        int(
            len(cal_patients)
        ),

    "internal_test_patients":
        int(
            len(test_patients)
        ),

    "patient_overlap":
        {
            k:
                int(len(v))
            for k, v
            in overlaps.items()
        },

    "selection_rule":
        (
            "largest pixel-count valid DICOM "
            "within the matched physical case, "
            "then metadata score as tie-breaker"
        ),

    "status":
        status,

    "final_manifest":
        str(FINAL),

    "failures":
        str(FAIL),
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
print("STEP 34A-v2 FULL-MAMMOGRAM FREEZE COMPLETE")
print("=" * 100)

print()
print(
    "Final manifest:",
    FINAL
)

print(
    "Failure manifest:",
    FAIL
)

print(
    "Audit:",
    REPORT
)

print()
print(
    "STATUS:",
    status
)

if status == "FULL_MAMMOGRAM_FREEZE_PASS":

    print()
    print(
        "READY FOR ACTUAL STEP 34A-v2 RESNET-50 TRAINING."
    )

else:

    print()
    print(
        "DO NOT TRAIN YET."
    )