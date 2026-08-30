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

OUT = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MAPPED_CSV = (
    OUT
    / "CBIS_EXACT_PHYSICAL_MAPPING.csv"
)

REPORT = (
    OUT
    / "CBIS_EXACT_PHYSICAL_MAPPING_REPORT.json"
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
# Helpers
# ============================================================

def normalize_path_text(value):

    return (
        str(value)
        .replace("\\", "/")
        .strip()
        .lower()
    )


def path_tokens(value):

    value = normalize_path_text(
        value
    )

    parts = [
        x
        for x in value.split("/")
        if x
    ]

    return parts


def uid_tokens(value):

    tokens = path_tokens(
        value
    )

    # UID-like path components
    return [
        token
        for token in tokens
        if token.startswith(
            "1.3.6.1."
        )
    ]


def leaf_name(value):

    parts = path_tokens(
        value
    )

    return (
        parts[-1]
        if parts
        else ""
    )


def remove_prefix_case(value):

    s = normalize_path_text(
        value
    )

    # Remove common dataset-folder prefixes
    prefixes = [
        "calc-training_",
        "calc-test_",
        "mass-training_",
        "mass-test_",
        "calc_training_",
        "calc_test_",
        "mass_training_",
        "mass_test_",
    ]

    for prefix in prefixes:

        if s.startswith(prefix):

            s = s[
                len(prefix):
            ]

    return s


# ============================================================
# Start
# ============================================================

print()
print("=" * 100)
print(
    "CBIS EXACT PHYSICAL PATH MAPPER"
)
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


# ============================================================
# Validate required fields
# ============================================================

required = {
    "source_table",
    "patient_id",
    "image_file_path_metadata",
}

missing = (
    required
    -
    set(df.columns)
)

if missing:

    raise RuntimeError(
        "Missing required columns: "
        + str(
            sorted(missing)
        )
    )


# ============================================================
# Index ACTUAL local DICOM files
# ============================================================

print()
print("=" * 100)
print(
    "INDEXING ACTUAL LOCAL DICOM FILES"
)
print("=" * 100)

local_index = {}

for source_table, root in SOURCE_ROOTS.items():

    if not root.is_dir():

        raise RuntimeError(
            f"Source root missing: {root}"
        )

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        == ".dcm"
    )

    print()
    print(
        source_table,
        "->",
        root
    )

    print(
        "DICOM files:",
        len(files)
    )

    source_records = []

    for p in files:

        rel = p.relative_to(
            root
        )

        rel_text = (
            str(rel)
            .replace("\\", "/")
            .lower()
        )

        tokens = path_tokens(
            rel_text
        )

        uids = uid_tokens(
            rel_text
        )

        source_records.append({
            "path":
                p,

            "relative":
                rel_text,

            "filename":
                p.name.lower(),

            "parts":
                tokens,

            "uids":
                uids,

            "stem":
                p.stem.lower(),
        })

    local_index[
        source_table
    ] = source_records


# ============================================================
# Diagnostic actual file examples
# ============================================================

print()
print("=" * 100)
print(
    "ACTUAL LOCAL CBIS PATH EXAMPLES"
)
print("=" * 100)

for source_table, records in (
    local_index.items()
):

    print()
    print(
        source_table
    )

    for record in records[:5]:

        print(
            "  ",
            record[
                "relative"
            ]
        )


# ============================================================
# Map each manifest row
# ============================================================

print()
print("=" * 100)
print(
    "MAPPING MANIFEST TO LOCAL FILES"
)
print("=" * 100)

mapped = []

method_counts = defaultdict(
    int
)

failures = []

for index, row in df.iterrows():

    source_table = str(
        row[
            "source_table"
        ]
    ).strip()

    metadata_path = str(
        row[
            "image_file_path_metadata"
        ]
    ).strip()

    records = local_index.get(
        source_table,
        []
    )

    metadata_norm = normalize_path_text(
        metadata_path
    )

    metadata_parts = path_tokens(
        metadata_path
    )

    metadata_uids = uid_tokens(
        metadata_path
    )

    metadata_leaf = (
        leaf_name(
            metadata_path
        )
    )

    metadata_patient = str(
        row[
            "patient_id"
        ]
    ).strip().lower()

    candidates = []

    # --------------------------------------------------------
    # METHOD 1:
    # exact relative path after source root
    # --------------------------------------------------------

    for record in records:

        if (
            record[
                "relative"
            ]
            ==
            metadata_norm
        ):

            candidates.append(
                (
                    "exact_relative_path",
                    record[
                        "path"
                    ],
                )
            )

    # --------------------------------------------------------
    # METHOD 2:
    # exact UID suffix
    #
    # Manifest structure:
    #   topfolder / UID1 / UID2 / file
    #
    # Local structure may have a different top folder,
    # so match the UID hierarchy + filename.
    # --------------------------------------------------------

    if not candidates:

        if (
            len(metadata_uids)
            >= 2
        ):

            target_uids = tuple(
                metadata_uids[-2:]
            )

            for record in records:

                if (
                    tuple(
                        record[
                            "uids"
                        ][-2:]
                    )
                    ==
                    target_uids
                    and
                    record[
                        "filename"
                    ]
                    ==
                    metadata_leaf
                ):

                    candidates.append(
                        (
                            "uid_suffix_plus_filename",
                            record[
                                "path"
                            ],
                        )
                    )

    # --------------------------------------------------------
    # METHOD 3:
    # UID1 + UID2 regardless of filename
    # --------------------------------------------------------

    if not candidates:

        if len(metadata_uids) >= 2:

            target_uids = tuple(
                metadata_uids[-2:]
            )

            uid_matches = []

            for record in records:

                if (
                    tuple(
                        record[
                            "uids"
                        ][-2:]
                    )
                    ==
                    target_uids
                ):

                    uid_matches.append(
                        record
                    )

            if len(uid_matches) == 1:

                candidates.append(
                    (
                        "unique_uid_pair",
                        uid_matches[0][
                            "path"
                        ],
                    )
                )

    # --------------------------------------------------------
    # METHOD 4:
    # Patient token + UID suffix
    # --------------------------------------------------------

    if not candidates:

        if len(metadata_uids) >= 2:

            target_uids = tuple(
                metadata_uids[-2:]
            )

            patient_token = (
                metadata_patient
                .replace(
                    "_",
                    "-"
                )
            )

            for record in records:

                rel = record[
                    "relative"
                ]

                if (
                    patient_token
                    in rel
                    and
                    tuple(
                        record[
                            "uids"
                        ][-2:]
                    )
                    ==
                    target_uids
                ):

                    candidates.append(
                        (
                            "patient_plus_uid_pair",
                            record[
                                "path"
                            ],
                        )
                    )

    # --------------------------------------------------------
    # Resolve only if unique
    # --------------------------------------------------------

    unique_candidates = {}

    for method, path in candidates:

        unique_candidates[
            str(path)
        ] = method

    if len(unique_candidates) == 1:

        resolved_path = Path(
            next(
                iter(
                    unique_candidates
                )
            )
        )

        method = next(
            iter(
                unique_candidates.values()
            )
        )

        method_counts[
            method
        ] += 1

        row_dict = row.to_dict()

        row_dict[
            "resolved_image_path"
        ] = str(
            resolved_path
        )

        row_dict[
            "mapping_method"
        ] = method

        mapped.append(
            row_dict
        )

    else:

        row_dict = row.to_dict()

        row_dict[
            "resolved_image_path"
        ] = ""

        if len(
            unique_candidates
        ) == 0:

            row_dict[
                "mapping_method"
            ] = "FAILED"

        else:

            row_dict[
                "mapping_method"
            ] = (
                "AMBIGUOUS_"
                + str(
                    len(
                        unique_candidates
                    )
                )
            )

        mapped.append(
            row_dict
        )

        failures.append({
            "manifest_row":
                int(index),

            "source_table":
                source_table,

            "patient_id":
                str(
                    row[
                        "patient_id"
                    ]
                ),

            "metadata_path":
                metadata_path,

            "metadata_uids":
                metadata_uids,

            "candidate_count":
                len(
                    unique_candidates
                ),

            "candidates":
                list(
                    unique_candidates
                ),
        })

    if (
        (index + 1) % 250
        == 0
    ):

        print(
            f"Mapped {index + 1}/{len(df)}...",
            flush=True
        )


mapped_df = pd.DataFrame(
    mapped
)


# ============================================================
# Mapping result
# ============================================================

resolved_mask = (
    mapped_df[
        "resolved_image_path"
    ]
    .astype(str)
    .str.len()
    > 0
)

resolved_count = int(
    resolved_mask.sum()
)

unresolved_count = (
    len(mapped_df)
    -
    resolved_count
)

print()
print("=" * 100)
print(
    "MAPPING RESULT"
)
print("=" * 100)

print(
    "Manifest records:",
    len(mapped_df)
)

print(
    "Resolved:",
    resolved_count
)

print(
    "Unresolved:",
    unresolved_count
)

print()
print(
    "Methods:"
)

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
# Validate resolved paths
# ============================================================

bad_existing = []

for p in mapped_df[
    "resolved_image_path"
]:

    p = str(p).strip()

    if not p:
        continue

    if not Path(
        p
    ).is_file():

        bad_existing.append(
            p
        )

print()
print(
    "Resolved paths that do not exist:",
    len(bad_existing)
)

if bad_existing:

    raise RuntimeError(
        "Mapper generated invalid paths."
    )


# ============================================================
# Save mapping
# ============================================================

mapped_df.to_csv(
    MAPPED_CSV,
    index=False,
)


# ============================================================
# Save failures
# ============================================================

FAIL_CSV = (
    OUT
    / "CBIS_EXACT_PHYSICAL_MAPPING_FAILURES.csv"
)

pd.DataFrame(
    failures
).to_csv(
    FAIL_CSV,
    index=False,
)


# ============================================================
# Final report
# ============================================================

result = {

    "manifest":
        str(MANIFEST),

    "manifest_rows":
        int(len(df)),

    "source_roots":
        {
            k:
                str(v)
            for k, v
            in SOURCE_ROOTS.items()
        },

    "local_dicom_counts":
        {
            k:
                len(v)
            for k, v
            in local_index.items()
        },

    "resolved":
        resolved_count,

    "unresolved":
        unresolved_count,

    "mapping_methods":
        dict(
            method_counts
        ),

    "failure_count":
        len(failures),

    "status":
        (
            "FULL_MAPPING_PASS"
            if (
                resolved_count
                == len(df)
                and
                not bad_existing
            )
            else
            "MAPPING_INCOMPLETE"
        ),
}

REPORT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 100)
print(
    "CBIS EXACT PHYSICAL PATH MAPPING FINISHED"
)
print("=" * 100)

print()
print(
    "Mapped CSV:",
    MAPPED_CSV
)

print(
    "Failure CSV:",
    FAIL_CSV
)

print(
    "Report:",
    REPORT
)

print()
print(
    "STATUS:",
    result[
        "status"
    ]
)

if result[
    "status"
] != "FULL_MAPPING_PASS":

    print()
    print(
        "NO TRAINING STARTED."
    )

else:

    print()
    print(
        "ALL MANIFEST RECORDS RESOLVED."
    )

    print(
        "READY FOR PUBLICATION TRAINING."
    )