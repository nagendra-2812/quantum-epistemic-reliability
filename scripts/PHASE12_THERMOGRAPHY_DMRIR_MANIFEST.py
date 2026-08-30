from pathlib import Path
from collections import Counter
import csv
import json
import re
import hashlib

from PIL import Image


ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

DATA_ROOT = (
    ROOT
    / "Breast Thermography"
)

OUT = (
    ROOT
    / "experiments"
    / "phase12_thermography_dmr_ir"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST = (
    OUT
    / "THERMOGRAPHY_DMRIR_MASTER_MANIFEST.csv"
)

AUDIT = (
    OUT
    / "THERMOGRAPHY_DMRIR_SOURCE_AUDIT.json"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# CONFIGURED SOURCE MAP
# ============================================================

SOURCE_MAP = {
    "Benign":
        {
            "source":
                "THERMOGRAPHY_BENIGN_MALIGNANT",

            "label":
                0,

            "label_name":
                "Benign",
        },

    "Malignant":
        {
            "source":
                "THERMOGRAPHY_BENIGN_MALIGNANT",

            "label":
                1,

            "label_name":
                "Malignant",
        },

    "Healthy":
        {
            "source":
                "DMR_IR",

            "label":
                0,

            "label_name":
                "Healthy",
        },

    "Sick":
        {
            "source":
                "DMR_IR",

            "label":
                1,

            "label_name":
                "Sick",
        },
}


# ============================================================
# BASIC CHECK
# ============================================================

if not DATA_ROOT.is_dir():

    raise RuntimeError(
        f"Thermography root not found: {DATA_ROOT}"
    )


print()
print("=" * 80)
print(
    "PHASE 12 — THERMOGRAPHY + DMR-IR SOURCE SEPARATION"
)
print("=" * 80)

print()
print(
    "Root:",
    DATA_ROOT,
)


# ============================================================
# DISCOVER FILES
# ============================================================

rows = []

for top_name, mapping in SOURCE_MAP.items():

    top = (
        DATA_ROOT
        / top_name
    )

    if not top.is_dir():

        raise RuntimeError(
            f"Expected directory missing: {top}"
        )

    files = sorted(
        p
        for p in top.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    print()
    print(
        top_name,
        "files:",
        len(files),
    )

    for path in files:

        relative_parts = path.relative_to(
            top
        ).parts

        # ----------------------------------------------------
        # Plausible subject/case extraction
        # ----------------------------------------------------

        candidates = []

        for part in relative_parts[:-1]:

            if re.search(
                r"(?i)"
                r"(IIR|patient|subject|case|person|"
                r"healthy|sick|benign|malignant)",
                part,
            ):

                candidates.append(
                    part
                )

        if candidates:

            subject_id = candidates[-1]

        elif len(relative_parts) >= 2:

            # Fallback:
            # immediate parent directory
            subject_id = relative_parts[-2]

        else:

            subject_id = path.stem

        # ----------------------------------------------------
        # Image properties
        # ----------------------------------------------------

        try:

            with Image.open(
                path
            ) as im:

                width, height = im.size
                image_mode = im.mode

        except Exception as exc:

            raise RuntimeError(
                f"Cannot read image: {path}\n{exc}"
            )

        # ----------------------------------------------------
        # SHA256
        # ----------------------------------------------------

        digest = hashlib.sha256()

        with path.open(
            "rb"
        ) as f:

            for block in iter(
                lambda:
                    f.read(
                        1024 * 1024
                    ),
                b"",
            ):

                digest.update(
                    block
                )

        rows.append({

            "image_path":
                str(path),

            "source":
                mapping[
                    "source"
                ],

            "top_level_class":
                top_name,

            "label":
                mapping[
                    "label"
                ],

            "label_name":
                mapping[
                    "label_name"
                ],

            "subject_id":
                subject_id,

            "relative_path":
                str(
                    path.relative_to(
                        DATA_ROOT
                    )
                ),

            "width":
                int(width),

            "height":
                int(height),

            "image_mode":
                image_mode,

            "bytes":
                int(
                    path.stat().st_size
                ),

            "sha256":
                digest.hexdigest(),
        })


# ============================================================
# DATAFRAME-LIKE AUDIT
# ============================================================

print()
print("=" * 80)
print("SOURCE SUMMARY")
print("=" * 80)

source_counts = Counter(
    row["source"]
    for row in rows
)

class_counts = Counter(
    row["top_level_class"]
    for row in rows
)

for source, count in sorted(
    source_counts.items()
):

    print(
        source,
        ":",
        count,
    )

print()

for cls, count in sorted(
    class_counts.items()
):

    print(
        cls,
        ":",
        count,
    )


# ============================================================
# SUBJECT COUNTS
# ============================================================

for source in sorted(
    source_counts
):

    source_rows = [
        r
        for r in rows
        if r["source"]
        == source
    ]

    subjects = {
        r["subject_id"]
        for r in source_rows
        if r["subject_id"]
    }

    print()
    print(
        source
    )

    print(
        "  images:",
        len(source_rows),
    )

    print(
        "  subjects:",
        len(subjects),
    )

    print(
        "  labels:",
        dict(
            Counter(
                r["label_name"]
                for r in source_rows
            )
        ),
    )


# ============================================================
# DUPLICATE AUDIT
# ============================================================

hash_groups = {}

for row in rows:

    hash_groups.setdefault(
        row["sha256"],
        []
    ).append(
        row["image_path"]
    )

duplicate_groups = {
    digest: paths
    for digest, paths
    in hash_groups.items()
    if len(paths) > 1
}

print()
print("=" * 80)
print("DUPLICATE CONTENT AUDIT")
print("=" * 80)

print(
    "Unique image hashes:",
    len(hash_groups),
)

print(
    "Duplicate hash groups:",
    len(duplicate_groups),
)

duplicate_images = sum(
    len(paths)
    for paths
    in duplicate_groups.values()
)

print(
    "Images participating in duplicate groups:",
    duplicate_images,
)


# ============================================================
# DIMENSION AUDIT
# ============================================================

dimension_counts = Counter(
    (
        r["width"],
        r["height"],
    )
    for r in rows
)

print()
print("=" * 80)
print("IMAGE DIMENSIONS")
print("=" * 80)

for dimension, count in (
    dimension_counts.most_common()
):

    print(
        f"{dimension[0]}x{dimension[1]}:",
        count,
    )


# ============================================================
# SOURCE CONTAMINATION CHECK
# ============================================================

subject_sources = {}

for row in rows:

    subject = row[
        "subject_id"
    ]

    if not subject:
        continue

    subject_sources.setdefault(
        subject,
        set()
    ).add(
        row["source"]
    )

cross_source_subjects = {
    subject: sorted(sources)
    for subject, sources
    in subject_sources.items()
    if len(sources) > 1
}

print()
print("=" * 80)
print("CROSS-SOURCE SUBJECT CHECK")
print("=" * 80)

print(
    "Subjects appearing in both source datasets:",
    len(cross_source_subjects),
)

if cross_source_subjects:

    print()
    print(
        "First 20:"
    )

    for subject, sources in list(
        cross_source_subjects.items()
    )[:20]:

        print(
            subject,
            "->",
            sources,
        )


# ============================================================
# WRITE MASTER MANIFEST
# ============================================================

fieldnames = [
    "image_path",
    "source",
    "top_level_class",
    "label",
    "label_name",
    "subject_id",
    "relative_path",
    "width",
    "height",
    "image_mode",
    "bytes",
    "sha256",
]

with MANIFEST.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# WRITE JSON AUDIT
# ============================================================

audit = {

    "root":
        str(DATA_ROOT),

    "total_images":
        len(rows),

    "source_counts":
        dict(source_counts),

    "top_level_class_counts":
        dict(class_counts),

    "subject_counts":
        {
            source:
                len({
                    r["subject_id"]
                    for r in rows
                    if r["source"]
                    == source
                    and r["subject_id"]
                })
            for source in source_counts
        },

    "dimension_counts":
        {
            f"{w}x{h}":
                count
            for (w, h), count
            in dimension_counts.items()
        },

    "duplicate_hash_groups":
        len(duplicate_groups),

    "duplicate_images":
        duplicate_images,

    "cross_source_subject_count":
        len(cross_source_subjects),

    "cross_source_subjects":
        cross_source_subjects,

    "status":
        "SOURCE_SEPARATION_COMPLETE",
}


AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("THERMOGRAPHY + DMR-IR MANIFEST COMPLETE")
print("=" * 80)

print()
print(
    "Total images:",
    len(rows),
)

print(
    "Thermography Benign/Malignant:",
    source_counts.get(
        "THERMOGRAPHY_BENIGN_MALIGNANT",
        0,
    ),
)

print(
    "DMR-IR Healthy/Sick:",
    source_counts.get(
        "DMR_IR",
        0,
    ),
)

print()
print(
    "Master manifest:",
    MANIFEST,
)

print(
    "Audit:",
    AUDIT,
)

print()
print(
    "STATUS: PHASE12_SOURCE_SEPARATION_COMPLETE"
)
