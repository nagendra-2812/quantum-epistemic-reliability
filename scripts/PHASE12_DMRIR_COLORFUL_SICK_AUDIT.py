from pathlib import Path
from collections import Counter
import json
import pandas as pd
from PIL import Image
import numpy as np

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    ROOT
    / "experiments"
    / "phase12_thermography_dmr_ir"
    / "THERMOGRAPHY_DMRIR_MASTER_MANIFEST.csv"
)

OUT = (
    ROOT
    / "experiments"
    / "phase12_thermography_dmr_ir"
)

REPORT = (
    OUT
    / "DMRIR_COLORFUL_SICK_AUDIT.json"
)

COLORFUL_CSV = (
    OUT
    / "DMRIR_COLORFUL_SICK_IMAGES.csv"
)

if not MANIFEST.is_file():
    raise RuntimeError(
        f"Master manifest not found: {MANIFEST}"
    )

df = pd.read_csv(
    MANIFEST
)

# ------------------------------------------------------------
# Select DMR-IR Sick
# ------------------------------------------------------------

sick = df[
    (df["source"] == "DMR_IR")
    &
    (df["label_name"] == "Sick")
].copy()

print()
print("=" * 80)
print("DMR-IR SICK IMAGE COLOR AUDIT")
print("=" * 80)

print()
print(
    "Total Sick images:",
    len(sick),
)

# ------------------------------------------------------------
# Detect likely colour images
#
# We do NOT call them invalid.
#
# A simple conservative statistic:
# compare channel similarity.
# Grayscale images tend to have highly similar RGB channels.
# False-colour images generally have greater channel spread.
# ------------------------------------------------------------

records = []

for _, row in sick.iterrows():

    path = Path(
        row["image_path"]
    )

    try:

        with Image.open(path) as im:

            rgb = im.convert(
                "RGB"
            )

            arr = np.asarray(
                rgb,
                dtype=np.float32,
            )

    except Exception as exc:

        records.append({
            **row.to_dict(),
            "audit_status":
                "READ_ERROR",

            "channel_mean_spread":
                None,

            "channel_std_spread":
                None,

            "likely_colorful":
                None,

            "error":
                repr(exc),
        })

        continue

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    means = np.array([
        r.mean(),
        g.mean(),
        b.mean(),
    ])

    stds = np.array([
        r.std(),
        g.std(),
        b.std(),
    ])

    mean_spread = float(
        means.max()
        - means.min()
    )

    std_spread = float(
        stds.max()
        - stds.min()
    )

    # Conservative heuristic.
    #
    # This is ONLY for screening, not final labeling.
    likely_colorful = (
        mean_spread > 5.0
        or
        std_spread > 5.0
    )

    records.append({
        **row.to_dict(),

        "audit_status":
            "READ_OK",

        "channel_mean_spread":
            mean_spread,

        "channel_std_spread":
            std_spread,

        "likely_colorful":
            bool(likely_colorful),
    })

audit_df = pd.DataFrame(
    records
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

read_errors = int(
    (
        audit_df[
            "audit_status"
        ]
        == "READ_ERROR"
    ).sum()
)

likely_colorful = int(
    audit_df[
        "likely_colorful"
    ]
    .fillna(False)
    .sum()
)

likely_standard = (
    len(audit_df)
    - likely_colorful
    - read_errors
)

print()
print(
    "Readable Sick images:",
    len(audit_df) - read_errors,
)

print(
    "Read errors:",
    read_errors,
)

print(
    "Likely colour/false-colour:",
    likely_colorful,
)

print(
    "Likely standard:",
    likely_standard,
)

# ------------------------------------------------------------
# Export suspected colorful files
# ------------------------------------------------------------

colorful_df = audit_df[
    audit_df[
        "likely_colorful"
    ]
    == True
].copy()

colorful_df.to_csv(
    COLORFUL_CSV,
    index=False,
)

# ------------------------------------------------------------
# Distribution summary
# ------------------------------------------------------------

print()
print("=" * 80)
print("IMAGE DIMENSION COUNTS")
print("=" * 80)

print(
    audit_df[
        [
            "width",
            "height",
        ]
    ]
    .value_counts()
    .sort_index()
)

print()
print("=" * 80)
print("TOP COLOR-SPREAD VALUES")
print("=" * 80)

print(
    audit_df[
        [
            "image_path",
            "channel_mean_spread",
            "channel_std_spread",
            "likely_colorful",
        ]
    ]
    .sort_values(
        "channel_mean_spread",
        ascending=False,
    )
    .head(60)
    .to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Audit JSON
# ------------------------------------------------------------

result = {
    "dataset":
        "DMR-IR",

    "total_sick_images":
        int(len(sick)),

    "read_errors":
        read_errors,

    "heuristic_likely_colorful":
        likely_colorful,

    "heuristic_likely_standard":
        likely_standard,

    "important_note":
        (
            "The likely_colorful flag is a screening heuristic "
            "and must not be interpreted as evidence of corruption "
            "or invalidity."
        ),

    "primary_policy":
        (
            "Retain all valid Sick images in the primary analysis."
        ),

    "sensitivity_policy":
        (
            "Evaluate a secondary sensitivity analysis excluding "
            "the confirmed colorful subset only if its representation "
            "difference is independently verified."
        ),

    "colorful_csv":
        str(COLORFUL_CSV),

    "status":
        "COLOR_AUDIT_COMPLETE",
}

REPORT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 80)
print("COLOR AUDIT COMPLETE")
print("=" * 80)

print()
print(
    "Suspected colorful CSV:",
    COLORFUL_CSV,
)

print(
    "Audit:",
    REPORT,
)

print()
print(
    "PRIMARY DECISION:"
)

print(
    "KEEP ALL VALID SICK IMAGES."
)

print()
print(
    "STATUS: DMRIR_COLOR_AUDIT_COMPLETE"
)