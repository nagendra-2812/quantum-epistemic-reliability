from pathlib import Path
import pandas as pd
import json

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv"
)

print()
print("=" * 100)
print("CBIS PHYSICAL PATH DIAGNOSTIC")
print("=" * 100)

if not MANIFEST.is_file():
    raise RuntimeError(
        f"Manifest not found: {MANIFEST}"
    )

df = pd.read_csv(
    MANIFEST
)

print()
print("Manifest:")
print(MANIFEST)

print()
print("Rows:", len(df))

print()
print("Columns:")
for c in df.columns:
    print("  ", c)

print()
print("=" * 100)
print("FIRST THREE MANIFEST RECORDS")
print("=" * 100)

for i in range(min(3, len(df))):

    print()
    print("RECORD", i)

    row = df.iloc[i]

    for c in df.columns:

        value = row[c]

        if pd.isna(value):
            value = ""

        print(
            f"  {c}: {value}"
        )

print()
print("=" * 100)
print("PATH-LIKE COLUMN TEST")
print("=" * 100)

path_columns = [
    c
    for c in df.columns
    if any(
        token in str(c).lower()
        for token in [
            "path",
            "file",
            "dicom",
            "image",
            "physical",
            "crop",
            "roi",
        ]
    )
]

print()
print("Path-like columns:")

for c in path_columns:
    print("  ", c)

print()
print("=" * 100)
print("EXISTENCE COUNTS")
print("=" * 100)

results = {}

for c in path_columns:

    exists_count = 0
    nonempty_count = 0
    absolute_count = 0

    examples = []

    for value in df[c]:

        if pd.isna(value):
            value = ""

        value = str(value).strip()

        if not value:
            continue

        nonempty_count += 1

        p = Path(value)

        if p.is_absolute():
            absolute_count += 1

        candidates = [
            p,
            ROOT / p,
            ROOT / "CBIS-DDSM" / p,
            ROOT / "raw" / p,
            ROOT / "CBIS_DDSM" / p,
        ]

        found = None

        for candidate in candidates:

            if candidate.is_file():
                found = candidate
                break

        if found is not None:

            exists_count += 1

            if len(examples) < 5:
                examples.append(
                    {
                        "raw":
                            value,
                        "resolved":
                            str(found),
                    }
                )

    results[c] = {
        "nonempty":
            nonempty_count,

        "absolute":
            absolute_count,

        "existing":
            exists_count,

        "examples":
            examples,
    }

    print()
    print(
        c
    )

    print(
        "  nonempty:",
        nonempty_count,
    )

    print(
        "  absolute:",
        absolute_count,
    )

    print(
        "  existing:",
        exists_count,
    )

    if examples:

        print(
            "  examples:"
        )

        for x in examples:
            print(
                "    raw:",
                x["raw"],
            )

            print(
                "    resolved:",
                x["resolved"],
            )

print()
print("=" * 100)
print("SEARCH ACTUAL IMAGE FILES")
print("=" * 100)

extensions = {
    ".dcm",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

image_files = []

for p in ROOT.rglob("*"):

    if (
        p.is_file()
        and p.suffix.lower()
        in extensions
    ):

        image_files.append(
            p
        )

print()
print(
    "Candidate image files:",
    len(image_files),
)

for p in image_files[:30]:

    print(
        "  ",
        p
    )

print()
print("=" * 100)
print("CBIS DIRECTORY TOP LEVEL")
print("=" * 100)

for p in sorted(
    ROOT.iterdir()
):

    print(
        p.name,
        "DIR" if p.is_dir() else "FILE"
    )

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)

out = (
    ROOT
    / "experiments"
    / "step34a_manifest_reconciliation"
    / "CBIS_PHYSICAL_PATH_DIAGNOSTIC.json"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(
    "Report:",
    out
)

print()
print(
    "NO DATA MODIFIED."
)