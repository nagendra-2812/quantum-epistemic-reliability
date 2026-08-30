from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST_DIR = (
    ROOT / "manifests"
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


EXPECTED_HASHES = {
    "351ce90b9d4528cd4672f65fcaf7720868d631b7715db2be5659f11ddee32997",
    "7759f60af11dfbc6fe3c24606459c835661599d21294b94d81aa536d7c420a71",
}

EXPECTED = {
    "frozen_records": 3568,
    "patient_safe_records": 3289,
    "train_records": 2790,
    "test_records": 499,
}


def sha256(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


print()
print("=" * 100)
print("STEP 34A MANIFEST RECONCILIATION")
print("=" * 100)

# ------------------------------------------------------------
# All candidate manifests
# ------------------------------------------------------------

candidates = sorted(
    p
    for p in MANIFEST_DIR.glob("*.csv")
    if p.is_file()
)

print()
print(
    "CSV manifest candidates:",
    len(candidates),
)

for path in candidates:

    try:

        df = pd.read_csv(
            path
        )

        digest = sha256(
            path
        )

        print()
        print(
            "FILE:",
            path.name,
        )

        print(
            "  SHA256:",
            digest,
        )

        print(
            "  EXACT HISTORICAL HASH:",
            digest in EXPECTED_HASHES,
        )

        print(
            "  ROWS:",
            len(df),
        )

        if "patient_id" in df.columns:

            print(
                "  UNIQUE PATIENTS:",
                df[
                    "patient_id"
                ].nunique(),
            )

        print(
            "  COLUMNS:",
            list(df.columns),
        )

    except Exception as exc:

        print()
        print(
            "FILE:",
            path
        )

        print(
            "  READ ERROR:",
            repr(exc)
        )


# ------------------------------------------------------------
# Search for saved train/test split CSVs
# ------------------------------------------------------------

print()
print("=" * 100)
print("SPLIT FILE DISCOVERY")
print("=" * 100)

split_candidates = sorted(
    p
    for p in MANIFEST_DIR.glob("*.csv")
    if any(
        token in p.name.lower()
        for token in [
            "patient_safe",
            "train_records",
            "test_records",
            "split",
            "cohort",
        ]
    )
)

for path in split_candidates:

    try:

        df = pd.read_csv(
            path
        )

        print()
        print(
            path.name
        )

        print(
            "  rows:",
            len(df),
        )

        print(
            "  columns:",
            list(df.columns),
        )

    except Exception as exc:

        print(
            path.name,
            "ERROR",
            repr(exc),
        )


# ------------------------------------------------------------
# Look specifically for exact historical sizes
# ------------------------------------------------------------

print()
print("=" * 100)
print("SIZE-BASED MATCH SEARCH")
print("=" * 100)

size_matches = []

for path in candidates:

    try:

        df = pd.read_csv(
            path
        )

    except Exception:
        continue

    n = len(df)

    if n in {
        3568,
        3289,
        2790,
        499,
    }:

        size_matches.append(
            (
                path,
                n,
            )
        )

for path, n in size_matches:

    print(
        path.name,
        "->",
        n,
        "rows",
    )


# ------------------------------------------------------------
# Search JSON for historical references
# ------------------------------------------------------------

print()
print("=" * 100)
print("HISTORICAL STEP 34A JSON REFERENCES")
print("=" * 100)

json_hits = []

for path in MANIFEST_DIR.rglob("*.json"):

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        continue

    if (
        "7759f60af11dfbc6fe3c24606459c835661599d21294b94d81aa536d7c420a71"
        in text
        or
        "351ce90b9d4528cd4672f65fcaf7720868d631b7715db2be5659f11ddee32997"
        in text
        or
        "STEP34A_TRAIN_FIX_CBIS_RESNET50_BASELINE"
        in text
    ):

        json_hits.append(
            path
        )

for path in sorted(
    set(json_hits)
):

    print(
        path
    )


# ------------------------------------------------------------
# Final recommendation
# ------------------------------------------------------------

exact_hash_files = []

for path in candidates:

    try:
        digest = sha256(path)

    except Exception:
        continue

    if digest in EXPECTED_HASHES:

        exact_hash_files.append(
            path
        )

print()
print("=" * 100)
print("FINAL RECONCILIATION")
print("=" * 100)

print()
print(
    "Exact historical manifest files found:",
    len(exact_hash_files),
)

for path in exact_hash_files:

    print(
        "  ",
        path
    )

status = (
    "EXACT_HISTORICAL_MANIFEST_FOUND"
    if exact_hash_files
    else
    "HISTORICAL_MANIFEST_NOT_PRESENT"
)

result = {
    "expected":
        EXPECTED,

    "expected_hashes":
        sorted(
            EXPECTED_HASHES
        ),

    "exact_hash_files":
        [
            str(p)
            for p in exact_hash_files
        ],

    "size_matches":
        [
            {
                "file":
                    str(path),

                "rows":
                    n,
            }
            for path, n
            in size_matches
        ],

    "json_historical_references":
        [
            str(p)
            for p in sorted(
                set(json_hits)
            )
        ],

    "status":
        status,
}

REPORT = (
    OUT
    / "STEP34A_MANIFEST_RECONCILIATION.json"
)

REPORT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(
    "Report:",
    REPORT,
)

print()
print(
    "STATUS:",
    status,
)

print()
print("NO TRAINING WAS PERFORMED.")
print("NO DATA WERE MODIFIED.")
print("NO MANIFEST WAS MODIFIED.")