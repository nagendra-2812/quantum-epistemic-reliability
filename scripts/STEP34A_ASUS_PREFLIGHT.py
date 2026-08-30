from pathlib import Path
import hashlib
import json
import os
import sys

import pandas as pd
import torch


PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM"
)

MANIFEST_CANDIDATES = [
    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST_FINAL.csv",

    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST_PHYSICAL.csv",

    ROOT
    / "manifests"
    / "CBIS_DDSM_CANONICAL_MANIFEST.csv",

    ROOT
    / "manifests"
    / "CBIS_DDSM_MASTER_MANIFEST_FINAL.csv",
]

EXPECTED_STEP34A_HASHES = {
    # Hash recorded in the Step-34A training script
    "351ce90b9d4528cd4672f65fcaf7720868d631b7715db2be5659f11ddee32997",

    # Hash recorded in the later publication freeze
    "7759f60af11dfbc6fe3c24606459c835661599d21294b94d81aa536d7c420a71",
}

OUT = (
    ROOT
    / "experiments"
    / "step34a_asus_preflight"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path):

    h = hashlib.sha256()

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

            h.update(
                block
            )

    return h.hexdigest()


print()
print("=" * 100)
print("STEP 34A — ASUS PREFLIGHT")
print("=" * 100)

print()
print("Project:")
print(PROJECT)

print()
print("CBIS root:")
print(ROOT)

print()
print("=" * 100)
print("PYTHON / PYTORCH")
print("=" * 100)

print(
    "Python:",
    sys.version,
)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

if torch.cuda.is_available():

    print(
        "CUDA version:",
        torch.version.cuda,
    )

    print(
        "GPU count:",
        torch.cuda.device_count(),
    )

    for i in range(
        torch.cuda.device_count()
    ):

        print()
        print(
            "GPU",
            i,
            ":",
            torch.cuda.get_device_name(i),
        )

        props = (
            torch.cuda.get_device_properties(i)
        )

        print(
            "  VRAM GiB:",
            round(
                props.total_memory
                / (1024 ** 3),
                2,
            ),
        )


print()
print("=" * 100)
print("FROZEN MANIFEST SEARCH")
print("=" * 100)

found = []

for path in MANIFEST_CANDIDATES:

    if path.is_file():

        found.append(
            path
        )

        digest = sha256_file(
            path
        )

        print()
        print(
            "MANIFEST:",
            path,
        )

        print(
            "SHA256:",
            digest,
        )

        print(
            "MATCHES RECORDED STEP34A HASH:",
            digest
            in EXPECTED_STEP34A_HASHES,
        )

        try:

            df = pd.read_csv(
                path
            )

            print(
                "ROWS:",
                len(df),
            )

            print(
                "COLUMNS:",
                list(df.columns),
            )

            if "patient_id" in df.columns:

                print(
                    "UNIQUE PATIENTS:",
                    df[
                        "patient_id"
                    ]
                    .nunique(),
                )

        except Exception as exc:

            print(
                "CSV READ ERROR:",
                repr(exc),
            )


if not found:

    raise RuntimeError(
        "No candidate frozen CBIS manifest was found."
    )

print()
print(
    "Manifest candidates found:",
    len(found),
)

# ------------------------------------------------------------
# Publication-safe target
# ------------------------------------------------------------

TARGET_MANIFEST = None

for path in found:

    digest = sha256_file(
        path
    )

    if digest in EXPECTED_STEP34A_HASHES:

        TARGET_MANIFEST = path
        break


print()
print("=" * 100)
print("TARGET STEP 34A MANIFEST")
print("=" * 100)

if TARGET_MANIFEST is None:

    print(
        "NO EXACT HASH MATCH FOUND."
    )

    print()
    print(
        "This does NOT mean the data are wrong."
    )

    print(
        "It means we must reconcile the ASUS manifest"
    )

    print(
        "before training."
    )

else:

    print(
        TARGET_MANIFEST
    )

    print(
        "SHA256:",
        sha256_file(
            TARGET_MANIFEST
        ),
    )


# ------------------------------------------------------------
# Candidate split manifests
# ------------------------------------------------------------

print()
print("=" * 100)
print("STEP 34A SPLIT ARTIFACT SEARCH")
print("=" * 100)

split_candidates = []

for p in (
    ROOT
    / "manifests"
).glob("*"):

    if not p.is_file():
        continue

    name = p.name.lower()

    if (
        "split" in name
        or "patient_safe" in name
        or "patient" in name
        or "canonical" in name
    ):

        if p.suffix.lower() in {
            ".csv",
            ".json",
        }:

            split_candidates.append(
                p
            )

for p in sorted(
    split_candidates
):

    print(
        p
    )


# ------------------------------------------------------------
# Required Step 34A directories
# ------------------------------------------------------------

print()
print("=" * 100)
print("EXPECTED PUBLICATION DIRECTORIES")
print("=" * 100)

for p in [

    ROOT
    / "experiments",

    ROOT
    / "manifests",

]:

    print(
        str(p),
        "EXISTS=",
        p.is_dir(),
    )


# ------------------------------------------------------------
# Existing Step 34A local artifacts
# ------------------------------------------------------------

print()
print("=" * 100)
print("EXISTING LOCAL STEP 34A ARTIFACTS")
print("=" * 100)

step34_hits = []

for search_root in [
    ROOT / "experiments",
    PROJECT / "scripts",
]:

    if not search_root.is_dir():
        continue

    for p in search_root.rglob("*"):

        if not p.is_file():
            continue

        if "STEP34A" in p.name.upper():

            step34_hits.append(
                p
            )

for p in sorted(
    step34_hits
):

    print(
        p
    )

print()
print(
    "Local Step 34A artifacts found:",
    len(step34_hits),
)


# ------------------------------------------------------------
# Save preflight
# ------------------------------------------------------------

result = {

    "project":
        str(PROJECT),

    "cbis_root":
        str(ROOT),

    "cuda_available":
        bool(
            torch.cuda.is_available()
        ),

    "torch_version":
        torch.__version__,

    "manifests_found":
        [
            {
                "path":
                    str(p),

                "sha256":
                    sha256_file(p),
            }
            for p in found
        ],

    "target_manifest":
        (
            str(TARGET_MANIFEST)
            if TARGET_MANIFEST
            else None
        ),

    "target_manifest_exact_hash_match":
        TARGET_MANIFEST is not None,

    "step34a_local_artifacts":
        [
            str(p)
            for p in sorted(
                step34_hits
            )
        ],

    "status":
        (
            "READY_FOR_STEP34A_TRAINING"
            if TARGET_MANIFEST is not None
            and torch.cuda.is_available()
            else
            "REQUIRES_INPUT_RECONCILIATION"
        ),
}

OUTPUT = (
    OUT
    / "STEP34A_ASUS_PREFLIGHT.json"
)

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 100)
print("STEP 34A ASUS PREFLIGHT COMPLETE")
print("=" * 100)

print()
print(
    "Report:",
    OUTPUT,
)

print()
print(
    "STATUS:",
    result["status"],
)

print()
print("NO TRAINING WAS PERFORMED.")
print("NO DATA WERE MODIFIED.")
print("NO CHECKPOINTS WERE MODIFIED.")