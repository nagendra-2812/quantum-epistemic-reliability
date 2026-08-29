from pathlib import Path
import pandas as pd
import hashlib
import json

OLD = Path(r"D:\AI\quantum-uncertainty-shift")
BH = Path(r"E:\CBIS_DDSM_QUANTUM\BreaKHis_v1\BreaKHis_v1")
SPLIT = OLD / "splits" / "breakhis"

print("=" * 100)
print("STEP ASUS-02 — BreaKHis FIVE-FOLD RECONCILIATION")
print("=" * 100)

# Physical images
images = sorted(
    p for p in BH.rglob("*")
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

print("\nPHYSICAL DATASET")
print("Root:", BH)
print("Images:", len(images))

# Existing folds
fold_files = [SPLIT / f"fold_{i:02d}.csv" for i in range(1, 6)]

print("\nEXISTING FIVE FOLDS")
for f in fold_files:
    print(("✓ " if f.exists() else "✗ "), f)

if not all(f.exists() for f in fold_files):
    raise FileNotFoundError("One or more fold files are missing.")

frames = []
hashes = {}

print("\nFOLD DETAILS")
print("-" * 100)

for i, f in enumerate(fold_files, 1):
    df = pd.read_csv(f)
    frames.append(df)

    h = hashlib.sha256(f.read_bytes()).hexdigest()
    hashes[f.name] = h

    print(f"\nFold {i:02d}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Missing cells:", int(df.isna().sum().sum()))

combined = pd.concat(frames, ignore_index=True)

print("\nCOMBINED")
print("-" * 100)
print("Total rows:", len(combined))

# Identify columns
path_candidates = [
    c for c in combined.columns
    if any(x in c.lower() for x in
           ["path", "filepath", "file_path", "image_path", "filename", "image"])
]

label_candidates = [
    c for c in combined.columns
    if any(x in c.lower() for x in
           ["label", "class", "diagnosis", "target"])
]

identity_candidates = [
    c for c in combined.columns
    if any(x in c.lower() for x in
           ["patient", "specimen", "subject", "case", "slide", "lesion"])
]

print("Path candidates:", path_candidates)
print("Label candidates:", label_candidates)
print("Identity candidates:", identity_candidates)

if not path_candidates:
    raise RuntimeError("Could not identify image-path column.")

path_col = path_candidates[0]

def norm(x):
    return str(x).strip().replace("\\", "/").lstrip("./")

split_paths = combined[path_col].map(norm)

physical_rel = {
    p.relative_to(BH).as_posix().lower()
    for p in images
}

exact = sum(x.lower() in physical_rel for x in split_paths)
unique_paths = len(set(split_paths.str.lower()))
duplicate_rows = int(split_paths.duplicated(keep=False).sum())

print("\nPATH RECONCILIATION")
print("-" * 100)
print("Selected path column:", path_col)
print("Exact physical-path matches:", exact)
print("Unique split paths:", unique_paths)
print("Duplicated split-path rows:", duplicate_rows)

print("\nFOLD COUNTS")
for i, df in enumerate(frames, 1):
    print(f"Fold {i:02d}: {len(df)}")

# Fold overlap
sets = {
    i: set(df[path_col].map(norm).str.lower())
    for i, df in enumerate(frames, 1)
}

print("\nFOLD OVERLAP")
for i in range(1, 6):
    for j in range(i + 1, 6):
        print(
            f"Fold {i:02d} ∩ Fold {j:02d}: "
            f"{len(sets[i] & sets[j])}"
        )

# Labels
if label_candidates:
    label_col = label_candidates[0]
    print("\nLABEL DISTRIBUTION")
    print(combined[label_col].value_counts(dropna=False).to_string())
else:
    label_col = None

# Identity
print("\nIDENTITY AUDIT")
for c in identity_candidates:
    print(
        f"{c}: unique={combined[c].nunique(dropna=True)}, "
        f"missing={combined[c].isna().sum()}"
    )

# Hashes
print("\nSHA256")
for name, value in hashes.items():
    print(name, ":", value)

# Save audit
outdir = Path("protocols") / "breakhis"
outdir.mkdir(parents=True, exist_ok=True)

audit = {
    "step": "ASUS-02",
    "dataset": "BreaKHis",
    "physical_root": str(BH),
    "physical_images": len(images),
    "combined_fold_rows": len(combined),
    "path_column": path_col,
    "label_column": label_col,
    "identity_candidates": identity_candidates,
    "exact_physical_path_matches": exact,
    "unique_split_paths": unique_paths,
    "duplicate_split_path_rows": duplicate_rows,
    "fold_row_counts": {
        f"fold_{i:02d}": len(frames[i - 1])
        for i in range(1, 6)
    },
    "fold_overlap": {
        f"fold_{i:02d}_x_fold_{j:02d}":
        len(sets[i] & sets[j])
        for i in range(1, 6)
        for j in range(i + 1, 6)
    },
    "sha256": hashes
}

outfile = outdir / "ASUS-02_BreaKHis_EXISTING_FIVE_FOLD_AUDIT_v1.json"
outfile.write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\n" + "=" * 100)
print("STEP ASUS-02 COMPLETE")
print("=" * 100)
print("Audit:", outfile)
print("NO TRAINING PERFORMED.")
print("NO DATA MODIFIED.")
print("NO OLD PROJECT FILE MODIFIED.")
