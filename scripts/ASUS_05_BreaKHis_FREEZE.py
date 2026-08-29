from pathlib import Path
import pandas as pd
import hashlib
import json

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")
OLD = Path(r"D:\AI\quantum-uncertainty-shift")
BH = Path(r"E:\CBIS_DDSM_QUANTUM\BreaKHis_v1\BreaKHis_v1")
SPLIT = OLD / "splits" / "breakhis"
OUT = PROJECT / "manifests"

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("STEP ASUS-05 — BreaKHis VERIFIED FIVE-FOLD MANIFEST FREEZE")
print("=" * 100)

# ------------------------------------------------------------
# 1. Physical dataset
# ------------------------------------------------------------
images = sorted(
    p for p in BH.rglob("*")
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

print(f"\nPhysical images: {len(images)}")
assert len(images) == 7909, f"Expected 7909 images, found {len(images)}"

physical = {
    p.relative_to(BH).as_posix().lower(): p
    for p in images
}

# ------------------------------------------------------------
# 2. Read original folds
# ------------------------------------------------------------
folds = []
source_hashes = {}

print("\nSOURCE FOLDS")
print("-" * 100)

for i in range(1, 6):
    src = SPLIT / f"fold_{i:02d}.csv"

    assert src.exists(), f"Missing source fold: {src}"

    df = pd.read_csv(src)

    print(
        f"Fold {i:02d}: rows={len(df)}, "
        f"columns={list(df.columns)}"
    )

    assert len(df) == 7909
    assert set(
        ["image_path", "patient_id", "case_id", "label", "split", "fold"]
    ).issubset(df.columns)

    folds.append(df)
    source_hashes[src.name] = hashlib.sha256(
        src.read_bytes()
    ).hexdigest()

# ------------------------------------------------------------
# 3. Validation
# ------------------------------------------------------------
print("\nVALIDATION")
print("-" * 100)

all_results = []
verified_fold_paths = []

def norm_path(x):
    return str(x).strip().replace("\\", "/").lower()

for i, df in enumerate(folds, 1):

    print(f"\nFOLD {i:02d}")

    # fold identifier
    assert df["fold"].nunique() == 1
    assert int(df["fold"].iloc[0]) == i

    # split values
    allowed = {"train", "val", "test"}
    actual = set(df["split"].dropna().unique())

    assert actual == allowed, (
        f"Fold {i}: unexpected split values {actual}"
    )

    # duplicate images
    normalized = df["image_path"].map(norm_path)

    duplicate_count = int(normalized.duplicated().sum())

    print("Duplicate image rows:", duplicate_count)

    assert duplicate_count == 0

    # physical path reconciliation
    unresolved = []

    for value in normalized:
        if value.startswith("e:/dataset/"):
            # Old absolute root → locate using filename-relative suffix.
            marker = "/breast/"
            if marker in value:
                rel = "histology_slides" + value.split(marker, 1)[1]
            else:
                rel = ""
        else:
            rel = value

        if rel not in physical:
            # Try filename-based exact resolution.
            filename = Path(value).name
            matches = [
                key for key in physical
                if Path(key).name.lower() == filename
            ]

            if len(matches) != 1:
                unresolved.append(value)

    print("Unresolved physical paths:", len(unresolved))

    assert not unresolved, (
        f"Fold {i}: {len(unresolved)} paths could not be reconciled"
    )

    # Patient leakage
    train_patients = set(
        df.loc[df["split"] == "train", "patient_id"]
    )
    val_patients = set(
        df.loc[df["split"] == "val", "patient_id"]
    )
    test_patients = set(
        df.loc[df["split"] == "test", "patient_id"]
    )

    pv = train_patients & val_patients
    pt = train_patients & test_patients
    vt = val_patients & test_patients

    print("Patient train/val overlap:", len(pv))
    print("Patient train/test overlap:", len(pt))
    print("Patient val/test overlap:", len(vt))

    assert not pv and not pt and not vt

    # Case leakage
    train_cases = set(
        df.loc[df["split"] == "train", "case_id"]
    )
    val_cases = set(
        df.loc[df["split"] == "val", "case_id"]
    )
    test_cases = set(
        df.loc[df["split"] == "test", "case_id"]
    )

    cv = train_cases & val_cases
    ct = train_cases & test_cases
    vt_case = val_cases & test_cases

    print("Case train/val overlap:", len(cv))
    print("Case train/test overlap:", len(ct))
    print("Case val/test overlap:", len(vt_case))

    assert not cv and not ct and not vt_case

    all_results.append({
        "fold": i,
        "rows": len(df),
        "duplicate_images": duplicate_count,
        "unresolved_paths": len(unresolved),
        "patient_train_val_overlap": len(pv),
        "patient_train_test_overlap": len(pt),
        "patient_val_test_overlap": len(vt),
        "case_train_val_overlap": len(cv),
        "case_train_test_overlap": len(ct),
        "case_val_test_overlap": len(vt_case),
    })

# ------------------------------------------------------------
# 4. Create corrected manifests
# ------------------------------------------------------------
print("\nCREATING VERIFIED MANIFESTS")
print("-" * 100)

for i, df in enumerate(folds, 1):

    corrected = df.copy()

    original_paths = corrected["image_path"].astype(str)

    corrected["original_image_path"] = original_paths

    new_paths = []

    for value in original_paths:

        normalized = norm_path(value)

        # First attempt: path relative to /breast/
        marker = "/breast/"

        if marker in normalized:
            rel = "histology_slides" + normalized.split(
                marker, 1
            )[1]
        else:
            rel = ""

        if rel in physical:
            new_paths.append(str(physical[rel]))
            continue

        # Fallback: unique basename
        filename = Path(normalized).name

        matches = [
            p for key, p in physical.items()
            if Path(key).name.lower() == filename
        ]

        assert len(matches) == 1, (
            f"Could not uniquely resolve: {value}"
        )

        new_paths.append(str(matches[0]))

    corrected["image_path"] = new_paths

    outfile = OUT / f"BreaKHis_FOLD_{i:02d}_VERIFIED_v1.csv"

    corrected.to_csv(outfile, index=False)

    print(
        f"✓ Fold {i:02d}: {outfile.name} "
        f"({len(corrected)} rows)"
    )

    verified_fold_paths.append(outfile)

# ------------------------------------------------------------
# 5. Verify generated manifests
# ------------------------------------------------------------
print("\nPOST-CREATION VERIFICATION")
print("-" * 100)

verified_hashes = {}

for outfile in verified_fold_paths:

    df = pd.read_csv(outfile)

    assert len(df) == 7909
    assert df["image_path"].notna().all()

    missing = [
        p for p in df["image_path"]
        if not Path(p).is_file()
    ]

    assert not missing, (
        f"{outfile.name}: {len(missing)} physical files missing"
    )

    verified_hashes[outfile.name] = hashlib.sha256(
        outfile.read_bytes()
    ).hexdigest()

    print(f"✓ {outfile.name}: 7909 rows; all paths valid")

# ------------------------------------------------------------
# 6. Freeze record
# ------------------------------------------------------------
freeze = {
    "step": "ASUS-05",
    "dataset": "BreaKHis",
    "status": "VERIFIED_FROZEN",
    "physical_root": str(BH),
    "physical_image_count": len(images),
    "fold_count": 5,
    "rows_per_fold": 7909,
    "source_project": str(OLD),
    "source_fold_sha256": source_hashes,
    "verified_manifest_sha256": verified_hashes,
    "fold_validation": all_results,
    "patient_disjoint_all_folds": True,
    "case_disjoint_all_folds": True,
    "training_performed": False,
    "source_data_modified": False,
}

freeze_file = OUT / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"

freeze_file.write_text(
    json.dumps(freeze, indent=2),
    encoding="utf-8"
)

print("\nFREEZE RECORD:")
print(freeze_file)

print("\n" + "=" * 100)
print("STEP ASUS-05 COMPLETE — VERIFIED / FROZEN")
print("=" * 100)
print("✓ 7,909 physical images")
print("✓ 5 × 7,909 source fold records")
print("✓ Patient-disjoint")
print("✓ Case-disjoint")
print("✓ Physical paths reconciled")
print("✓ Corrected manifests created")
print("✓ SHA256 recorded")
print("✓ Original project untouched")
print("✓ No training performed")
print("=" * 100)
