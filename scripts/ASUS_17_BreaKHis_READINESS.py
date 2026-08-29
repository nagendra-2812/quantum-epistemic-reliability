from pathlib import Path
import pandas as pd
import json
import hashlib

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")
MAN = PROJECT / "manifests"
FREEZE = MAN / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"

print("=" * 100)
print("ASUS-17 — FROZEN MANIFEST EXPERIMENT-READINESS AUDIT")
print("=" * 100)

# ------------------------------------------------------------
# 1. Freeze record
# ------------------------------------------------------------

assert FREEZE.exists(), f"Missing freeze record: {FREEZE}"

freeze = json.loads(
    FREEZE.read_text(encoding="utf-8-sig")
)

assert freeze["status"] == "VERIFIED_FROZEN"
assert freeze["physical_image_count"] == 7909
assert freeze["fold_count"] == 5
assert freeze["rows_per_fold"] == 7909

print()
print("FREEZE RECORD")
print("-" * 100)
print("Status:", freeze["status"])
print("Physical images:", freeze["physical_image_count"])
print("Folds:", freeze["fold_count"])
print("Rows per fold:", freeze["rows_per_fold"])

# ------------------------------------------------------------
# 2. Validate each frozen manifest
# ------------------------------------------------------------

required_columns = {
    "image_path",
    "patient_id",
    "case_id",
    "label",
    "split",
    "fold",
    "original_image_path",
}

allowed_splits = {"train", "val", "test"}

fold_results = []

print()
print("FOLD READINESS")
print("-" * 100)

for i in range(1, 6):

    manifest = MAN / f"BreaKHis_FOLD_{i:02d}_VERIFIED_v1.csv"

    assert manifest.exists(), (
        f"Missing frozen manifest: {manifest}"
    )

    df = pd.read_csv(manifest)

    print()
    print(f"FOLD {i:02d}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Row count
    assert len(df) == 7909

    # Required columns
    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, (
        f"Fold {i}: missing columns {missing_columns}"
    )

    # Fold identity
    assert df["fold"].nunique() == 1
    assert int(df["fold"].iloc[0]) == i

    # Split semantics
    split_values = set(
        df["split"].dropna().unique()
    )

    assert split_values == allowed_splits

    train_count = int(
        (df["split"] == "train").sum()
    )

    val_count = int(
        (df["split"] == "val").sum()
    )

    test_count = int(
        (df["split"] == "test").sum()
    )

    assert train_count + val_count + test_count == 7909

    # Labels
    assert df["label"].notna().all()

    label_count = df["label"].nunique()

    assert label_count == 2

    # Image paths
    assert df["image_path"].notna().all()

    path_exists = df["image_path"].map(
        lambda x: Path(str(x)).is_file()
    )

    missing_files = int(
        (~path_exists).sum()
    )

    assert missing_files == 0

    # Duplicate image paths
    normalized_paths = (
        df["image_path"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    duplicate_paths = int(
        normalized_paths.duplicated().sum()
    )

    assert duplicate_paths == 0

    # Patient uniqueness within each split
    train_patients = set(
        df.loc[
            df["split"] == "train",
            "patient_id"
        ]
    )

    val_patients = set(
        df.loc[
            df["split"] == "val",
            "patient_id"
        ]
    )

    test_patients = set(
        df.loc[
            df["split"] == "test",
            "patient_id"
        ]
    )

    patient_tv = len(train_patients & val_patients)
    patient_tt = len(train_patients & test_patients)
    patient_vt = len(val_patients & test_patients)

    assert patient_tv == 0
    assert patient_tt == 0
    assert patient_vt == 0

    # Case uniqueness within each split
    train_cases = set(
        df.loc[
            df["split"] == "train",
            "case_id"
        ]
    )

    val_cases = set(
        df.loc[
            df["split"] == "val",
            "case_id"
        ]
    )

    test_cases = set(
        df.loc[
            df["split"] == "test",
            "case_id"
        ]
    )

    case_tv = len(train_cases & val_cases)
    case_tt = len(train_cases & test_cases)
    case_vt = len(val_cases & test_cases)

    assert case_tv == 0
    assert case_tt == 0
    assert case_vt == 0

    manifest_hash = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    print("Train:", train_count)
    print("Validation:", val_count)
    print("Test:", test_count)
    print("Labels:", label_count)
    print("Missing physical files:", missing_files)
    print("Duplicate image paths:", duplicate_paths)

    print(
        "Patient overlap:",
        patient_tv,
        patient_tt,
        patient_vt
    )

    print(
        "Case overlap:",
        case_tv,
        case_tt,
        case_vt
    )

    print("SHA256:", manifest_hash)

    print("✓ Fold is experiment-ready")

    fold_results.append(
        {
            "fold": i,
            "rows": len(df),
            "train": train_count,
            "validation": val_count,
            "test": test_count,
            "labels": label_count,
            "missing_physical_files": missing_files,
            "duplicate_image_paths": duplicate_paths,
            "patient_train_val_overlap": patient_tv,
            "patient_train_test_overlap": patient_tt,
            "patient_val_test_overlap": patient_vt,
            "case_train_val_overlap": case_tv,
            "case_train_test_overlap": case_tt,
            "case_val_test_overlap": case_vt,
            "sha256": manifest_hash,
        }
    )

# ------------------------------------------------------------
# 3. Save audit record
# ------------------------------------------------------------

audit_dir = PROJECT / "protocols" / "breakhis"
audit_dir.mkdir(
    parents=True,
    exist_ok=True
)

audit = {
    "step": "ASUS-17",
    "status": "VERIFIED",
    "dataset": "BreaKHis",
    "physical_images": 7909,
    "fold_count": 5,
    "rows_per_fold": 7909,
    "freeze_record": str(
        FREEZE.relative_to(PROJECT)
    ),
    "folds": fold_results,
    "new_random_patient_split": False,
    "source_dataset_modified": False,
    "training_performed": False,
    "kaggle_data_touched": False,
    "historical_project_used_for_execution": False,
}

outfile = (
    audit_dir /
    "ASUS-17_BreaKHis_EXPERIMENT_READINESS_v1.json"
)

outfile.write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 4. Final result
# ------------------------------------------------------------

print()
print("=" * 100)
print("ASUS-17 COMPLETE — VERIFIED / EXPERIMENT READY")
print("=" * 100)

print("✓ Frozen BreaKHis record verified")
print("✓ 7,909 physical images")
print("✓ Five frozen manifests")
print("✓ 7,909 rows per fold")
print("✓ Train/validation/test semantics valid")
print("✓ No duplicate image paths")
print("✓ All physical image paths valid")
print("✓ Patient-disjoint splits verified")
print("✓ Case-disjoint splits verified")
print("✓ Manifest SHA256 recorded")
print("✓ No new random split created")
print("✓ Source dataset not modified")
print("✓ Kaggle CBIS-DDSM not touched")
print("✓ Historical project not used for execution")
print()
print("Audit:", outfile)
print()
print("NO TRAINING PERFORMED")
print("=" * 100)
