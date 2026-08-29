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
print("STEP ASUS-05R — BreaKHis FAST VERIFIED MANIFEST FREEZE")
print("=" * 100)

# ------------------------------------------------------------------
# Physical dataset index
# ------------------------------------------------------------------
images = sorted(
    p for p in BH.rglob("*")
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

print("\nPHYSICAL DATASET")
print("Images:", len(images))

assert len(images) == 7909

# Fast filename index
by_name = {}
for p in images:
    key = p.name.lower()
    by_name.setdefault(key, []).append(p)

# ------------------------------------------------------------------
# Process five folds
# ------------------------------------------------------------------
source_hashes = {}
verified_hashes = {}
fold_validation = []

for i in range(1, 6):

    print("\n" + "-" * 100)
    print(f"FOLD {i:02d}")

    src = SPLIT / f"fold_{i:02d}.csv"
    assert src.exists(), f"Missing source fold: {src}"

    source_hashes[src.name] = hashlib.sha256(
        src.read_bytes()
    ).hexdigest()

    df = pd.read_csv(src)

    assert len(df) == 7909

    required = {
        "image_path",
        "patient_id",
        "case_id",
        "label",
        "split",
        "fold",
    }

    assert required.issubset(df.columns)

    assert df["fold"].nunique() == 1
    assert int(df["fold"].iloc[0]) == i

    assert set(df["split"].unique()) == {
        "train", "val", "test"
    }

    # No duplicate image records within fold
    paths = df["image_path"].astype(str).str.strip()
    assert paths.nunique() == 7909

    # --------------------------------------------------------------
    # Fast physical reconciliation
    # --------------------------------------------------------------
    resolved = []
    unresolved = []

    for value in paths:

        name = Path(value).name.lower()
        matches = by_name.get(name, [])

        if len(matches) != 1:
            unresolved.append(value)
        else:
            resolved.append(matches[0])

    print("Rows:", len(df))
    print("Unique image paths:", paths.nunique())
    print("Unresolved paths:", len(unresolved))

    assert not unresolved

    # --------------------------------------------------------------
    # Patient leakage
    # --------------------------------------------------------------
    trp = set(df.loc[df["split"] == "train", "patient_id"])
    vap = set(df.loc[df["split"] == "val", "patient_id"])
    tep = set(df.loc[df["split"] == "test", "patient_id"])

    pv = trp & vap
    pt = trp & tep
    vt = vap & tep

    print("Patient overlaps:",
          len(pv), len(pt), len(vt))

    assert not pv and not pt and not vt

    # --------------------------------------------------------------
    # Case leakage
    # --------------------------------------------------------------
    trc = set(df.loc[df["split"] == "train", "case_id"])
    vac = set(df.loc[df["split"] == "val", "case_id"])
    tec = set(df.loc[df["split"] == "test", "case_id"])

    cv = trc & vac
    ct = trc & tec
    vt_case = vac & tec

    print("Case overlaps:",
          len(cv), len(ct), len(vt_case))

    assert not cv and not ct and not vt_case

    # --------------------------------------------------------------
    # Create verified manifest
    # --------------------------------------------------------------
    corrected = df.copy()

    corrected["original_image_path"] = corrected["image_path"].astype(str)

    corrected["image_path"] = [
        str(p) for p in resolved
    ]

    outfile = OUT / f"BreaKHis_FOLD_{i:02d}_VERIFIED_v1.csv"

    corrected.to_csv(outfile, index=False)

    # Final physical verification
    assert len(corrected) == 7909
    assert corrected["image_path"].notna().all()

    missing = sum(
        not Path(p).is_file()
        for p in corrected["image_path"]
    )

    print("Missing physical files:", missing)

    assert missing == 0

    h = hashlib.sha256(
        outfile.read_bytes()
    ).hexdigest()

    verified_hashes[outfile.name] = h

    print("✓ VERIFIED:", outfile.name)
    print("SHA256:", h)

    fold_validation.append({
        "fold": i,
        "rows": len(df),
        "unique_image_paths": int(paths.nunique()),
        "unresolved_paths": len(unresolved),
        "patient_train_val_overlap": len(pv),
        "patient_train_test_overlap": len(pt),
        "patient_val_test_overlap": len(vt),
        "case_train_val_overlap": len(cv),
        "case_train_test_overlap": len(ct),
        "case_val_test_overlap": len(vt_case),
    })

# ------------------------------------------------------------------
# Freeze record
# ------------------------------------------------------------------
freeze = {
    "step": "ASUS-05R",
    "dataset": "BreaKHis",
    "status": "VERIFIED_FROZEN",
    "physical_root": str(BH),
    "physical_image_count": 7909,
    "fold_count": 5,
    "rows_per_fold": 7909,
    "source_project": str(OLD),
    "source_fold_sha256": source_hashes,
    "verified_manifest_sha256": verified_hashes,
    "fold_validation": fold_validation,
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

print("\n" + "=" * 100)
print("STEP ASUS-05R COMPLETE — VERIFIED / FROZEN")
print("=" * 100)
print("✓ 7,909 physical images")
print("✓ Five folds")
print("✓ 7,909 records per fold")
print("✓ Patient-disjoint")
print("✓ Case-disjoint")
print("✓ Physical paths reconciled")
print("✓ Verified manifests created")
print("✓ SHA256 recorded")
print("✓ Original project untouched")
print("✓ No training performed")
print("Freeze:", freeze_file)
print("=" * 100)
