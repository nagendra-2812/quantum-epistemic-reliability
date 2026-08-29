from pathlib import Path
import pandas as pd
import hashlib
import json

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")
MANIFESTS = PROJECT / "manifests"

print("=" * 100)
print("STEP ASUS-06 — BreaKHis FROZEN MANIFEST FINAL VERIFICATION")
print("=" * 100)

freeze = MANIFESTS / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"

print("\nFREEZE RECORD")
print("-" * 100)

if not freeze.exists():
    raise FileNotFoundError(f"Freeze record missing: {freeze}")

data = json.loads(freeze.read_text(encoding="utf-8"))

print("Status:", data.get("status"))
print("Physical images:", data.get("physical_image_count"))
print("Fold count:", data.get("fold_count"))
print("Rows per fold:", data.get("rows_per_fold"))

assert data["status"] == "VERIFIED_FROZEN"
assert data["physical_image_count"] == 7909
assert data["fold_count"] == 5
assert data["rows_per_fold"] == 7909

print("✓ Freeze metadata valid")

print("\nVERIFIED MANIFESTS")
print("-" * 100)

verified_hashes = data["verified_manifest_sha256"]

for i in range(1, 6):

    f = MANIFESTS / f"BreaKHis_FOLD_{i:02d}_VERIFIED_v1.csv"

    if not f.exists():
        raise FileNotFoundError(f"Missing manifest: {f}")

    df = pd.read_csv(f)

    print(f"\nFold {i:02d}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    assert len(df) == 7909
    assert df["image_path"].notna().all()

    missing = sum(not Path(p).is_file() for p in df["image_path"])

    print("Missing physical files:", missing)

    assert missing == 0

    actual_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    expected_hash = verified_hashes[f.name]

    print("SHA256:", actual_hash)

    assert actual_hash == expected_hash

    print("✓ Hash matches freeze record")
    print("✓ All 7,909 physical paths valid")

print("\n" + "=" * 100)
print("ASUS-06 COMPLETE")
print("=" * 100)
print("✓ Five verified manifests exist")
print("✓ 7,909 rows in every fold")
print("✓ All physical image paths valid")
print("✓ SHA256 hashes match frozen record")
print("✓ Frozen record verified")
print("✓ No training performed")
print("✓ No source dataset modified")
print("=" * 100)
