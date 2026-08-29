from pathlib import Path
import json
import hashlib

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")
OLD = Path(r"D:\AI\quantum-uncertainty-shift")
MANIFESTS = PROJECT / "manifests"

print("=" * 100)
print("ASUS-08 — BreaKHis EXISTING PROJECT CONFIGURATION RECONCILIATION")
print("=" * 100)

print("\nCURRENT PROJECT")
print(PROJECT)

print("\nPREVIOUS PROJECT")
print(OLD)

print("\nFROZEN BreaKHis MANIFEST")
freeze = MANIFESTS / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"

assert freeze.exists(), "Frozen BreaKHis freeze record missing."

freeze_data = json.loads(
    freeze.read_text(encoding="utf-8")
)

print("Status:", freeze_data["status"])
print("Physical images:", freeze_data["physical_image_count"])
print("Folds:", freeze_data["fold_count"])
print("Rows/fold:", freeze_data["rows_per_fold"])

print("\nPREVIOUS PROJECT CONFIG / SOURCE FILES")

candidates = []

for pattern in [
    "*.json",
    "*.yaml",
    "*.yml",
    "*.py",
    "*.ipynb",
]:
    candidates.extend(OLD.rglob(pattern))

for p in sorted(candidates):
    name = p.name.lower()

    if any(
        key in name
        for key in [
            "config",
            "train",
            "baseline",
            "experiment",
            "feature",
            "uncertainty",
            "quantum",
            "model",
        ]
    ):
        print(p)

print("\nEXISTING SPLIT ARTIFACTS")

split_root = OLD / "splits" / "breakhis"

for p in sorted(split_root.glob("fold_*.csv")):
    print(
        p.name,
        "| SHA256:",
        hashlib.sha256(p.read_bytes()).hexdigest()
    )

print("\n" + "=" * 100)
print("ASUS-08 AUDIT COMPLETE")
print("=" * 100)
print("NO TRAINING PERFORMED.")
print("NO DATA MODIFIED.")
print("NO OLD PROJECT FILE MODIFIED.")
