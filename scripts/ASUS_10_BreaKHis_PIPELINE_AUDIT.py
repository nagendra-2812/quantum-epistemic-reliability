from pathlib import Path
import re

OLD = Path(r"D:\AI\quantum-uncertainty-shift")

print("=" * 100)
print("ASUS-10 — BreaKHis OLD DATA / FEATURE PIPELINE AUDIT")
print("=" * 100)

files = [
    OLD / "src" / "data" / "dataset.py",
    OLD / "src" / "data" / "create_splits.py",
    OLD / "src" / "data" / "build_manifest.py",
    OLD / "features" / "extract_features.py",
    OLD / "src" / "training" / "train_classifier.py",
    OLD / "src" / "models" / "classical_mlp.py",
]

keywords = [
    "breakhis",
    "image_path",
    "patient_id",
    "case_id",
    "label",
    "split",
    "fold",
    "train",
    "val",
    "test",
    "ResNet18",
    "resnet18",
    "ResNet-18",
    "weights",
    "pretrained",
    "ImageNet",
    "Normalize",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "PCA",
    "PCA(",
    "fit(",
    "transform(",
    "StandardScaler",
    "training-only",
    "fit_transform",
    "seed",
]

for f in files:

    print("\n" + "=" * 100)
    print("FILE:", f)
    print("=" * 100)

    if not f.exists():
        print("MISSING")
        continue

    text = f.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    print("Total lines:", len(lines))

    for i, line in enumerate(lines, 1):
        if any(k.lower() in line.lower() for k in keywords):
            print(f"{i:04d}: {line}")

print("\n" + "=" * 100)
print("ASUS-10 COMPLETE")
print("=" * 100)
print("NO TRAINING PERFORMED.")
print("NO DATA MODIFIED.")
print("NO OLD PROJECT FILE MODIFIED.")
print("=" * 100)
