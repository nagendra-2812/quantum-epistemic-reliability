from pathlib import Path

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")

print("=" * 100)
print("ASUS-08 — NEW PROJECT IMPLEMENTATION AUDIT")
print("=" * 100)

print("\nPROJECT ROOT")
print("-" * 100)

for p in sorted(PROJECT.iterdir()):
    print(p)

print("\n" + "=" * 100)
print("NEW PROJECT FILE TREE")
print("=" * 100)

for p in sorted(PROJECT.rglob("*")):
    if p.is_file():
        print(p.relative_to(PROJECT))

print("\n" + "=" * 100)
print("MANIFESTS")
print("=" * 100)

manifest_dir = PROJECT / "manifests"

for p in sorted(manifest_dir.glob("*")):
    print(p.name, p.stat().st_size, "bytes")

print("\n" + "=" * 100)
print("SCRIPTS")
print("=" * 100)

scripts = PROJECT / "scripts"

for p in sorted(scripts.glob("*.py")):
    print(p.name)

print("\n" + "=" * 100)
print("SOURCE CODE AUDIT")
print("=" * 100)

keywords = [
    "ResNet",
    "resnet",
    "PCA",
    "pca",
    "BreaKHis",
    "breakhis",
    "patient_id",
    "case_id",
    "train",
    "val",
    "test",
    "seed",
    "Adam",
    "AdamW",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
    "n_qubits",
    "n_layers",
    "StronglyEntanglingLayers",
    "qml.",
    "QNode",
    "uncertainty",
    "calibration",
    "selective",
]

for p in sorted(PROJECT.rglob("*.py")):

    # Don't dump ASUS audit scripts themselves
    if p.parent.name == "__pycache__":
        continue

    text = p.read_text(
        encoding="utf-8",
        errors="replace"
    )

    matches = []

    for i, line in enumerate(text.splitlines(), 1):
        if any(k.lower() in line.lower() for k in keywords):
            matches.append((i, line))

    if matches:
        print("\n" + "-" * 100)
        print("FILE:", p.relative_to(PROJECT))
        print("LINES:", len(text.splitlines()))

        for i, line in matches:
            print(f"{i:04d}: {line}")

print("\n" + "=" * 100)
print("ASUS-08 COMPLETE")
print("=" * 100)
print("AUDIT ONLY.")
print("NO TRAINING PERFORMED.")
print("NO DATA MODIFIED.")
print("=" * 100)
