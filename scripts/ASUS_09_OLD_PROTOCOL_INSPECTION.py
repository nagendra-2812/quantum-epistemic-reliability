from pathlib import Path
import json

OLD = Path(r"D:\AI\quantum-uncertainty-shift")

print("=" * 100)
print("ASUS-09 — OLD BreaKHis EXPERIMENT PROTOCOL INSPECTION")
print("=" * 100)

# ------------------------------------------------------------
# 1. Locate actual project source/config files
# ------------------------------------------------------------

print("\nACTUAL SOURCE FILES")
print("-" * 100)

roots = [
    OLD / "src",
    OLD / "features",
    OLD / "manuscript_evidence",
    OLD / "experiments",
]

patterns = [
    "*.py",
    "*.json",
    "*.yaml",
    "*.yml",
]

files = []

for root in roots:
    if root.exists():
        for pattern in patterns:
            files.extend(root.rglob(pattern))

for p in sorted(set(files)):
    print(p)

# ------------------------------------------------------------
# 2. Inspect one representative MLP configuration
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("REPRESENTATIVE MLP CONFIGURATIONS")
print("=" * 100)

mlp_configs = sorted(
    (OLD / "experiments" / "breakhis").glob(
        "fold_*/mlp/seed_*/config.json"
    )
)

for p in mlp_configs[:5]:
    print("\nFILE:", p)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2))
    except Exception as e:
        print("ERROR:", e)

# ------------------------------------------------------------
# 3. Inspect one representative VQC configuration
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("REPRESENTATIVE VQC CONFIGURATIONS")
print("=" * 100)

vqc_configs = sorted(
    (OLD / "experiments" / "breakhis").glob(
        "fold_*/vqc/seed_*/config.json"
    )
)

for p in vqc_configs[:5]:
    print("\nFILE:", p)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2))
    except Exception as e:
        print("ERROR:", e)

# ------------------------------------------------------------
# 4. Inspect training implementation
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("TRAINING IMPLEMENTATION")
print("=" * 100)

train_files = [
    OLD / "src" / "training" / "train_classifier.py",
    OLD / "manuscript_evidence" / "src" / "training" / "train_classifier.py",
]

for p in train_files:
    print("\nFILE:", p)
    if not p.exists():
        print("MISSING")
        continue

    text = p.read_text(encoding="utf-8", errors="replace")

    print("Lines:", len(text.splitlines()))

    keywords = [
        "ResNet",
        "resnet",
        "Adam",
        "AdamW",
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
        "batch_size",
        "learning_rate",
        "lr",
        "epochs",
        "seed",
        "scheduler",
        "early",
        "patience",
        "weight_decay",
        "image_size",
        "224",
        "512",
    ]

    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        if any(k in line for k in keywords):
            print(f"{i:04d}: {line}")

# ------------------------------------------------------------
# 5. Inspect quantum model implementation
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("QUANTUM MODEL IMPLEMENTATION")
print("=" * 100)

qfiles = [
    OLD / "src" / "models" / "quantum_vqc.py",
    OLD / "manuscript_evidence" / "src" / "models" / "quantum_vqc.py",
]

for p in qfiles:
    print("\nFILE:", p)

    if not p.exists():
        print("MISSING")
        continue

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    print("Lines:", len(lines))

    for i, line in enumerate(lines, 1):
        if any(
            k in line
            for k in [
                "n_qubits",
                "n_layers",
                "qml.",
                "QNode",
                "AngleEmbedding",
                "StronglyEntanglingLayers",
                "device",
                "shots",
                "diff_method",
            ]
        ):
            print(f"{i:04d}: {line}")

print("\n" + "=" * 100)
print("ASUS-09 COMPLETE")
print("=" * 100)
print("NO TRAINING PERFORMED.")
print("NO DATA MODIFIED.")
print("NO OLD PROJECT FILE MODIFIED.")
