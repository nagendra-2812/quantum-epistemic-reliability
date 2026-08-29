from pathlib import Path
import json
import hashlib
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT / "manifests"
FREEZE = MANIFEST_DIR / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"
OUT = PROJECT / "metadata" / "ASUS-15_DATALOADER_VERIFICATION_v1.json"

DATASET_ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM\BreaKHis_v1\BreaKHis_v1"
)

print("=" * 100)
print("ASUS-15 — BreaKHis REPRODUCIBLE DATA LOADER VERIFICATION")
print("=" * 100)

# ------------------------------------------------------------------
# 1. Contract / freeze verification
# ------------------------------------------------------------------

assert FREEZE.exists(), f"Missing freeze record: {FREEZE}"

freeze = json.loads(
    FREEZE.read_text(encoding="utf-8-sig")
)

assert freeze["status"] == "VERIFIED_FROZEN"
assert freeze["physical_image_count"] == 7909
assert freeze["fold_count"] == 5
assert freeze["rows_per_fold"] == 7909
assert freeze["patient_disjoint_all_folds"] is True
assert freeze["case_disjoint_all_folds"] is True

print("\nFREEZE:")
print("  Physical images:", freeze["physical_image_count"])
print("  Folds:", freeze["fold_count"])
print("  Rows/fold:", freeze["rows_per_fold"])
print("  Patient-disjoint:", freeze["patient_disjoint_all_folds"])
print("  Case-disjoint:", freeze["case_disjoint_all_folds"])
print("  ✓ Frozen dataset contract verified")

# ------------------------------------------------------------------
# 2. Dataset root verification
# ------------------------------------------------------------------

assert DATASET_ROOT.exists(), DATASET_ROOT

physical_images = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

assert len(physical_images) == 7909

print("\nDATASET:")
print("  Root:", DATASET_ROOT)
print("  Physical images:", len(physical_images))
print("  ✓ Physical dataset verified")

# ------------------------------------------------------------------
# 3. Frozen manifest verification
# ------------------------------------------------------------------

required_columns = {
    "image_path",
    "patient_id",
    "case_id",
    "label",
    "split",
    "fold",
}

manifest_hashes = {}
manifest_summary = {}

for fold in range(1, 6):

    manifest = (
        MANIFEST_DIR
        / f"BreaKHis_FOLD_{fold:02d}_VERIFIED_v1.csv"
    )

    assert manifest.exists(), manifest

    sha256 = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    expected = freeze["verified_manifest_sha256"][
        manifest.name
    ]

    assert sha256 == expected, (
        f"Hash mismatch: {manifest.name}"
    )

    df = pd.read_csv(manifest)

    assert len(df) == 7909
    assert required_columns.issubset(df.columns)

    assert df["image_path"].notna().all()
    assert df["patient_id"].notna().all()
    assert df["case_id"].notna().all()
    assert df["label"].notna().all()
    assert df["split"].notna().all()

    assert set(df["split"].unique()) == {
        "train",
        "val",
        "test",
    }

    assert df["fold"].nunique() == 1
    assert int(df["fold"].iloc[0]) == fold

    duplicate_paths = int(
        df["image_path"].duplicated().sum()
    )

    assert duplicate_paths == 0

    manifest_hashes[manifest.name] = sha256

    manifest_summary[f"fold_{fold:02d}"] = {
        "rows": len(df),
        "train": int((df["split"] == "train").sum()),
        "val": int((df["split"] == "val").sum()),
        "test": int((df["split"] == "test").sum()),
        "benign": int((df["label"] == "benign").sum()),
        "malignant": int((df["label"] == "malignant").sum()),
        "unique_patients": int(df["patient_id"].nunique()),
        "unique_cases": int(df["case_id"].nunique()),
        "duplicate_paths": duplicate_paths,
    }

    print(
        f"  ✓ Fold {fold:02d}: "
        f"{len(df)} rows | "
        f"train={manifest_summary[f'fold_{fold:02d}']['train']} | "
        f"val={manifest_summary[f'fold_{fold:02d}']['val']} | "
        f"test={manifest_summary[f'fold_{fold:02d}']['test']}"
    )

# ------------------------------------------------------------------
# 4. Dataset class
# ------------------------------------------------------------------

class BreaKHisDataset(Dataset):

    LABEL_MAP = {
        "benign": 0,
        "malignant": 1,
    }

    def __init__(self, dataframe, image_size=224):

        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size

        unknown = set(self.df["label"]) - set(self.LABEL_MAP)

        assert not unknown, (
            f"Unknown labels: {unknown}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = Path(row["image_path"])

        assert path.is_file(), (
            f"Missing image: {path}"
        )

        with Image.open(path) as image:

            image = image.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

        tensor = torch.from_numpy(
            array.transpose(2, 0, 1)
        ).contiguous()

        label = self.LABEL_MAP[row["label"]]

        return {
            "image": tensor,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "image_path": str(path),
            "patient_id": int(row["patient_id"]),
            "case_id": str(row["case_id"]),
        }

# ------------------------------------------------------------------
# 5. Verify every fold's split semantics
# ------------------------------------------------------------------

print("\nSPLIT SEMANTICS:")
print("-" * 100)

fold_objects = {}

for fold in range(1, 6):

    manifest = (
        MANIFEST_DIR
        / f"BreaKHis_FOLD_{fold:02d}_VERIFIED_v1.csv"
    )

    df = pd.read_csv(manifest)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert not train_patients & val_patients
    assert not train_patients & test_patients
    assert not val_patients & test_patients

    train_cases = set(train_df["case_id"])
    val_cases = set(val_df["case_id"])
    test_cases = set(test_df["case_id"])

    assert not train_cases & val_cases
    assert not train_cases & test_cases
    assert not val_cases & test_cases

    fold_objects[fold] = {
        "dataframe": df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    print(
        f"  ✓ Fold {fold:02d}: "
        f"patient-disjoint + case-disjoint"
    )

# ------------------------------------------------------------------
# 6. Loader smoke test — Fold 01 only
# ------------------------------------------------------------------

print("\nDATALOADER SMOKE TEST:")
print("-" * 100)

fold1_train = fold_objects[1]["train"]

dataset = BreaKHisDataset(
    fold1_train,
    image_size=224,
)

assert len(dataset) == len(fold1_train)

sample = dataset[0]

assert sample["image"].shape == (3, 224, 224)
assert sample["image"].dtype == torch.float32
assert 0.0 <= float(sample["image"].min())
assert float(sample["image"].max()) <= 1.0
assert sample["label"].item() in {0, 1}

print("  Sample image shape:", tuple(sample["image"].shape))
print("  Sample dtype:", sample["image"].dtype)
print("  Sample label:", sample["label"].item())
print("  Sample patient:", sample["patient_id"])
print("  Sample case:", sample["case_id"])
print("  ✓ Single-sample loader test passed")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

batch = next(iter(loader))

assert batch["image"].shape == (8, 3, 224, 224)
assert batch["image"].dtype == torch.float32
assert batch["label"].shape == (8,)

print("  Batch image shape:", tuple(batch["image"].shape))
print("  Batch label shape:", tuple(batch["label"].shape))
print("  ✓ Batch DataLoader test passed")

# ------------------------------------------------------------------
# 7. Reproducibility check
# ------------------------------------------------------------------

print("\nREPRODUCIBILITY CHECK:")
print("-" * 100)

SEED = 20260829

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch_a = torch.rand(10)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch_b = torch.rand(10)

assert torch.equal(torch_a, torch_b)

print("  Seed:", SEED)
print("  ✓ Deterministic seed test passed")

# ------------------------------------------------------------------
# 8. Final provenance record
# ------------------------------------------------------------------

record = {
    "step": "ASUS-15",
    "status": "VERIFIED",
    "dataset": "BreaKHis",
    "dataset_root": str(DATASET_ROOT),
    "physical_images": 7909,
    "fold_count": 5,
    "rows_per_fold": 7909,
    "freeze_record": str(
        FREEZE.relative_to(PROJECT)
    ),
    "manifest_sha256": manifest_hashes,
    "manifest_summary": manifest_summary,
    "loader": {
        "class": "BreaKHisDataset",
        "image_size": 224,
        "channels": 3,
        "dtype": "float32",
        "value_range": [0.0, 1.0],
        "label_map": {
            "benign": 0,
            "malignant": 1,
        },
    },
    "split_integrity": {
        "patient_disjoint": True,
        "case_disjoint": True,
        "new_random_split": False,
    },
    "smoke_test": {
        "sample_shape": [3, 224, 224],
        "batch_size": 8,
        "batch_shape": [8, 3, 224, 224],
        "num_workers": 0,
    },
    "seed_test": {
        "seed": SEED,
        "deterministic_tensor_reproduction": True,
    },
    "training_performed": False,
    "dataset_modified": False,
    "kaggle_touched": False,
    "historical_project_used_for_execution": False,
}

OUT.write_text(
    json.dumps(record, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 100)
print("ASUS-15 COMPLETE — VERIFIED")
print("=" * 100)
print("✓ Frozen manifests loaded")
print("✓ Manifest SHA256 hashes verified")
print("✓ 7,909 physical images confirmed")
print("✓ Five folds verified")
print("✓ Patient-disjoint verified")
print("✓ Case-disjoint verified")
print("✓ BreaKHisDataset implemented")
print("✓ RGB conversion verified")
print("✓ 224×224 preprocessing verified")
print("✓ [0,1] tensor range verified")
print("✓ DataLoader batch verified")
print("✓ Deterministic seed verified")
print("✓ Provenance record written:")
print(OUT)
print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("NO HISTORICAL PROJECT USED FOR EXECUTION")
print("=" * 100)
