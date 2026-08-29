from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import resnet18, ResNet18_Weights


# ================================================================
# ASUS-20B
# REAL RESNET-18 FEATURE EXTRACTION + FOLD-SAFE PCA
# ================================================================

PROJECT = Path(__file__).resolve().parents[1]

CONFIG = PROJECT / "configs" / "ASUS_19_IMPLEMENTATION_CONTRACT_v1.json"
FREEZE = PROJECT / "manifests" / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"

DATASET_ROOT = Path(
    r"E:\CBIS_DDSM_QUANTUM\BreaKHis_v1\BreaKHis_v1"
)

FEATURE_ROOT = PROJECT / "features" / "BreaKHis"
CACHE_DIR = FEATURE_ROOT / "resnet18_512_cache"
PCA_DIR = FEATURE_ROOT / "pca_6d"

METADATA_DIR = PROJECT / "metadata"

SEEDS = [42, 123, 2025]

FEATURE_DIM = 512
PCA_DIM = 6

BATCH_SIZE = 32
NUM_WORKERS = 0

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ================================================================
# HELPERS
# ================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_command(args):
    return subprocess.check_output(
        ["git"] + args,
        cwd=PROJECT,
        text=True
    ).strip()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def normalize_path(path_value):
    return str(Path(path_value).resolve())


# ================================================================
# HEADER
# ================================================================

print("=" * 100)
print("ASUS-20B — RESNET-18 512-D FEATURE EXTRACTION + FOLD-SAFE PCA")
print("=" * 100)


# ================================================================
# 1. REQUIRED ARTIFACTS
# ================================================================

print()
print("REQUIRED ARTIFACTS")
print("-" * 100)

assert CONFIG.exists(), f"Missing implementation contract: {CONFIG}"
assert FREEZE.exists(), f"Missing BreaKHis freeze: {FREEZE}"
assert DATASET_ROOT.exists(), f"Missing dataset root: {DATASET_ROOT}"

config = json.loads(
    CONFIG.read_text(encoding="utf-8-sig")
)

freeze = json.loads(
    FREEZE.read_text(encoding="utf-8-sig")
)

assert config["project"] == "quantum-epistemic-reliability"
assert config["dataset"] == "BreaKHis"

assert config["experiment"]["outer_folds"] == 5
assert config["experiment"]["seeds"] == SEEDS

assert config["experiment"]["feature_dimension"] == FEATURE_DIM
assert config["experiment"]["pca_components"] == PCA_DIM

assert config["pca_policy"]["fit_on_training_data_only"] is True
assert config["pca_policy"]["validation_transform_only"] is True
assert config["pca_policy"]["test_transform_only"] is True

assert freeze["status"] == "VERIFIED_FROZEN"
assert freeze["physical_image_count"] == 7909
assert freeze["fold_count"] == 5
assert freeze["rows_per_fold"] == 7909

print("✓ ASUS-19 contract verified")
print("✓ BreaKHis freeze verified")
print("✓ Dataset root verified")


# ================================================================
# 2. GIT PROVENANCE
# ================================================================

print()
print("GIT PROVENANCE")
print("-" * 100)

git_branch = git_command(["branch", "--show-current"])
git_commit = git_command(["rev-parse", "HEAD"])
git_status = git_command(["status", "--porcelain"])

print("Branch:", git_branch)
print("Commit:", git_commit)
print("Working tree clean:", git_status == "")

assert git_status == "", (
    "Working tree must be clean before ASUS-20B execution."
)

print("✓ Git provenance valid")
print("✓ Working tree clean")


# ================================================================
# 3. DATASET PHYSICAL IMAGE INVENTORY
# ================================================================

print()
print("DATASET INVENTORY")
print("-" * 100)

physical_images = sorted(
    p.resolve()
    for p in DATASET_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

print("Physical image count:", len(physical_images))

assert len(physical_images) == 7909, (
    f"Expected 7909 physical images, found {len(physical_images)}"
)

print("✓ Exactly 7,909 physical images found")


# ================================================================
# 4. LOAD ALL FROZEN MANIFESTS
# ================================================================

print()
print("FROZEN MANIFESTS")
print("-" * 100)

manifests = {}

for fold in range(1, 6):

    manifest = (
        PROJECT
        / "manifests"
        / f"BreaKHis_FOLD_{fold:02d}_VERIFIED_v1.csv"
    )

    assert manifest.exists(), f"Missing manifest: {manifest}"

    df = pd.read_csv(manifest)

    # Frozen manifests store labels as text. Convert once to the
    # locked binary representation required by the experiment.
    df["label_numeric"] = df["label"].map({
        "benign": 0,
        "malignant": 1,
    })

    assert df["label_numeric"].notna().all(), (
        f"Unexpected label value in fold {fold:02d}"
    )

    assert set(df["label_numeric"].unique()) == {0, 1}

    assert len(df) == 7909

    assert set(df["split"].unique()) == {
        "train",
        "val",
        "test",
    }

    assert df["image_path"].notna().all()

    df["resolved_image_path"] = (
        df["image_path"]
        .map(normalize_path)
    )

    assert df["resolved_image_path"].is_unique, (
        f"Duplicate image paths in fold {fold:02d}"
    )

    missing = sum(
        not Path(p).is_file()
        for p in df["resolved_image_path"]
    )

    assert missing == 0, (
        f"Fold {fold:02d} has {missing} missing images"
    )

    manifests[fold] = df

    print(
        f"Fold {fold:02d}: "
        f"rows={len(df)}, "
        f"train={(df['split'] == 'train').sum()}, "
        f"val={(df['split'] == 'val').sum()}, "
        f"test={(df['split'] == 'test').sum()}"
    )

print("✓ Five frozen manifests loaded")


# ================================================================
# 5. VERIFY ALL FOLDS COVER THE SAME 7,909 IMAGES
# ================================================================

print()
print("CROSS-FOLD IMAGE COVERAGE")
print("-" * 100)

reference_paths = set(
    manifests[1]["resolved_image_path"]
)

assert len(reference_paths) == 7909

for fold in range(2, 6):

    current_paths = set(
        manifests[fold]["resolved_image_path"]
    )

    assert current_paths == reference_paths, (
        f"Fold {fold:02d} does not contain the same physical image universe"
    )

print("✓ Every frozen fold references the same 7,909 images")


# ================================================================
# 6. DEVICE
# ================================================================

print()
print("COMPUTE DEVICE")
print("-" * 100)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    print("GPU: NONE")

print("✓ Device selected")


# ================================================================
# 7. DETERMINISM
# ================================================================

BASE_SEED = SEEDS[0]

set_seed(BASE_SEED)

print()
print("DETERMINISM")
print("-" * 100)

print("Seed:", BASE_SEED)
print(
    "cuDNN deterministic:",
    torch.backends.cudnn.deterministic
)
print(
    "cuDNN benchmark:",
    torch.backends.cudnn.benchmark
)

assert torch.backends.cudnn.deterministic is True
assert torch.backends.cudnn.benchmark is False

print("✓ Deterministic configuration active")


# ================================================================
# 8. FROZEN RESNET-18
# ================================================================

print()
print("FROZEN RESNET-18")
print("-" * 100)

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

for parameter in feature_extractor.parameters():
    parameter.requires_grad = False

feature_extractor.to(device)

parameter_count = sum(
    p.numel()
    for p in feature_extractor.parameters()
)

print("Model: ResNet-18")
print("Weights:", weights)
print("Feature dimension:", FEATURE_DIM)
print("Parameters:", parameter_count)
print("Frozen:", True)

assert parameter_count == 11176512

print("✓ ResNet-18 backbone frozen")


# ================================================================
# 9. PREPROCESSING
# ================================================================

preprocess = weights.transforms()

print()
print("IMAGE PREPROCESSING")
print("-" * 100)
print("ImageNet ResNet-18 official transforms")
print("✓ Preprocessing locked")


# ================================================================
# 10. FEATURE CACHE
# ================================================================

FEATURE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

cache_file = CACHE_DIR / "BreaKHis_RESNET18_512_FEATURES_v1.npz"

print()
print("512-D FEATURE CACHE")
print("-" * 100)

# ---------------------------------------------------------------
# We intentionally do NOT overwrite an existing cache.
# ---------------------------------------------------------------

if cache_file.exists():

    print("Existing cache found:")
    print(cache_file)

    cache = np.load(
        cache_file,
        allow_pickle=False
    )

    cached_paths = cache["image_path"].astype(str)
    cached_features = cache["features"].astype(np.float32)

    assert cached_features.shape == (
        7909,
        FEATURE_DIM
    )

    assert len(cached_paths) == 7909

    assert len(set(cached_paths)) == 7909

    assert set(cached_paths) == reference_paths

    assert np.isfinite(cached_features).all()

    feature_paths = cached_paths
    features_512 = cached_features

    print("✓ Existing immutable 512-D cache verified")
    print("Shape:", features_512.shape)

else:

    print("No cache exists.")
    print("Extracting 512-D features once for all 7,909 images...")

    feature_list = []

    with torch.inference_mode():

        for start in range(
            0,
            len(physical_images),
            BATCH_SIZE
        ):

            batch_paths = physical_images[
                start:start + BATCH_SIZE
            ]

            tensors = []

            for image_path in batch_paths:

                image = Image.open(
                    image_path
                ).convert("RGB")

                tensor = preprocess(image)

                tensors.append(tensor)

            batch = torch.stack(
                tensors,
                dim=0
            ).to(device)

            output = feature_extractor(batch)

            output = output.flatten(1)

            assert output.shape[1] == FEATURE_DIM

            assert torch.isfinite(output).all()

            feature_list.append(
                output.cpu().numpy().astype(
                    np.float32
                )
            )

            processed = min(
                start + len(batch_paths),
                len(physical_images)
            )

            print(
                f"Processed {processed}/{len(physical_images)}",
                end="\r"
            )

    print()

    features_512 = np.concatenate(
        feature_list,
        axis=0
    )

    feature_paths = np.array(
        [
            str(p)
            for p in physical_images
        ],
        dtype=str
    )

    assert features_512.shape == (
        7909,
        FEATURE_DIM
    )

    assert len(feature_paths) == 7909

    assert np.isfinite(features_512).all()

    temporary_cache = (
        cache_file.with_suffix(".tmp.npz")
    )

    np.savez_compressed(
        temporary_cache,
        image_path=feature_paths,
        features=features_512,
    )

    temporary_cache.replace(cache_file)

    print()
    print("✓ 512-D feature cache created")
    print("Cache:", cache_file)
    print("Shape:", features_512.shape)


# ================================================================
# 11. FEATURE CACHE HASH
# ================================================================

cache_hash = sha256_file(cache_file)

print()
print("FEATURE CACHE PROVENANCE")
print("-" * 100)

print("Cache SHA256:", cache_hash)
print("Images:", len(feature_paths))
print("Dimensions:", features_512.shape)

assert features_512.shape == (7909, 512)

print("✓ 512-D feature cache verified")


# ================================================================
# 12. PATH → FEATURE INDEX
# ================================================================

feature_index = {
    str(path): index
    for index, path in enumerate(feature_paths)
}

assert len(feature_index) == 7909


# ================================================================
# 13. FOLD-SAFE PCA
# ================================================================

print()
print("FOLD-SAFE PCA")
print("-" * 100)

all_results = []

for fold in range(1, 6):

    df = manifests[fold]

    print()
    print(
        f"========== FOLD {fold:02d} =========="
    )

    fold_dir = PCA_DIR / f"fold_{fold:02d}"
    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_mask = (
        df["split"] == "train"
    )

    val_mask = (
        df["split"] == "val"
    )

    test_mask = (
        df["split"] == "test"
    )

    train_paths = df.loc[
        train_mask,
        "resolved_image_path"
    ].tolist()

    val_paths = df.loc[
        val_mask,
        "resolved_image_path"
    ].tolist()

    test_paths = df.loc[
        test_mask,
        "resolved_image_path"
    ].tolist()

    train_indices = [
        feature_index[p]
        for p in train_paths
    ]

    val_indices = [
        feature_index[p]
        for p in val_paths
    ]

    test_indices = [
        feature_index[p]
        for p in test_paths
    ]

    X_train = features_512[
        train_indices
    ]

    X_val = features_512[
        val_indices
    ]

    X_test = features_512[
        test_indices
    ]

    assert X_train.shape[1] == 512
    assert X_val.shape[1] == 512
    assert X_test.shape[1] == 512

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    # ------------------------------------------------------------
    # Each seed receives a separately fitted PCA object.
    # PCA fitting uses TRAIN ONLY.
    # ------------------------------------------------------------

    for seed in SEEDS:

        print()
        print(
            f"Fold {fold:02d} / Seed {seed}"
        )

        set_seed(seed)

        seed_dir = (
            fold_dir
            / f"seed_{seed}"
        )

        if seed_dir.exists():

            existing_files = list(
                seed_dir.iterdir()
            )

            if existing_files:

                raise FileExistsError(
                    "Refusing to overwrite existing "
                    f"ASUS-20B run artifacts: {seed_dir}"
                )

        seed_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        # --------------------------------------------------------
        # PCA FIT — TRAINING DATA ONLY
        # --------------------------------------------------------

        pca = PCA(
            n_components=PCA_DIM,
            svd_solver="full",
            random_state=seed,
        )

        pca.fit(X_train)

        # --------------------------------------------------------
        # TRANSFORM ONLY
        # --------------------------------------------------------

        X_train_6 = pca.transform(
            X_train
        ).astype(np.float32)

        X_val_6 = pca.transform(
            X_val
        ).astype(np.float32)

        X_test_6 = pca.transform(
            X_test
        ).astype(np.float32)

        assert X_train_6.shape == (
            len(X_train),
            PCA_DIM
        )

        assert X_val_6.shape == (
            len(X_val),
            PCA_DIM
        )

        assert X_test_6.shape == (
            len(X_test),
            PCA_DIM
        )

        assert np.isfinite(X_train_6).all()
        assert np.isfinite(X_val_6).all()
        assert np.isfinite(X_test_6).all()

        # --------------------------------------------------------
        # SAVE PCA
        # --------------------------------------------------------

        pca_file = seed_dir / "PCA_6D.npy"

        np.save(
            pca_file,
            pca.components_.astype(np.float32)
        )

        # --------------------------------------------------------
        # SAVE TRANSFORMED DATA
        # --------------------------------------------------------

        output_file = (
            seed_dir
            / "BreaKHis_FOLD_FEATURES_v1.npz"
        )

        np.savez_compressed(
            output_file,

            train_image_path=np.array(
                train_paths,
                dtype=str
            ),

            train_patient_id=np.array(
                df.loc[
                    train_mask,
                    "patient_id"
                ].astype(str),
                dtype=str
            ),

            train_case_id=np.array(
                df.loc[
                    train_mask,
                    "case_id"
                ].astype(str),
                dtype=str
            ),

            train_label=np.array(
                df.loc[
                    train_mask,
                    "label_numeric"
                ],
                dtype=np.int64
            ),

            train_features_6=X_train_6,

            val_image_path=np.array(
                val_paths,
                dtype=str
            ),

            val_patient_id=np.array(
                df.loc[
                    val_mask,
                    "patient_id"
                ].astype(str),
                dtype=str
            ),

            val_case_id=np.array(
                df.loc[
                    val_mask,
                    "case_id"
                ].astype(str),
                dtype=str
            ),

            val_label=np.array(
                df.loc[
                    val_mask,
                    "label_numeric"
                ],
                dtype=np.int64
            ),

            val_features_6=X_val_6,

            test_image_path=np.array(
                test_paths,
                dtype=str
            ),

            test_patient_id=np.array(
                df.loc[
                    test_mask,
                    "patient_id"
                ].astype(str),
                dtype=str
            ),

            test_case_id=np.array(
                df.loc[
                    test_mask,
                    "case_id"
                ].astype(str),
                dtype=str
            ),

            test_label=np.array(
                df.loc[
                    test_mask,
                    "label_numeric"
                ],
                dtype=np.int64
            ),

            test_features_6=X_test_6,
        )

        # --------------------------------------------------------
        # PCA PROVENANCE
        # --------------------------------------------------------

        metadata = {

            "step": "ASUS-20B",

            "status": "VERIFIED",

            "project":
                "quantum-epistemic-reliability",

            "dataset":
                "BreaKHis",

            "fold": fold,

            "seed": seed,

            "git": {
                "branch": git_branch,
                "commit": git_commit,
            },

            "manifest": {
                "file":
                    f"manifests/"
                    f"BreaKHis_FOLD_{fold:02d}_"
                    f"VERIFIED_v1.csv",

                "sha256":
                    sha256_file(
                        PROJECT
                        / "manifests"
                        / f"BreaKHis_FOLD_{fold:02d}_"
                          f"VERIFIED_v1.csv"
                    )
            },

            "backbone": {
                "name": "ResNet-18",
                "weights": str(weights),
                "frozen": True,
                "feature_dimension": 512,
                "parameter_count":
                    parameter_count,
            },

            "pca": {
                "input_dimension": 512,
                "output_dimension": 6,
                "solver": "full",
                "random_state": seed,
                "fit_scope":
                    "training_data_only",
                "validation_refit": False,
                "test_refit": False,
            },

            "counts": {
                "train": int(
                    len(X_train)
                ),
                "validation": int(
                    len(X_val)
                ),
                "test": int(
                    len(X_test)
                )
            },

            "explained_variance_ratio":
                pca.explained_variance_ratio_
                .astype(float)
                .tolist(),

            "explained_variance_total":
                float(
                    pca.explained_variance_ratio_.sum()
                ),

            "artifacts": {
                "feature_cache":
                    str(
                        cache_file.relative_to(
                            PROJECT
                        )
                    ),

                "feature_cache_sha256":
                    cache_hash,

                "pca_components":
                    str(
                        pca_file.relative_to(
                            PROJECT
                        )
                    ),

                "feature_output":
                    str(
                        output_file.relative_to(
                            PROJECT
                        )
                    ),
            },

            "data_integrity": {
                "frozen_manifest": True,
                "new_random_split": False,
                "source_dataset_modified": False,
                "test_used_for_pca": False,
                "validation_used_for_pca": False,
                "test_used_for_model_selection": False,
            },

            "execution_separation": {
                "asus_breakhis":
                    "independent",
                "kaggle_cbis_ddsm":
                    "independent",
                "historical_project":
                    "reference_only",
            },

            "training": {
                "model_training_performed":
                    False
            },

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        metadata_file = (
            seed_dir
            / "PCA_PROVENANCE.json"
        )

        metadata_file.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )

        # --------------------------------------------------------
        # RELOAD VERIFICATION
        # --------------------------------------------------------

        saved = np.load(
            output_file,
            allow_pickle=False
        )

        assert saved["train_features_6"].shape == (
            len(X_train),
            6
        )

        assert saved["val_features_6"].shape == (
            len(X_val),
            6
        )

        assert saved["test_features_6"].shape == (
            len(X_test),
            6
        )

        assert len(
            saved["train_image_path"]
        ) == len(X_train)

        assert len(
            saved["val_image_path"]
        ) == len(X_val)

        assert len(
            saved["test_image_path"]
        ) == len(X_test)

        output_hash = sha256_file(
            output_file
        )

        provenance_hash = sha256_file(
            metadata_file
        )

        all_results.append(
            {
                "fold": fold,
                "seed": seed,
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
                "feature_output":
                    str(
                        output_file.relative_to(
                            PROJECT
                        )
                    ),
                "feature_output_sha256":
                    output_hash,
                "provenance":
                    str(
                        metadata_file.relative_to(
                            PROJECT
                        )
                    ),
                "provenance_sha256":
                    provenance_hash,
                "explained_variance":
                    float(
                        pca.explained_variance_ratio_
                        .sum()
                    ),
            }
        )

        print(
            "✓ PCA fitted on TRAIN ONLY"
        )

        print(
            "✓ Validation transformed only"
        )

        print(
            "✓ Test transformed only"
        )

        print(
            "✓ 6-D feature artifact verified"
        )

        print(
            "Output SHA256:",
            output_hash
        )


# ================================================================
# 14. GLOBAL ASUS-20B AUDIT RECORD
# ================================================================

audit = {

    "step": "ASUS-20B",

    "status":
        "VERIFIED_FEATURE_EXTRACTION_COMPLETE",

    "project":
        "quantum-epistemic-reliability",

    "dataset":
        "BreaKHis",

    "physical_images":
        7909,

    "folds":
        5,

    "seeds":
        SEEDS,

    "backbone": {
        "name": "ResNet-18",
        "weights": str(weights),
        "frozen": True,
        "feature_dimension": 512,
    },

    "pca": {
        "components": 6,
        "fit_scope":
            "training_data_only",
        "validation_refit":
            False,
        "test_refit":
            False,
    },

    "feature_cache": {
        "path":
            str(
                cache_file.relative_to(
                    PROJECT
                )
            ),
        "sha256":
            cache_hash,
        "shape":
            list(features_512.shape),
    },

    "runs": all_results,

    "run_count":
        len(all_results),

    "expected_run_count":
        15,

    "data_integrity": {
        "frozen_manifests":
            True,
        "new_random_split":
            False,
        "source_dataset_modified":
            False,
        "test_used_for_pca":
            False,
        "validation_used_for_pca":
            False,
    },

    "execution_separation": {
        "asus_breakhis":
            "independent",
        "kaggle_cbis_ddsm":
            "independent",
        "historical_project":
            "reference_only",
    },

    "training": {
        "performed":
            False
    },

    "git": {
        "branch":
            git_branch,
        "commit":
            git_commit,
    },

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}


assert len(all_results) == 15

AUDIT_FILE = (
    METADATA_DIR
    / "ASUS-20B_FEATURE_EXTRACTION_AUDIT_v1.json"
)

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_FILE.write_text(
    json.dumps(
        audit,
        indent=2
    ),
    encoding="utf-8"
)


# ================================================================
# 15. FINAL VERIFICATION
# ================================================================

print()
print("=" * 100)
print("ASUS-20B COMPLETE — VERIFIED")
print("=" * 100)

print("✓ 7,909 physical images verified")
print("✓ Five frozen manifests verified")
print("✓ Same image universe verified across folds")
print("✓ Frozen ImageNet ResNet-18 verified")
print("✓ 512-D representation generated/verified")
print("✓ Single immutable 512-D cache created/verified")
print("✓ Five fold-specific PCA pipelines executed")
print("✓ Three seeds per fold executed")
print("✓ 15 fold/seed PCA artifacts created")
print("✓ PCA fitted on TRAINING DATA ONLY")
print("✓ Validation used for transformation only")
print("✓ Test used for transformation only")
print("✓ No new random split created")
print("✓ No source dataset modified")
print("✓ No model training performed")
print("✓ Kaggle CBIS-DDSM untouched")
print("✓ Historical project untouched")
print("✓ Manifest SHA256 recorded")
print("✓ Feature-cache SHA256 recorded")
print("✓ Feature-output SHA256 recorded")
print("✓ Git commit recorded")
print("✓ PCA provenance recorded")

print()
print("512-D CACHE:")
print(
    cache_file.relative_to(PROJECT)
)

print()
print("AUDIT:")
print(
    AUDIT_FILE
)

print()
print("TOTAL FOLD/SEED ARTIFACTS:",
      len(all_results))

print()
print("TRAINING PERFORMED: FALSE")
print("DATASET MODIFIED: FALSE")
print("KAGGLE CBIS-DDSM TOUCHED: FALSE")
print("HISTORICAL PROJECT USED FOR EXECUTION: FALSE")

print("=" * 100)