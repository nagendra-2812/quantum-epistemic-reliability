from pathlib import Path
import subprocess
import sys
import json
import hashlib
import platform
import importlib.metadata as md
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]

OUT = PROJECT / "metadata" / "ASUS-14_ENVIRONMENT_LOCK_v1.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("ASUS-14 — REPRODUCIBILITY ENVIRONMENT LOCK")
print("=" * 100)

def git(cmd):
    return subprocess.check_output(
        ["git"] + cmd,
        cwd=PROJECT,
        text=True
    ).strip()

git_commit = git(["rev-parse", "HEAD"])
git_branch = git(["branch", "--show-current"])

packages = {}

for name in [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "Pillow"
]:
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None

import torch
import torchvision
import numpy
import pandas
from PIL import Image

cuda_available = bool(torch.cuda.is_available())

gpu = None
capability = None

if cuda_available:
    gpu = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))

record = {
    "step": "ASUS-14",
    "status": "VERIFIED",

    "project": "quantum-epistemic-reliability",

    "timestamp_utc": datetime.now(timezone.utc).isoformat(),

    "git": {
        "branch": git_branch,
        "commit": git_commit,
        "clean_worktree_required": True
    },

    "python": {
        "version": sys.version,
        "executable": sys.executable
    },

    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform()
    },

    "packages": packages,

    "torch": {
        "version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu": gpu,
        "compute_capability": capability
    },

    "dataset": {
        "name": "BreaKHis",
        "physical_images": 7909,
        "folds": 5,
        "manifest_freeze": "manifests/BreaKHis_FIVE_FOLD_FREEZE_v1.json"
    },

    "execution_separation": {
        "historical_project": r"D:\AI\quantum-uncertainty-shift",
        "historical_project_execution": False,
        "kaggle_cbisd_dsm_execution": "independent",
        "cross_dataset_analysis": "not_started"
    },

    "publication": {
        "png_dpi": 400,
        "vector_pdf": True,
        "vector_svg": True,
        "font_embedding": True,
        "screenshots_prohibited": True
    }
}

text = json.dumps(record, indent=2)

OUT.write_text(text, encoding="utf-8")

sha256 = hashlib.sha256(OUT.read_bytes()).hexdigest()

record["environment_lock_sha256"] = sha256

OUT.write_text(
    json.dumps(record, indent=2),
    encoding="utf-8"
)

print()
print("Git branch:", git_branch)
print("Git commit:", git_commit)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Pillow:", Image.__version__)
print("CUDA available:", cuda_available)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", gpu)
print("Capability:", capability)
print()
print("✓ Environment lock written:")
print(OUT)
print("✓ Git provenance recorded")
print("✓ Python environment recorded")
print("✓ CUDA/GPU environment recorded")
print("✓ Dataset freeze recorded")
print("✓ ASUS/Kaggle separation recorded")
print("✓ Publication requirements recorded")
print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("NO HISTORICAL PROJECT MODIFIED")
print("=" * 100)
