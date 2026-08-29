from pathlib import Path
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]

CONFIG = PROJECT / "configs" / "ASUS_12_EXECUTION_CONTRACT_v1.json"
FREEZE = PROJECT / "manifests" / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"
ENV_LOCK = PROJECT / "metadata" / "ASUS-14_ENVIRONMENT_LOCK_v1.json"

OUT_DIR = PROJECT / "experiments" / "schemas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_OUT = PROJECT / "scripts" / "ASUS_16_DETERMINISM.py"


print("=" * 100)
print("ASUS-16 — DETERMINISTIC EXPERIMENT CONTROL VERIFICATION")
print("=" * 100)


# ------------------------------------------------------------------
# 1. Required project artifacts
# ------------------------------------------------------------------

for p in [CONFIG, FREEZE, ENV_LOCK]:
    assert p.exists(), f"Required artifact missing: {p}"

print("\nREQUIRED ARTIFACTS")
print("-" * 100)

print("✓ Execution contract:", CONFIG.relative_to(PROJECT))
print("✓ BreaKHis freeze:", FREEZE.relative_to(PROJECT))
print("✓ Environment lock:", ENV_LOCK.relative_to(PROJECT))


# ------------------------------------------------------------------
# 2. SHA256 helper
# ------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


config_hash = sha256_file(CONFIG)
freeze_hash = sha256_file(FREEZE)
environment_hash = sha256_file(ENV_LOCK)

print("\nARTIFACT HASHES")
print("-" * 100)

print("Execution contract SHA256:", config_hash)
print("Freeze record SHA256:", freeze_hash)
print("Environment lock SHA256:", environment_hash)


# ------------------------------------------------------------------
# 3. Git provenance
# ------------------------------------------------------------------

def git(cmd):
    return subprocess.check_output(
        ["git"] + cmd,
        cwd=PROJECT,
        text=True
    ).strip()


git_commit = git(["rev-parse", "HEAD"])
git_branch = git(["branch", "--show-current"])

status = subprocess.check_output(
    ["git", "status", "--porcelain"],
    cwd=PROJECT,
    text=True
).strip()

print("\nGIT PROVENANCE")
print("-" * 100)

print("Branch:", git_branch)
print("Commit:", git_commit)
print("Working tree clean:", status == "")

assert git_branch == "main"
assert git_commit
assert status == "", (
    "Git working tree must be clean before deterministic "
    "experiment execution."
)

print("✓ Git provenance valid")
print("✓ Working tree clean")


# ------------------------------------------------------------------
# 4. Deterministic seed function
# ------------------------------------------------------------------

SEED = 20260829


def set_global_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_global_seed(SEED)


# ------------------------------------------------------------------
# 5. Deterministic PyTorch configuration
# ------------------------------------------------------------------

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
    deterministic_algorithms = True
except Exception:
    deterministic_algorithms = False


print("\nDETERMINISM CONFIGURATION")
print("-" * 100)

print("Seed:", SEED)
print(
    "CUDA available:",
    torch.cuda.is_available()
)
print(
    "cuDNN deterministic:",
    torch.backends.cudnn.deterministic
)
print(
    "cuDNN benchmark:",
    torch.backends.cudnn.benchmark
)
print(
    "Deterministic algorithms:",
    deterministic_algorithms
)

assert torch.backends.cudnn.deterministic is True
assert torch.backends.cudnn.benchmark is False
assert deterministic_algorithms is True

print("✓ PyTorch deterministic configuration verified")


# ------------------------------------------------------------------
# 6. Numerical reproducibility test
# ------------------------------------------------------------------

set_global_seed(SEED)

cpu_a = torch.rand(32)

set_global_seed(SEED)

cpu_b = torch.rand(32)

assert torch.equal(cpu_a, cpu_b)

print("\nCPU REPRODUCIBILITY")
print("-" * 100)

print("Tensor length:", len(cpu_a))
print("✓ Identical CPU random tensors reproduced")


# ------------------------------------------------------------------
# 7. CUDA reproducibility test
# ------------------------------------------------------------------

cuda_test = False

if torch.cuda.is_available():

    device = torch.device("cuda:0")

    set_global_seed(SEED)

    cuda_a = torch.rand(
        32,
        device=device
    )

    set_global_seed(SEED)

    cuda_b = torch.rand(
        32,
        device=device
    )

    assert torch.equal(cuda_a, cuda_b)

    cuda_test = True

    print("\nCUDA REPRODUCIBILITY")
    print("-" * 100)
    print("GPU:", torch.cuda.get_device_name(0))
    print("✓ Identical CUDA random tensors reproduced")

else:

    print("\nCUDA REPRODUCIBILITY")
    print("-" * 100)
    print("CUDA unavailable; CUDA reproducibility test skipped")


# ------------------------------------------------------------------
# 8. Run identity
# ------------------------------------------------------------------

timestamp = datetime.now(timezone.utc)

run_id = (
    f"asus16_breakhis_seed{SEED}_"
    f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
)

print("\nRUN ID")
print("-" * 100)
print(run_id)


# ------------------------------------------------------------------
# 9. Run directory policy
# ------------------------------------------------------------------

run_root = PROJECT / "experiments" / "runs" / run_id

assert not run_root.exists(), (
    f"Run directory already exists: {run_root}"
)

run_root.mkdir(
    parents=True,
    exist_ok=False
)

print("✓ New immutable run directory created:")
print(run_root.relative_to(PROJECT))


# ------------------------------------------------------------------
# 10. Run metadata
# ------------------------------------------------------------------

metadata = {
    "step": "ASUS-16",
    "status": "VERIFIED",

    "run": {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "dataset": "BreaKHis",
        "backend": "local_ASUS_RTX_5060",
        "seed": SEED,
        "fold": None,
    },

    "git": {
        "branch": git_branch,
        "commit": git_commit,
        "working_tree_clean": True,
    },

    "artifacts": {
        "execution_contract": str(
            CONFIG.relative_to(PROJECT)
        ),
        "execution_contract_sha256": config_hash,

        "freeze_record": str(
            FREEZE.relative_to(PROJECT)
        ),
        "freeze_record_sha256": freeze_hash,

        "environment_lock": str(
            ENV_LOCK.relative_to(PROJECT)
        ),
        "environment_lock_sha256": environment_hash,
    },

    "determinism": {
        "pythonhashseed": str(SEED),
        "python_random": True,
        "numpy_random": True,
        "torch_random": True,
        "cuda_random": torch.cuda.is_available(),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": deterministic_algorithms,
        "cpu_reproducibility_test": True,
        "cuda_reproducibility_test": cuda_test,
    },

    "execution_separation": {
        "historical_project_execution": False,
        "kaggle_cbisd_dsm_execution": "independent",
        "cross_dataset_analysis": "not_started",
    },

    "training": {
        "performed": False,
    },

    "data": {
        "modified": False,
    },
}


metadata_file = run_root / "RUN_METADATA.json"

metadata_file.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------------
# 11. Final verification
# ------------------------------------------------------------------

assert metadata_file.exists()

saved = json.loads(
    metadata_file.read_text(
        encoding="utf-8"
    )
)

assert saved["step"] == "ASUS-16"
assert saved["run"]["seed"] == SEED
assert saved["git"]["commit"] == git_commit
assert saved["artifacts"]["freeze_record_sha256"] == freeze_hash
assert saved["determinism"]["cpu_reproducibility_test"] is True

if torch.cuda.is_available():
    assert saved["determinism"]["cuda_reproducibility_test"] is True


print("\n" + "=" * 100)
print("ASUS-16 COMPLETE — VERIFIED")
print("=" * 100)

print("✓ Global seed fixed:", SEED)
print("✓ Python reproducibility configured")
print("✓ NumPy reproducibility configured")
print("✓ PyTorch reproducibility configured")
print("✓ CUDA reproducibility configured")
print("✓ cuDNN deterministic mode enabled")
print("✓ cuDNN benchmark disabled")
print("✓ Deterministic algorithms enabled")
print("✓ CPU reproducibility test passed")

if cuda_test:
    print("✓ CUDA reproducibility test passed")

print("✓ Git commit captured")
print("✓ Working tree was clean")
print("✓ Config SHA256 captured")
print("✓ Dataset freeze SHA256 captured")
print("✓ Environment SHA256 captured")
print("✓ Unique run ID generated")
print("✓ Immutable run directory created")
print("✓ Run metadata written:")
print(metadata_file)

print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("NO HISTORICAL PROJECT USED FOR EXECUTION")

print("=" * 100)
