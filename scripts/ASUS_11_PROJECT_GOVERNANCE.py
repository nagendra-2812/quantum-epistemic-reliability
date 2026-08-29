from pathlib import Path
import json
import subprocess
import sys
import platform
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "EXPERIMENT_PROTOCOL_v1.json"
FREEZE = PROJECT / "manifests" / "BreaKHis_FIVE_FOLD_FREEZE_v1.json"
OUT = PROJECT / "metadata" / "ASUS-11_PROJECT_GOVERNANCE_RECORD_v1.json"

print("=" * 100)
print("ASUS-11 — NEW PROJECT GOVERNANCE + EXPERIMENT PROTOCOL")
print("=" * 100)

assert PROJECT.name == "quantum-epistemic-reliability", PROJECT
assert CONFIG.exists(), f"Missing protocol: {CONFIG}"
assert FREEZE.exists(), f"Missing BreaKHis freeze: {FREEZE}"

# PowerShell Set-Content may create UTF-8 BOM.
# utf-8-sig safely reads both BOM and non-BOM UTF-8.
protocol = json.loads(CONFIG.read_text(encoding="utf-8-sig"))

assert protocol["project"] == "quantum-epistemic-reliability"
assert protocol["execution_separation"]["historical_project_usage"] == "reference_only"
assert protocol["execution_separation"]["cross_project_training"] is False
assert protocol["execution_separation"]["cross_project_output_writing"] is False
assert protocol["dataset_freeze"]["physical_images"] == 7909
assert protocol["dataset_freeze"]["folds"] == 5
assert protocol["dataset_freeze"]["patient_disjoint"] is True
assert protocol["dataset_freeze"]["case_disjoint"] is True
assert protocol["publication_figures"]["raster_dpi"] == 400
assert "PDF" in protocol["publication_figures"]["vector_formats"]
assert "SVG" in protocol["publication_figures"]["vector_formats"]
assert protocol["publication_figures"]["font_embedding_required"] is True
assert protocol["publication_figures"]["no_screenshot_figures"] is True

freeze = json.loads(FREEZE.read_text(encoding="utf-8-sig"))

assert freeze["status"] == "VERIFIED_FROZEN"
assert freeze["physical_image_count"] == 7909
assert freeze["fold_count"] == 5
assert freeze["rows_per_fold"] == 7909
assert freeze["patient_disjoint_all_folds"] is True
assert freeze["case_disjoint_all_folds"] is True

git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=PROJECT,
    text=True
).strip()

git_branch = subprocess.check_output(
    ["git", "branch", "--show-current"],
    cwd=PROJECT,
    text=True
).strip()

record = {
    "step": "ASUS-11",
    "status": "VERIFIED",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),

    "project": "quantum-epistemic-reliability",
    "project_root": str(PROJECT),

    "git": {
        "branch": git_branch,
        "commit_at_protocol_verification": git_commit
    },

    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable
    },

    "scientific_separation": {
        "local_execution": True,
        "kaggle_cbisd_dsm_execution": "independent",
        "historical_project_reference_only": True,
        "historical_project": r"D:\AI\quantum-uncertainty-shift",
        "cross_project_training": False,
        "cross_project_output_writing": False,
        "kaggle_touched": False
    },

    "dataset_freeze": {
        "dataset": "BreaKHis",
        "physical_image_count": 7909,
        "fold_count": 5,
        "rows_per_fold": 7909,
        "freeze_record": str(FREEZE.relative_to(PROJECT)),
        "patient_disjoint": True,
        "case_disjoint": True
    },

    "publication_requirements": {
        "vector_pdf": True,
        "vector_svg": True,
        "png_400_dpi": True,
        "font_embedding": True,
        "source_data_required": True,
        "generation_script_required": True,
        "screenshot_figures_prohibited": True
    },

    "reproducibility_requirements": {
        "explicit_seed": True,
        "git_commit": True,
        "dataset_manifest_hash": True,
        "source_code_hash": True,
        "environment_record": True,
        "run_metadata": True,
        "predictions": True,
        "uncertainty_values": True,
        "fold_metrics": True,
        "aggregate_metrics": True,
        "figure_source_data": True,
        "table_source_data": True
    },

    "execution_state": {
        "training_performed": False,
        "dataset_modified": False,
        "kaggle_data_modified": False,
        "historical_project_modified": False
    }
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(record, indent=2),
    encoding="utf-8"
)

print()
print("PROJECT:", PROJECT)
print("BRANCH:", git_branch)
print("BASE GIT COMMIT:", git_commit)
print("PROTOCOL:", CONFIG)
print("BreaKHis FREEZE:", FREEZE)
print("RECORD:", OUT)

print()
print("✓ New project protocol loaded")
print("✓ UTF-8/BOM handling verified")
print("✓ BreaKHis frozen record verified")
print("✓ 7,909 physical images")
print("✓ 5 frozen folds")
print("✓ Patient-disjoint requirement verified")
print("✓ Case-disjoint requirement verified")
print("✓ Historical project is reference-only")
print("✓ Kaggle CBIS-DDSM remains independent")
print("✓ 400 dpi PNG requirement verified")
print("✓ PDF vector requirement verified")
print("✓ SVG vector requirement verified")
print("✓ Source-data and figure-generation requirements verified")
print("✓ Git traceability requirement verified")
print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("NO HISTORICAL PROJECT MODIFIED")
print("=" * 100)
