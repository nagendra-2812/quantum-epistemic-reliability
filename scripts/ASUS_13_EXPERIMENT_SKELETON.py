from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]

dirs = [
    "experiments",
    "experiments/runs",
    "experiments/schemas",
    "metrics",
    "metrics/fold",
    "metrics/aggregate",
    "predictions",
    "predictions/fold",
    "figures",
    "figures/source_data",
    "figures/pdf",
    "figures/svg",
    "figures/png_400dpi",
    "tables",
    "tables/source_data",
    "run_registry",
    "source_data",
    "metadata",
    "logs",
]

print("=" * 100)
print("ASUS-13 — REPRODUCIBLE EXPERIMENT SKELETON")
print("=" * 100)

for d in dirs:
    p = PROJECT / d
    p.mkdir(parents=True, exist_ok=True)
    print("✓", p.relative_to(PROJECT))

schema = {
    "schema_version": "ASUS-13-v1",
    "run_identity": {
        "run_id": "required",
        "git_commit": "required",
        "timestamp_utc": "required",
        "dataset": "required",
        "fold": "required",
        "seed": "required"
    },
    "data": {
        "manifest": "required",
        "manifest_sha256": "required",
        "train_count": "required",
        "validation_count": "required",
        "test_count": "required"
    },
    "model": {
        "model_name": "required",
        "model_version": "required",
        "parameter_count": "required",
        "checkpoint": "required"
    },
    "predictions": {
        "sample_id": "required",
        "true_label": "required",
        "predicted_label": "required",
        "probability": "required"
    },
    "uncertainty": {
        "epistemic_uncertainty": "required",
        "uncertainty_method": "required"
    },
    "metrics": {
        "accuracy": "required",
        "balanced_accuracy": "required",
        "precision": "required",
        "recall": "required",
        "f1": "required",
        "roc_auc": "required",
        "pr_auc": "required",
        "ece": "required",
        "selective_metrics": "required"
    },
    "publication": {
        "source_data": "required",
        "generation_script": "required",
        "pdf": "required",
        "svg": "required",
        "png_400dpi": "required"
    }
}

schema_file = PROJECT / "experiments" / "schemas" / "EXPERIMENT_RESULT_SCHEMA_v1.json"
schema_file.write_text(
    json.dumps(schema, indent=2),
    encoding="utf-8"
)

print()
print("Schema:", schema_file)
print()
print("✓ Prediction provenance reserved")
print("✓ Epistemic uncertainty provenance reserved")
print("✓ Fold metrics reserved")
print("✓ Aggregate metrics reserved")
print("✓ Figure source-data provenance reserved")
print("✓ Table source-data provenance reserved")
print("✓ PDF/SVG/400-DPI PNG outputs reserved")
print("✓ Run/Git/seed provenance reserved")
print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("NO HISTORICAL PROJECT MODIFIED")
print("=" * 100)
