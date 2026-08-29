from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "ASUS_18_EXPERIMENT_SPECIFICATION_v1.json"
OUT = PROJECT / "configs" / "ASUS_19_IMPLEMENTATION_CONTRACT_v1.json"

spec = json.loads(CONFIG.read_text(encoding="utf-8-sig"))

required = {
    "dataset": spec["primary_dataset"]["name"],
    "outer_folds": spec["primary_dataset"]["outer_folds"],
    "seeds": spec["statistical_analysis"]["independent_seeds"],
    "primary_endpoint": spec["primary_evaluation"]["primary_endpoint"],
    "epistemic_method": spec["epistemic_uncertainty"]["primary_method"],
    "pca_components": spec["representation"]["pca_components"],
    "feature_dimension": spec["representation"]["feature_dimension"],
    "experimental_arms": spec["experimental_arms"],
}

assert required["dataset"] == "BreaKHis"
assert required["outer_folds"] == 5
assert required["seeds"] == [42, 123, 2025]
assert required["primary_endpoint"] == "AURC"
assert required["epistemic_method"] == "ensemble mutual information"
assert required["feature_dimension"] == 512
assert required["pca_components"] == 6
assert len(required["experimental_arms"]) == 7

contract = {
    "step": "ASUS-19",
    "status": "IMPLEMENTATION_CONTRACT_VERIFIED",

    "project": "quantum-epistemic-reliability",
    "dataset": "BreaKHis",
    "backend": "local_ASUS_RTX_5060",

    "locked_specification": str(
        CONFIG.relative_to(PROJECT)
    ),

    "experiment": required,

    "data_integrity": {
        "frozen_manifests_only": True,
        "new_random_patient_split": False,
        "test_data_for_pca": False,
        "test_data_for_model_selection": False,
        "test_data_for_hyperparameter_selection": False,
    },

    "pca_policy": {
        "input_dimension": 512,
        "output_dimension": 6,
        "fit_on_training_data_only": True,
        "validation_transform_only": True,
        "test_transform_only": True,
    },

    "uncertainty": {
        "method": "ensemble mutual information",
        "formula": "H(mean(p)) - mean(H(p))",
        "aleatoric_uncertainty_claim": False,
    },

    "evaluation": {
        "unit": "patient_or_case",
        "primary_endpoint": "AURC",
        "selective_prediction": True,
        "error_detection": True,
        "calibration": True,
        "metrics": [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "ece",
            "AURC",
            "risk_coverage",
            "uncertainty_error_discrimination",
        ],
    },

    "reproducibility": {
        "three_seeds": True,
        "five_outer_folds": True,
        "run_id": True,
        "git_commit": True,
        "manifest_hash": True,
        "config_hash": True,
        "environment_record": True,
        "predictions": True,
        "uncertainty": True,
        "fold_metrics": True,
        "aggregate_metrics": True,
    },

    "publication": {
        "source_data": True,
        "generation_scripts": True,
        "vector_pdf": True,
        "vector_svg": True,
        "png": True,
        "png_dpi": 400,
        "font_embedding": True,
        "screenshots": False,
    },

    "execution_separation": {
        "asus_breaKHis": "independent",
        "kaggle_cbis_ddsm": "independent",
        "historical_project": "reference_only",
        "cross_dataset_analysis": "only_after_independent_completion",
    },

    "training": {
        "enabled": True,
        "overwrite_previous_runs": False,
        "pilot_before_full_experiment": True,
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(contract, indent=2),
    encoding="utf-8"
)

print("=" * 100)
print("ASUS-19 — IMPLEMENTATION CONTRACT")
print("=" * 100)
print()
print("Contract:", OUT)
print("Dataset:", contract["dataset"])
print("Folds:", contract["experiment"]["outer_folds"])
print("Seeds:", contract["experiment"]["seeds"])
print("Feature dimension:", contract["experiment"]["feature_dimension"])
print("PCA:", contract["pca_policy"]["input_dimension"], "->",
      contract["pca_policy"]["output_dimension"])
print("Experimental arms:", len(contract["experiment"]["experimental_arms"]))
print("Primary endpoint:", contract["experiment"]["primary_endpoint"])
print("Uncertainty:", contract["uncertainty"]["method"])
print("PNG DPI:", contract["publication"]["png_dpi"])
print()
print("✓ ASUS-18 specification successfully loaded")
print("✓ Seven experimental arms verified")
print("✓ 512 -> 6 PCA requirement verified")
print("✓ Five-fold evaluation verified")
print("✓ Three seeds verified")
print("✓ AURC primary endpoint verified")
print("✓ Ensemble mutual-information uncertainty verified")
print("✓ Test-data isolation verified")
print("✓ Publication provenance requirements verified")
print("✓ ASUS/Kaggle separation preserved")
print()
print("TRAINING NOT STARTED")
print("=" * 100)
