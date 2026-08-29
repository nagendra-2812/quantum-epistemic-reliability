from pathlib import Path
import json
from datetime import datetime, timezone

PROJECT = Path(r"D:\AI\quantum-epistemic-reliability")

OUT = PROJECT / "configs" / "ASUS_18_EXPERIMENT_SPECIFICATION_v1.json"

spec = {
    "stage": "ASUS-18",
    "status": "FROZEN",
    "project": "quantum-epistemic-reliability",

    "primary_dataset": {
        "name": "BreaKHis",
        "physical_images": 7909,
        "outer_folds": 5,
        "manifest": "manifests/BreaKHis_FIVE_FOLD_FREEZE_v1.json",
        "new_random_patient_split": False,
    },

    "representation": {
        "backbone": "ImageNet-pretrained ResNet-18",
        "backbone_status": "frozen",
        "feature_dimension": 512,
        "dimensionality_reduction": "PCA",
        "pca_components": 6,
        "pca_fit_scope": "inner-training data only",
        "pca_refit_on_validation": False,
        "pca_refit_on_test": False,
        "shared_representation": True,
    },

    "inner_split": {
        "required": True,
        "group_disjoint": True,
        "purpose": [
            "model training",
            "empirical-Fisher estimation",
            "early stopping",
            "hyperparameter selection",
        ],
        "training_fraction": 0.80,
        "validation_fraction": 0.20,
        "group_unit": "patient_or_case",
    },

    "experimental_arms": [
        {
            "name": "softmax",
            "architecture": "Linear(6->1)+sigmoid",
            "trainable_parameters": 7,
        },
        {
            "name": "temperature_scaled",
            "architecture": "Softmax+post-hoc temperature",
            "trainable_parameters": 8,
        },
        {
            "name": "mc_dropout",
            "architecture": "Linear(6->16)+Dropout(0.2)+Linear(16->1)",
            "passes": 30,
            "trainable_parameters": 129,
        },
        {
            "name": "deep_ensemble",
            "architecture": "5 independent Softmax models",
            "ensemble_members": 5,
            "trainable_parameters": 35,
        },
        {
            "name": "deterministic_vqc",
            "architecture": "6-qubit VQC",
            "qubits": 6,
            "ansatz_layers": 2,
            "encoding": "RY+RZ",
            "entanglement": "ring CNOT",
            "trainable_parameters": 24,
            "encoding_parameters_trainable": False,
            "simulation": "exact statevector",
        },
        {
            "name": "laplace_vqc",
            "architecture": "Deterministic VQC + Laplace perturbation",
            "trainable_parameters": 24,
        },
        {
            "name": "laplace_mlp",
            "architecture": "Linear(6->3)+tanh+Linear(3->1)+Laplace perturbation",
            "trainable_parameters": 25,
        },
    ],

    "quantum_input": {
        "dimensions": 6,
        "encoding": "bounded angle encoding",
        "encoding_parameters_trainable": False,
    },

    "epistemic_uncertainty": {
        "primary_method": "ensemble mutual information",
        "formula": "H(mean(p)) - mean(H(p))",
        "interpretation": "epistemic uncertainty",
        "aleatoric_uncertainty_claim": False,
    },

    "primary_evaluation": {
        "unit": "patient_or_case",
        "primary_endpoint": "AURC",
        "selective_prediction": True,
        "error_detection": True,
        "calibration": True,
    },

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

    "statistical_analysis": {
        "outer_folds": 5,
        "independent_seeds": [42, 123, 2025],
        "patient_or_case_clustered_analysis": True,
        "multiple_comparison_correction": "Benjamini-Hochberg",
        "primary_comparison": "controlled classical-versus-quantum reliability comparison",
    },

    "data_integrity": {
        "frozen_manifests_must_be_used": True,
        "manifest_modification": False,
        "source_dataset_modification": False,
        "test_data_used_for_model_selection": False,
        "test_data_used_for_pca_fitting": False,
        "test_data_used_for_hyperparameter_selection": False,
    },

    "reproducibility": {
        "run_id_required": True,
        "git_commit_required": True,
        "seed_required": True,
        "manifest_hash_required": True,
        "config_hash_required": True,
        "environment_record_required": True,
        "predictions_required": True,
        "uncertainty_values_required": True,
        "fold_metrics_required": True,
        "aggregate_metrics_required": True,
    },

    "publication": {
        "figure_source_data_required": True,
        "figure_generation_script_required": True,
        "vector_pdf_required": True,
        "vector_svg_required": True,
        "png_required": True,
        "png_dpi": 400,
        "font_embedding_required": True,
        "screenshots_prohibited": True,
    },

    "execution_separation": {
        "asus_breaKHis": "independent",
        "kaggle_cbis_ddsm": "independent",
        "historical_project": "reference_only",
        "cross_dataset_analysis": "only_after_independent_completion",
    },

    "training_policy": {
        "training_allowed_after_specification_freeze": True,
        "overwrite_previous_runs": False,
    },

    "source_basis": {
        "description": (
            "Scientific specification aligned to the current manuscript "
            "and the frozen fold-safe PCA decision."
        )
    },

    "created_utc": datetime.now(timezone.utc).isoformat(),
}

OUT.parent.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(spec, indent=2),
    encoding="utf-8",
)

print("=" * 100)
print("ASUS-18 — EXPERIMENT SPECIFICATION")
print("=" * 100)
print()
print("Specification:", OUT)
print("Dataset:", spec["primary_dataset"]["name"])
print("Folds:", spec["primary_dataset"]["outer_folds"])
print("Representation:", spec["representation"]["feature_dimension"], "->", spec["representation"]["pca_components"])
print("Experimental arms:", len(spec["experimental_arms"]))
print("Seeds:", spec["statistical_analysis"]["independent_seeds"])
print("Primary endpoint:", spec["primary_evaluation"]["primary_endpoint"])
print("Epistemic method:", spec["epistemic_uncertainty"]["primary_method"])
print("Figure PNG DPI:", spec["publication"]["png_dpi"])
print()
print("✓ Seven-arm experiment specification written")
print("✓ Fold-safe PCA locked")
print("✓ Three manuscript experiment seeds locked")
print("✓ Patient/case-level evaluation locked")
print("✓ Mutual-information epistemic uncertainty locked")
print("✓ Publication provenance requirements locked")
print("✓ ASUS/Kaggle execution separation preserved")
print()
print("NO TRAINING PERFORMED")
print("NO DATASET MODIFIED")
print("NO KAGGLE DATA TOUCHED")
print("=" * 100)
