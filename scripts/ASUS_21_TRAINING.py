from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys


# ============================================================
# ASUS-21 — AUTOMATIC EXPERIMENT RUNNER
# ============================================================
#
# INFRASTRUCTURE VERIFICATION ONLY
#
# This stage verifies:
#
#   5 outer folds
#   3 independent seeds
#   7 experimental arms
#
# Total:
#
#   15 fold/seed runs
#   105 planned arm executions
#
# NO MODEL TRAINING IS PERFORMED BY THIS SCRIPT.
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

CONFIG_FILE = (
    PROJECT
    / "configs"
    / "ASUS_21_TRAINING_SPECIFICATION_v1.json"
)

FEATURE_ROOT = (
    PROJECT
    / "features"
    / "BreaKHis"
    / "pca_6d"
)

OUTPUT_ROOT = (
    PROJECT
    / "features"
    / "BreaKHis"
    / "asus_21"
)

METADATA_DIR = (
    PROJECT
    / "metadata"
)


# ============================================================
# LOCKED EXPERIMENT STRUCTURE
# ============================================================

EXPECTED_FOLDS = 5

EXPECTED_SEEDS = [
    42,
    123,
    2025,
]

EXPECTED_ARMS = [
    "softmax",
    "temperature_scaled",
    "mc_dropout",
    "deep_ensemble",
    "deterministic_vqc",
    "laplace_vqc",
    "laplace_mlp",
]


# ============================================================
# SHA256
# ============================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# REQUIRE
# ============================================================

def require(
    condition,
    message,
):

    if not condition:

        raise RuntimeError(
            message
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "ASUS-21 — AUTOMATIC EXPERIMENT RUNNER"
    )

    print(
        "=" * 100
    )


    # ========================================================
    # PROJECT
    # ========================================================

    print()
    print(
        "PROJECT"
    )
    print(
        "-" * 100
    )

    print(
        "Project:",
        PROJECT,
    )

    require(
        PROJECT.exists(),
        f"Project directory missing:\n{PROJECT}",
    )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    print()
    print(
        "ASUS-21 CONFIGURATION"
    )
    print(
        "-" * 100
    )

    require(
        CONFIG_FILE.exists(),
        (
            "Missing ASUS-21 configuration:\n"
            f"{CONFIG_FILE}"
        ),
    )

    config_hash = sha256_file(
        CONFIG_FILE
    )

    config = json.loads(
        CONFIG_FILE.read_text(
            encoding="utf-8-sig"
        )
    )

    print(
        "Configuration:",
        CONFIG_FILE,
    )

    print(
        "Configuration SHA256:",
        config_hash,
    )


    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    print()
    print(
        "CROSS-VALIDATION"
    )
    print(
        "-" * 100
    )

    require(
        "cross_validation" in config,
        (
            "Missing 'cross_validation' "
            "section in ASUS-21 specification."
        ),
    )

    cv = config[
        "cross_validation"
    ]

    outer_folds = cv.get(
        "outer_folds"
    )

    seeds = cv.get(
        "seeds"
    )

    total_fold_seed_runs = cv.get(
        "total_fold_seed_runs"
    )

    print(
        "Outer folds:",
        outer_folds,
    )

    print(
        "Seeds:",
        seeds,
    )

    print(
        "Declared fold/seed runs:",
        total_fold_seed_runs,
    )

    require(
        outer_folds == EXPECTED_FOLDS,
        (
            "Expected 5 outer folds."
        ),
    )

    require(
        seeds == EXPECTED_SEEDS,
        (
            "Seed mismatch.\n"
            f"Expected: {EXPECTED_SEEDS}\n"
            f"Found: {seeds}"
        ),
    )

    require(
        total_fold_seed_runs == 15,
        (
            "Expected 15 fold/seed runs."
        ),
    )


    # ========================================================
    # TRAINING PARAMETERS
    # ========================================================

    print()
    print(
        "TRAINING SPECIFICATION"
    )
    print(
        "-" * 100
    )

    require(
        "training" in config,
        (
            "Missing 'training' section."
        ),
    )

    training = config[
        "training"
    ]

    epochs = training.get(
        "epochs"
    )

    batch_size = training.get(
        "batch_size"
    )

    learning_rate = training.get(
        "learning_rate"
    )

    weight_decay = training.get(
        "weight_decay"
    )

    selection_metric = training.get(
        "selection_metric"
    )

    threshold = training.get(
        "decision_threshold"
    )

    overwrite_previous_runs = training.get(
        "overwrite_previous_runs"
    )

    print(
        "Epochs:",
        epochs,
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Learning rate:",
        learning_rate,
    )

    print(
        "Weight decay:",
        weight_decay,
    )

    print(
        "Selection metric:",
        selection_metric,
    )

    print(
        "Decision threshold:",
        threshold,
    )

    print(
        "Overwrite previous runs:",
        overwrite_previous_runs,
    )

    require(
        epochs == 30,
        "Expected 30 epochs.",
    )

    require(
        batch_size == 32,
        "Expected batch size 32.",
    )

    require(
        learning_rate == 0.001,
        "Expected learning rate 0.001.",
    )

    require(
        weight_decay == 0.0001,
        "Expected weight decay 0.0001.",
    )

    require(
        selection_metric
        == "validation ROC-AUC",
        (
            "Unexpected selection metric."
        ),
    )

    require(
        threshold == 0.5,
        "Expected decision threshold 0.5.",
    )

    require(
        overwrite_previous_runs is False,
        (
            "Overwrite protection must remain "
            "enabled."
        ),
    )


    # ========================================================
    # EXPERIMENTAL ARMS
    # ========================================================

    print()
    print(
        "EXPERIMENTAL ARMS"
    )
    print(
        "-" * 100
    )

    require(
        "experimental_arms" in config,
        (
            "Missing 'experimental_arms' "
            "section."
        ),
    )

    arms = config[
        "experimental_arms"
    ]

    configured_arm_names = [
        arm["name"]
        for arm in arms
    ]

    print(
        "Configured arms:"
    )

    for name in configured_arm_names:

        print(
            "  ",
            name,
        )

    require(
        configured_arm_names
        == EXPECTED_ARMS,
        (
            "Experimental-arm mismatch.\n"
            f"Expected:\n{EXPECTED_ARMS}\n"
            f"Found:\n{configured_arm_names}"
        ),
    )

    print()

    for name in EXPECTED_ARMS:

        print(
            "✓",
            name,
        )


    # ========================================================
    # UNCERTAINTY
    # ========================================================

    print()
    print(
        "UNCERTAINTY SPECIFICATION"
    )
    print(
        "-" * 100
    )

    require(
        "uncertainty" in config,
        (
            "Missing 'uncertainty' section."
        ),
    )

    uncertainty = config[
        "uncertainty"
    ]

    print(
        "Primary method:",
        uncertainty.get(
            "primary_method"
        ),
    )

    print(
        "Formula:",
        uncertainty.get(
            "formula"
        ),
    )

    print(
        "MC-dropout passes:",
        uncertainty.get(
            "mc_dropout_passes"
        ),
    )

    print(
        "Deep-ensemble members:",
        uncertainty.get(
            "deep_ensemble_members"
        ),
    )

    require(
        uncertainty.get(
            "primary_method"
        )
        == "ensemble mutual information",
        (
            "Unexpected primary uncertainty method."
        ),
    )

    require(
        uncertainty.get(
            "formula"
        )
        == "H(mean(p)) - mean(H(p))",
        (
            "Unexpected uncertainty formula."
        ),
    )

    require(
        uncertainty.get(
            "aleatoric_uncertainty_claim"
        )
        is False,
        (
            "Aleatoric uncertainty claim "
            "must remain false."
        ),
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print(
        "EVALUATION"
    )
    print(
        "-" * 100
    )

    require(
        "evaluation" in config,
        (
            "Missing 'evaluation' section."
        ),
    )

    evaluation = config[
        "evaluation"
    ]

    print(
        "Evaluation unit:",
        evaluation.get(
            "unit"
        ),
    )

    print(
        "Primary endpoint:",
        evaluation.get(
            "primary_endpoint"
        ),
    )

    print(
        "Metrics:"
    )

    for metric in evaluation.get(
        "metrics",
        []
    ):

        print(
            "  ",
            metric,
        )

    require(
        evaluation.get(
            "unit"
        )
        == "patient_or_case",
        (
            "Evaluation unit must be "
            "patient_or_case."
        ),
    )

    require(
        evaluation.get(
            "primary_endpoint"
        )
        == "AURC",
        (
            "Primary endpoint must be AURC."
        ),
    )


    # ========================================================
    # PILOT POLICY
    # ========================================================

    print()
    print(
        "PILOT POLICY"
    )
    print(
        "-" * 100
    )

    require(
        "pilot" in config,
        (
            "Missing 'pilot' section."
        ),
    )

    pilot = config[
        "pilot"
    ]

    print(
        "Pilot required:",
        pilot.get(
            "required"
        ),
    )

    print(
        "Purpose:",
        pilot.get(
            "purpose"
        ),
    )

    print(
        "Full experiment blocked until pilot verification:",
        pilot.get(
            "full_experiment_must_not_start_before_pilot_verification"
        ),
    )

    require(
        pilot.get(
            "required"
        )
        is True,
        (
            "Pilot must be required."
        ),
    )

    require(
        pilot.get(
            "full_experiment_must_not_start_before_pilot_verification"
        )
        is True,
        (
            "Full experiment must remain blocked "
            "until pilot verification."
        ),
    )


    # ========================================================
    # ASUS-20B INPUT VERIFICATION
    # ========================================================

    print()
    print(
        "ASUS-20B INPUT VERIFICATION"
    )
    print(
        "-" * 100
    )

    verified_inputs = []

    input_count = 0

    for fold in range(
        1,
        EXPECTED_FOLDS + 1
    ):

        for seed in EXPECTED_SEEDS:

            run_dir = (
                FEATURE_ROOT
                / f"fold_{fold:02d}"
                / f"seed_{seed}"
            )

            feature_file = (
                run_dir
                / "BreaKHis_FOLD_FEATURES_v1.npz"
            )

            pca_file = (
                run_dir
                / "PCA_6D.npy"
            )

            provenance_file = (
                run_dir
                / "PCA_PROVENANCE.json"
            )

            require(
                feature_file.exists(),
                (
                    "Missing ASUS-20B feature file:\n"
                    f"{feature_file}"
                ),
            )

            require(
                pca_file.exists(),
                (
                    "Missing ASUS-20B PCA file:\n"
                    f"{pca_file}"
                ),
            )

            require(
                provenance_file.exists(),
                (
                    "Missing ASUS-20B provenance file:\n"
                    f"{provenance_file}"
                ),
            )

            verified_inputs.append(
                {
                    "fold": fold,
                    "seed": seed,

                    "feature_file":
                        str(
                            feature_file.relative_to(
                                PROJECT
                            )
                        ),

                    "feature_sha256":
                        sha256_file(
                            feature_file
                        ),

                    "pca_file":
                        str(
                            pca_file.relative_to(
                                PROJECT
                            )
                        ),

                    "pca_sha256":
                        sha256_file(
                            pca_file
                        ),

                    "provenance_file":
                        str(
                            provenance_file.relative_to(
                                PROJECT
                            )
                        ),

                    "provenance_sha256":
                        sha256_file(
                            provenance_file
                        ),
                }
            )

            input_count += 1

            print(
                f"✓ Fold {fold:02d} / Seed {seed}"
            )


    require(
        input_count == 15,
        (
            "Expected 15 ASUS-20B "
            "fold/seed inputs."
        ),
    )


    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    print()
    print(
        "OUTPUT SAFETY"
    )
    print(
        "-" * 100
    )

    print(
        "Output root:",
        OUTPUT_ROOT,
    )

    if OUTPUT_ROOT.exists():

        existing_files = [
            p
            for p in OUTPUT_ROOT.rglob("*")
            if p.is_file()
        ]

        if existing_files:

            raise FileExistsError(
                (
                    "ASUS-21 output directory "
                    "already contains files.\n"
                    "Overwrite is prohibited.\n\n"
                    f"{OUTPUT_ROOT}"
                )
            )

        print(
            "✓ Output directory exists "
            "and is empty."
        )

    else:

        print(
            "✓ ASUS-21 output directory "
            "does not yet exist."
        )


    # ========================================================
    # AUTOMATIC RUN PLAN
    # ========================================================

    print()
    print(
        "AUTOMATIC RUN PLAN"
    )
    print(
        "-" * 100
    )

    run_plan = []

    for fold in range(
        1,
        EXPECTED_FOLDS + 1
    ):

        for seed in EXPECTED_SEEDS:

            for arm in EXPECTED_ARMS:

                run_plan.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                    }
                )

    print(
        "Outer folds:",
        EXPECTED_FOLDS,
    )

    print(
        "Seeds:",
        len(EXPECTED_SEEDS),
    )

    print(
        "Experimental arms:",
        len(EXPECTED_ARMS),
    )

    print(
        "Fold/seed combinations:",
        EXPECTED_FOLDS
        * len(EXPECTED_SEEDS),
    )

    print(
        "Planned model runs:",
        len(run_plan),
    )

    require(
        len(run_plan) == 105,
        (
            "Expected exactly 105 "
            "planned model runs."
        ),
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    print()
    print(
        "ENVIRONMENT"
    )
    print(
        "-" * 100
    )

    print(
        "Python:",
        platform.python_version(),
    )

    print(
        "Python executable:",
        sys.executable,
    )


    # ========================================================
    # AUDIT RECORD
    # ========================================================

    audit = {

        "step":
            "ASUS-21",

        "status":
            "INFRASTRUCTURE_VERIFIED",

        "project":
            "quantum-epistemic-reliability",

        "configuration":
            str(
                CONFIG_FILE.relative_to(
                    PROJECT
                )
            ),

        "configuration_sha256":
            config_hash,

        "cross_validation":
            {
                "outer_folds":
                    EXPECTED_FOLDS,

                "seeds":
                    EXPECTED_SEEDS,

                "total_fold_seed_runs":
                    15,
            },

        "training":
            {
                "epochs":
                    epochs,

                "batch_size":
                    batch_size,

                "learning_rate":
                    learning_rate,

                "weight_decay":
                    weight_decay,

                "selection_metric":
                    selection_metric,

                "decision_threshold":
                    threshold,

                "overwrite_previous_runs":
                    overwrite_previous_runs,
            },

        "experimental_arms":
            EXPECTED_ARMS,

        "planned_model_runs":
            len(run_plan),

        "asus_20b_inputs_verified":
            input_count,

        "asus_20b_inputs":
            verified_inputs,

        "pilot_required":
            True,

        "training_performed":
            False,

        "scientific_outputs_created":
            False,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


    # ========================================================
    # WRITE INFRASTRUCTURE AUDIT
    # ========================================================

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_file = (
        METADATA_DIR
        / "ASUS-21_INFRASTRUCTURE_AUDIT_v1.json"
    )

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "ASUS-21 INFRASTRUCTURE CHECK — VERIFIED"
    )

    print(
        "=" * 100
    )

    print(
        "✓ ASUS-21 configuration verified"
    )

    print(
        "✓ UTF-8 / UTF-8-BOM handling verified"
    )

    print(
        "✓ Five outer folds verified"
    )

    print(
        "✓ Three independent seeds verified"
    )

    print(
        "✓ Seven experimental arms verified"
    )

    print(
        "✓ Training parameters verified"
    )

    print(
        "✓ Uncertainty specification verified"
    )

    print(
        "✓ AURC primary endpoint verified"
    )

    print(
        "✓ Pilot gate verified"
    )

    print(
        "✓ All 15 ASUS-20B inputs verified"
    )

    print(
        "✓ 105 automatic model runs planned"
    )

    print(
        "✓ Overwrite protection active"
    )

    print(
        "✓ Training NOT performed"
    )

    print(
        "✓ Scientific outputs NOT created"
    )

    print()
    print(
        "Infrastructure audit:"
    )

    print(
        audit_file
    )

    print()
    print(
        "TRAINING PERFORMED: FALSE"
    )

    print(
        "PILOT REQUIRED: TRUE"
    )

    print(
        "FULL EXPERIMENT BLOCKED UNTIL PILOT VERIFICATION: TRUE"
    )

    print(
        "=" * 100
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()