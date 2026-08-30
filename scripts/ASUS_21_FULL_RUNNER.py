from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(
    r"D:\AI\quantum-epistemic-reliability"
)

TRAINING_SCRIPT = (
    PROJECT
    / "scripts"
    / "ASUS_21_TRAINING.py"
)

EXPERIMENT_ROOT = (
    PROJECT
    / "experiments"
    / "BreaKHis"
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2025]

EXPECTED_ARMS = [
    "softmax",
    "temperature_scaled",
    "mc_dropout",
    "deep_ensemble",
    "deterministic_vqc",
    "laplace_vqc",
    "laplace_mlp",
]


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def cell_directory(
    fold,
    seed,
):
    return (
        EXPERIMENT_ROOT
        / f"fold_{fold:02d}"
        / f"seed_{seed}"
    )


def completion_file(
    fold,
    seed,
):
    return (
        cell_directory(
            fold,
            seed,
        )
        / "CELL_COMPLETE.json"
    )


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        f"asus21_training_fold_runtime",
        TRAINING_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load ASUS-21 training module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "asus21_training_fold_runtime"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def is_complete(
    fold,
    seed,
):

    marker = completion_file(
        fold,
        seed,
    )

    if not marker.exists():
        return False

    try:

        payload = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )

        return (
            payload.get("status")
            == "COMPLETE"
            and
            payload.get("fold")
            == fold
            and
            payload.get("seed")
            == seed
        )

    except Exception:

        return False


def remove_incomplete_cell(
    fold,
    seed,
):

    directory = cell_directory(
        fold,
        seed,
    )

    if directory.exists():

        print(
            f"Removing incomplete cell: "
            f"Fold {fold:02d} / Seed {seed}"
        )

        shutil.rmtree(
            directory
        )


def write_complete_marker(
    fold,
    seed,
):

    directory = cell_directory(
        fold,
        seed,
    )

    marker = (
        directory
        / "CELL_COMPLETE.json"
    )

    marker.write_text(
        json.dumps(
            {
                "status":
                    "COMPLETE",

                "fold":
                    fold,

                "seed":
                    seed,

                "created_utc":
                    utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_failed_marker(
    fold,
    seed,
    error_text,
):

    directory = cell_directory(
        fold,
        seed,
    )

    if not directory.exists():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    marker = (
        directory
        / "CELL_FAILED.json"
    )

    marker.write_text(
        json.dumps(
            {
                "status":
                    "FAILED",

                "fold":
                    fold,

                "seed":
                    seed,

                "error":
                    error_text,

                "created_utc":
                    utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def validate_results(
    fold,
    seed,
):

    directory = cell_directory(
        fold,
        seed,
    )

    results_file = (
        directory
        / "PILOT_RESULTS.json"
    )

    if not results_file.exists():

        raise RuntimeError(
            "Missing results file:\n"
            f"{results_file}"
        )

    payload = json.loads(
        results_file.read_text(
            encoding="utf-8"
        )
    )

    executed = payload.get(
        "executed_arms",
        [],
    )

    blocked = payload.get(
        "blocked_arms",
        [],
    )

    if executed != EXPECTED_ARMS:

        raise RuntimeError(
            "Unexpected executed-arm list:\n"
            f"{executed}"
        )

    if blocked != []:

        raise RuntimeError(
            "Blocked arms detected:\n"
            f"{blocked}"
        )

    if len(
        payload.get(
            "results",
            [],
        )
    ) != 7:

        raise RuntimeError(
            "Expected exactly 7 arm result records."
        )

    return payload


def run_cell(
    fold,
    seed,
):

    directory = cell_directory(
        fold,
        seed,
    )

    print()
    print("=" * 100)
    print(
        "ASUS-21 FULL EXPERIMENT"
    )
    print(
        f"FOLD {fold:02d}/05 | SEED {seed}"
    )
    print("=" * 100)

    # ------------------------------------------------------------
    # IMPORTANT:
    # Never create the output directory here.
    #
    # ASUS_21_TRAINING.py creates OUTPUT_ROOT itself with
    # exist_ok=False. We therefore remove only an incomplete
    # previous directory before starting.
    # ------------------------------------------------------------

    remove_incomplete_cell(
        fold,
        seed,
    )

    module = load_training_module()

    module.PILOT_FOLD = fold
    module.PILOT_SEED = seed

    module.OUTPUT_ROOT = directory

    module.AUDIT_FILE = (
        directory
        / "ASUS-21_CELL_AUDIT.json"
    )

    # ------------------------------------------------------------
    # Let ASUS_21_TRAINING.py create the cell directory.
    # ------------------------------------------------------------

    module.main()

    payload = validate_results(
        fold,
        seed,
    )

    write_complete_marker(
        fold,
        seed,
    )

    print()
    print(
        f"COMPLETED: Fold {fold:02d} / Seed {seed}"
    )

    return payload


def main():

    if not TRAINING_SCRIPT.exists():

        raise FileNotFoundError(
            TRAINING_SCRIPT
        )

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_cells = (
        len(FOLDS)
        * len(SEEDS)
    )

    total_arm_runs = (
        total_cells
        * len(EXPECTED_ARMS)
    )

    completed = 0
    skipped = 0
    failed = 0

    print("=" * 100)
    print(
        "ASUS-21 - FULL 5-FOLD X 3-SEED RUNNER"
    )
    print("=" * 100)

    print(
        "Cells:",
        total_cells,
    )

    print(
        "Folds:",
        FOLDS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Experimental arms:",
        len(EXPECTED_ARMS),
    )

    print(
        "Total arm-level executions:",
        total_arm_runs,
    )

    print("=" * 100)

    for fold in FOLDS:

        for seed in SEEDS:

            if is_complete(
                fold,
                seed,
            ):

                skipped += 1

                print()
                print(
                    f"SKIP COMPLETE: "
                    f"Fold {fold:02d} / Seed {seed}"
                )

                continue

            try:

                run_cell(
                    fold,
                    seed,
                )

                completed += 1

            except KeyboardInterrupt:

                print()
                print(
                    "RUNNER INTERRUPTED."
                )

                print(
                    "Completed:",
                    completed,
                )

                print(
                    "Skipped:",
                    skipped,
                )

                print(
                    "Failed:",
                    failed,
                )

                print(
                    "Rerun the same command to resume."
                )

                return 2

            except Exception:

                failed += 1

                error_text = (
                    traceback.format_exc()
                )

                print()
                print(
                    f"FAILED: Fold {fold:02d} / Seed {seed}"
                )

                print(
                    error_text
                )

                write_failed_marker(
                    fold,
                    seed,
                    error_text,
                )

                print(
                    "Stopping after the first genuine failure."
                )

                print(
                    "The failed cell will be removed automatically "
                    "when the runner is resumed."
                )

                return 1

    print()
    print("=" * 100)
    print(
        "ASUS-21 FULL EXPERIMENT COMPLETE"
    )
    print("=" * 100)

    print(
        "Completed cells:",
        completed,
    )

    print(
        "Skipped complete cells:",
        skipped,
    )

    print(
        "Failed cells:",
        failed,
    )

    if (
        completed + skipped == total_cells
        and failed == 0
    ):

        print()
        print(
            "ALL 105 ARM EXECUTIONS COMPLETE"
        )

        print(
            f"Results root: {EXPERIMENT_ROOT}"
        )

        return 0

    return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )