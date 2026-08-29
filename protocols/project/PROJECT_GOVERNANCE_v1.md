# PROJECT GOVERNANCE — NEW PROJECT

Project:
quantum-epistemic-reliability

Local execution:
D:\AI\quantum-epistemic-reliability

Primary local dataset:
BreaKHis

Independent external-shift execution:
CBIS-DDSM on Kaggle

Historical reference only:
D:\AI\quantum-uncertainty-shift

The historical project must never be used as an execution directory,
training directory, prediction directory, or result directory.

## Scientific rule

Every experiment must be:

1. explicitly configured;
2. reproducible from a recorded seed;
3. associated with a Git commit;
4. associated with frozen dataset manifests;
5. associated with recorded environment information;
6. associated with saved predictions;
7. associated with saved uncertainty values;
8. associated with fold-level metrics;
9. associated with aggregate metrics;
10. independently reproducible for figure/table generation.

## Dataset rule

BreaKHis uses the verified five-fold manifests already frozen in:

manifests/BreaKHis_FIVE_FOLD_FREEZE_v1.json

No new random patient split may silently replace these folds.

## Publication rule

Every manuscript figure must retain:

- source data;
- generation script;
- vector PDF;
- vector SVG where appropriate;
- 400 dpi PNG;
- reproducible metadata.

Screenshots are not acceptable as final figures.

## Git rule

Git checkpoints are created:

- after protocol freezes;
- after pipeline implementation;
- after each major verified experiment stage;
- before manuscript tables/figures;
- before final ZIP release.

Every final result must be traceable to a Git commit.

## Separation rule

The local ASUS experiment and the Kaggle CBIS-DDSM experiment remain independent.
Results are combined only during the predefined analysis stage after both datasets
have independently passed their validation checks.
