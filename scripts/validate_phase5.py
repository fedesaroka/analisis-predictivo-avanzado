"""Validate the Bayesian optimization result set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender import (
    load_experiment_config,
    prepare_data,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "optuna"
)

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
    / "default_models"
)


REQUIRED_FILES = {
    "best_params.json",
    "best_recommendations.npz",
    "best_training_history.csv",
    "best_validation_metrics.csv",
    "parameter_importances.csv",
    "recommendation_examples.csv",
    "run_summary.json",
    "trials.csv",
    "validation_comparison.csv",
}


def select_expected_trial(
    trials: pd.DataFrame,
) -> pd.Series:
    """Apply the documented deterministic selection rule."""

    completed = trials[
        trials["state"].eq("COMPLETE")
    ].copy()

    return (
        completed
        .sort_values(
            [
                "precision_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "trial_number",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )


def main() -> None:
    """Validate all Phase 5 output files."""

    missing_files = sorted(
        file_name
        for file_name in REQUIRED_FILES
        if not (
            RESULTS_DIR
            / file_name
        ).exists()
    )

    if missing_files:
        raise FileNotFoundError(
            "Missing Phase 5 files: "
            + ", ".join(missing_files)
        )

    config = load_experiment_config(
        ROOT
        / "config"
        / "experiment.json"
    )

    prepared = prepare_data(
        anime_path=(
            ROOT
            / "data"
            / "anime.csv"
        ),
        ratings_path=(
            ROOT
            / "data"
            / "rating.parquet"
        ),
        config=config,
    )

    trials = pd.read_csv(
        RESULTS_DIR
        / "trials.csv"
    )

    best_metrics = pd.read_csv(
        RESULTS_DIR
        / "best_validation_metrics.csv"
    )

    comparison = pd.read_csv(
        RESULTS_DIR
        / "validation_comparison.csv"
    )

    history = pd.read_csv(
        RESULTS_DIR
        / "best_training_history.csv"
    )

    importances = pd.read_csv(
        RESULTS_DIR
        / "parameter_importances.csv"
    )

    examples = pd.read_csv(
        RESULTS_DIR
        / "recommendation_examples.csv"
    )

    with (
        RESULTS_DIR
        / "best_params.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        best_params = json.load(file)

    with (
        RESULTS_DIR
        / "run_summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        run_summary = json.load(file)

    with (
        DEFAULT_RESULTS_DIR
        / "run_summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        default_summary = json.load(file)

    requested_trials = int(
        config["optuna_trials"]
    )

    if len(trials) != requested_trials:
        raise AssertionError(
            f"Expected {requested_trials} trials, "
            f"found {len(trials)}."
        )

    if trials["trial_number"].tolist() != list(
        range(requested_trials)
    ):
        raise AssertionError(
            "Trial numbers are missing or out of order."
        )

    if not trials["state"].eq(
        "COMPLETE"
    ).all():
        raise AssertionError(
            "At least one trial did not complete."
        )

    if not bool(
        trials.loc[
            trials["trial_number"].eq(0),
            "is_default_trial",
        ].iloc[0]
    ):
        raise AssertionError(
            "Trial zero is not identified as the default trial."
        )

    metric_columns = [
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "hit_rate_at_10",
        "catalog_coverage_at_10",
        "intra_list_diversity_at_10",
        "sampled_auc",
    ]

    for column in metric_columns:
        values = trials[
            column
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise AssertionError(
                f"{column} contains non-finite trial values."
            )

        if not np.all(
            (values >= 0)
            & (values <= 1)
        ):
            raise AssertionError(
                f"{column} contains values outside [0, 1]."
            )

    expected_trial = select_expected_trial(
        trials
    )

    if int(
        expected_trial["trial_number"]
    ) != int(
        run_summary["selected_trial_number"]
    ):
        raise AssertionError(
            "The selected trial does not match the "
            "documented ranking rule."
        )

    if len(best_metrics) != 1:
        raise AssertionError(
            "best_validation_metrics.csv must contain one row."
        )

    tuned_precision = float(
        best_metrics.iloc[0][
            "precision_at_10"
        ]
    )

    selected_trial_precision = float(
        expected_trial[
            "precision_at_10"
        ]
    )

    if not np.isclose(
        tuned_precision,
        selected_trial_precision,
        rtol=0,
        atol=1e-8,
    ):
        raise AssertionError(
            "The clean retraining does not reproduce "
            "the selected trial."
        )

    default_precision = float(
        default_summary[
            "selected_default_model_metrics"
        ][
            "precision_at_10"
        ]
    )

    if tuned_precision + 1e-12 < (
        default_precision
    ):
        raise AssertionError(
            "The tuned result is worse than the included "
            "default trial."
        )

    if run_summary[
        "cohort_split_sha256"
    ] != prepared.summary[
        "cohort_split_sha256"
    ]:
        raise AssertionError(
            "The Optuna results use a different cohort."
        )

    if bool(
        run_summary[
            "test_set_accessed"
        ]
    ):
        raise AssertionError(
            "The Optuna phase accessed the test set."
        )

    if run_summary[
        "source_default_model"
    ] != default_summary[
        "selected_default_model"
    ]:
        raise AssertionError(
            "The tuned source model differs from Phase 4."
        )

    if best_params[
        "source_model"
    ] != default_summary[
        "selected_default_model"
    ]:
        raise AssertionError(
            "best_params.json identifies the wrong source model."
        )

    recommendations_archive = np.load(
        RESULTS_DIR
        / "best_recommendations.npz"
    )

    expected_model_name = run_summary[
        "selected_model_name"
    ]

    if recommendations_archive.files != [
        expected_model_name
    ]:
        raise AssertionError(
            "The recommendation archive has unexpected keys."
        )

    recommendations = recommendations_archive[
        expected_model_name
    ]

    expected_shape = (
        int(
            prepared.summary[
                "final_users"
            ]
        ),
        int(
            config["recommendation_k"]
        ),
    )

    if recommendations.shape != expected_shape:
        raise AssertionError(
            f"Recommendation shape {recommendations.shape} "
            f"does not match {expected_shape}."
        )

    for user_index, item_indices in enumerate(
        recommendations
    ):
        if prepared.train_matrix[
            user_index,
            item_indices,
        ].nnz:
            raise AssertionError(
                f"A seen training item was recommended "
                f"to user {user_index}."
            )

    expected_example_rows = (
        3
        * int(config["recommendation_k"])
    )

    if len(examples) != expected_example_rows:
        raise AssertionError(
            f"Expected {expected_example_rows} example rows, "
            f"found {len(examples)}."
        )

    expected_epochs = int(
        best_params["epochs"]
    )

    if history["epoch"].astype(int).tolist() != list(
        range(
            1,
            expected_epochs + 1,
        )
    ):
        raise AssertionError(
            "The selected training history has invalid epochs."
        )

    if not np.isfinite(
        history["loss"].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise AssertionError(
            "The selected training history has invalid losses."
        )

    if importances.empty:
        raise AssertionError(
            "Parameter importance output is empty."
        )

    importance_values = importances[
        "importance"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        importance_values
    ).all():
        raise AssertionError(
            "Parameter importances contain invalid values."
        )

    if not np.isclose(
        importance_values.sum(),
        1.0,
        rtol=0,
        atol=1e-8,
    ):
        raise AssertionError(
            "Parameter importances do not sum to one."
        )

    if comparison["variant"].tolist() != [
        "default",
        "tuned",
    ]:
        raise AssertionError(
            "The comparison rows are not default then tuned."
        )

    print(
        trials[
            [
                "trial_number",
                "precision_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
            ]
        ].to_string(index=False)
    )

    print()

    print(
        comparison.to_string(index=False)
    )

    print()

    print(
        "Selected trial:",
        run_summary[
            "selected_trial_number"
        ],
    )

    print(
        "Phase 5 validation passed."
    )


if __name__ == "__main__":
    main()
