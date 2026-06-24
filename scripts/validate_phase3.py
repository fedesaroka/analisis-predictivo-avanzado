"""Validate metadata, evaluation metrics, and baseline models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender import (
    build_item_features,
    evaluate_baselines,
    load_experiment_config,
    prepare_data,
)


ROOT = Path(__file__).resolve().parents[1]


def sparse_matrices_are_equal(
    left,
    right,
) -> bool:
    """Compare sparse matrices exactly."""

    if left.shape != right.shape:
        return False

    return (
        left != right
    ).nnz == 0


def assert_no_seen_recommendations(
    recommendations: np.ndarray,
    train_matrix,
) -> None:
    """Ensure no recommendation was already positive in training."""

    for user_index, item_indices in enumerate(
        recommendations
    ):
        seen_count = train_matrix[
            user_index,
            item_indices,
        ].nnz

        if seen_count:
            raise AssertionError(
                f"User {user_index} received "
                f"{seen_count} seen recommendations."
            )


def main() -> None:
    config = load_experiment_config(
        ROOT / "config" / "experiment.json"
    )

    preparation_arguments = {
        "anime_path": (
            ROOT / "data" / "anime.csv"
        ),
        "ratings_path": (
            ROOT / "data" / "rating.parquet"
        ),
        "config": config,
    }

    prepared = prepare_data(
        **preparation_arguments
    )

    first_features = build_item_features(
        prepared.catalog
    )

    second_features = build_item_features(
        prepared.catalog
    )

    if first_features.feature_names != (
        second_features.feature_names
    ):
        raise AssertionError(
            "Feature vocabulary changed between executions."
        )

    if not sparse_matrices_are_equal(
        first_features.matrix,
        second_features.matrix,
    ):
        raise AssertionError(
            "Raw metadata matrix changed between executions."
        )

    if not sparse_matrices_are_equal(
        first_features.normalized_matrix,
        second_features.normalized_matrix,
    ):
        raise AssertionError(
            "Normalized metadata matrix changed "
            "between executions."
        )

    if first_features.matrix.shape[0] != (
        prepared.summary[
            "final_items"
        ]
    ):
        raise AssertionError(
            "Metadata row count does not equal final item count."
        )

    normalized_norms = np.sqrt(
        np.asarray(
            first_features.normalized_matrix.multiply(
                first_features.normalized_matrix
            ).sum(axis=1)
        ).ravel()
    )

    np.testing.assert_allclose(
        normalized_norms,
        1.0,
        rtol=0,
        atol=1e-6,
    )

    first_results, first_recommendations = (
        evaluate_baselines(
            prepared=prepared,
            item_features=first_features,
            config=config,
        )
    )

    second_results, second_recommendations = (
        evaluate_baselines(
            prepared=prepared,
            item_features=second_features,
            config=config,
        )
    )

    pd.testing.assert_frame_equal(
        first_results,
        second_results,
        check_exact=True,
    )

    for model_name in (
        first_recommendations
    ):
        np.testing.assert_array_equal(
            first_recommendations[
                model_name
            ],
            second_recommendations[
                model_name
            ],
        )

        assert_no_seen_recommendations(
            recommendations=(
                first_recommendations[
                    model_name
                ]
            ),
            train_matrix=(
                prepared.train_matrix
            ),
        )

    metric_columns = [
        column
        for column in first_results.columns
        if column != "model"
    ]

    for metric_column in metric_columns:
        metric_values = first_results[
            metric_column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            (
                metric_values >= 0
            )
            & (
                metric_values <= 1
            )
        ):
            raise AssertionError(
                f"{metric_column} contains values "
                "outside [0, 1]."
            )

    print(
        json.dumps(
            first_features.summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()

    print(
        first_results.to_string(
            index=False
        )
    )

    print()

    print(
        "Phase 3 deterministic validation passed."
    )


if __name__ == "__main__":
    main()
