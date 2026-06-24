"""Validate objectives, neural outputs, and recommendation files."""

from __future__ import annotations

import json
from pathlib import Path

# PyTorch must initialize before pandas and SciPy load native
# numerical runtimes on Windows.
import torch

import numpy as np
import pandas as pd
from scipy import sparse

from anime_recommender import (
    build_item_features,
    load_experiment_config,
    prepare_data,
)
from anime_recommender.training import (
    TrainingSpec,
    score_all_items,
    train_recommender,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "default_models"
)

EXPECTED_MODELS = [
    "popularity",
    "content",
    "collaborative_logistic",
    "collaborative_bpr",
    "collaborative_warp_style",
    "hybrid_logistic",
    "hybrid_bpr",
    "hybrid_warp_style",
]

NEURAL_MODELS = EXPECTED_MODELS[
    2:
]


def validate_objective_determinism() -> None:
    """Train every objective twice on a small deterministic problem."""

    train_matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
    )

    item_features = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
    )

    for objective in [
        "logistic",
        "bpr",
        "warp_style_hard_negative",
    ]:
        for use_metadata in [
            False,
            True,
        ]:
            model_name = (
                f"synthetic_{objective}_"
                f"{'hybrid' if use_metadata else 'collaborative'}"
            )

            specification = TrainingSpec(
                name=model_name,
                objective=objective,
                use_metadata=use_metadata,
                latent_dim=4,
                epochs=3,
                batch_size=4,
                learning_rate=0.01,
                weight_decay=1e-6,
                metadata_weight=1.0,
                negatives_per_positive=1,
                max_sampled=3,
                margin=1.0,
                gradient_clip_norm=5.0,
                seed=42,
            )

            first_run = train_recommender(
                train_matrix=train_matrix,
                item_feature_matrix=item_features,
                specification=specification,
                verbose=False,
            )

            first_scores = score_all_items(
                model=first_run.model,
                item_feature_matrix=item_features,
                n_users=train_matrix.shape[0],
                user_batch_size=2,
            )

            second_run = train_recommender(
                train_matrix=train_matrix,
                item_feature_matrix=item_features,
                specification=specification,
                verbose=False,
            )

            second_scores = score_all_items(
                model=second_run.model,
                item_feature_matrix=item_features,
                n_users=train_matrix.shape[0],
                user_batch_size=2,
            )

            pd.testing.assert_frame_equal(
                pd.DataFrame(
                    first_run.history
                ),
                pd.DataFrame(
                    second_run.history
                ),
                check_exact=True,
            )

            np.testing.assert_allclose(
                first_scores,
                second_scores,
                rtol=0,
                atol=1e-7,
            )


def main() -> None:
    """Validate generated default-model results."""

    required_files = {
        "validation_metrics.csv",
        "training_history.csv",
        "recommendation_examples.csv",
        "recommendations.npz",
        "run_summary.json",
    }

    missing_files = sorted(
        file_name
        for file_name in required_files
        if not (
            RESULTS_DIR
            / file_name
        ).exists()
    )

    if missing_files:
        raise FileNotFoundError(
            "Missing Phase 4 result files: "
            + ", ".join(
                missing_files
            )
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

    item_features = build_item_features(
        prepared.catalog
    )

    metrics = pd.read_csv(
        RESULTS_DIR
        / "validation_metrics.csv"
    )

    history = pd.read_csv(
        RESULTS_DIR
        / "training_history.csv"
    )

    examples = pd.read_csv(
        RESULTS_DIR
        / "recommendation_examples.csv"
    )

    with (
        RESULTS_DIR
        / "run_summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        run_summary = json.load(
            file
        )

    if metrics["model"].tolist() != (
        EXPECTED_MODELS
    ):
        raise AssertionError(
            "Unexpected model order or missing models."
        )

    if metrics["model"].duplicated().any():
        raise AssertionError(
            "validation_metrics.csv contains duplicate models."
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
        values = metrics[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():
            raise AssertionError(
                f"{column} contains non-finite values."
            )

        if not np.all(
            (
                values >= 0
            )
            & (
                values <= 1
            )
        ):
            raise AssertionError(
                f"{column} contains values outside [0, 1]."
            )

    expected_epochs = int(
        config[
            "default_training"
        ][
            "epochs"
        ]
    )

    for model_name in NEURAL_MODELS:
        model_history = history[
            history[
                "model"
            ].eq(model_name)
        ].copy()

        if len(
            model_history
        ) != expected_epochs:
            raise AssertionError(
                f"{model_name} has "
                f"{len(model_history)} history rows, "
                f"expected {expected_epochs}."
            )

        expected_epoch_numbers = list(
            range(
                1,
                expected_epochs + 1,
            )
        )

        if model_history[
            "epoch"
        ].astype(int).tolist() != (
            expected_epoch_numbers
        ):
            raise AssertionError(
                f"{model_name} epoch numbering is invalid."
            )

        if not np.isfinite(
            model_history[
                "loss"
            ].to_numpy(
                dtype=np.float64
            )
        ).all():
            raise AssertionError(
                f"{model_name} contains non-finite losses."
            )

    recommendation_archive = np.load(
        RESULTS_DIR
        / "recommendations.npz"
    )

    if sorted(
        recommendation_archive.files
    ) != sorted(
        EXPECTED_MODELS
    ):
        raise AssertionError(
            "recommendations.npz has unexpected model keys."
        )

    expected_shape = (
        int(
            prepared.summary[
                "final_users"
            ]
        ),
        int(
            config[
                "recommendation_k"
            ]
        ),
    )

    for model_name in EXPECTED_MODELS:
        recommendations = (
            recommendation_archive[
                model_name
            ]
        )

        if recommendations.shape != (
            expected_shape
        ):
            raise AssertionError(
                f"{model_name} recommendations have "
                f"shape {recommendations.shape}, "
                f"expected {expected_shape}."
            )

        for user_index, item_indices in enumerate(
            recommendations
        ):
            if prepared.train_matrix[
                user_index,
                item_indices,
            ].nnz:
                raise AssertionError(
                    f"{model_name} recommended a seen item "
                    f"for user {user_index}."
                )

    expected_example_rows = (
        len(
            EXPECTED_MODELS
        )
        * 3
        * int(
            config[
                "recommendation_k"
            ]
        )
    )

    if len(
        examples
    ) != expected_example_rows:
        raise AssertionError(
            "Unexpected recommendation-example row count: "
            f"{len(examples)}, expected "
            f"{expected_example_rows}."
        )

    if run_summary[
        "cohort_split_sha256"
    ] != prepared.summary[
        "cohort_split_sha256"
    ]:
        raise AssertionError(
            "The run summary refers to a different cohort."
        )

    if bool(
        run_summary[
            "test_set_accessed"
        ]
    ):
        raise AssertionError(
            "The default-model run incorrectly accessed test data."
        )

    neural_metrics = metrics[
        metrics[
            "model_family"
        ].eq("pytorch")
    ].copy()

    expected_selected_model = str(
        neural_metrics
        .sort_values(
            [
                "precision_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "model",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .iloc[0][
            "model"
        ]
    )

    if run_summary[
        "selected_default_model"
    ] != expected_selected_model:
        raise AssertionError(
            "The selected default model does not match "
            "the validation ranking."
        )

    if item_features.matrix.shape[0] != (
        prepared.train_matrix.shape[1]
    ):
        raise AssertionError(
            "Feature and item dimensions are inconsistent."
        )

    validate_objective_determinism()

    print(
        metrics[
            [
                "model",
                "precision_at_10",
                "ndcg_at_10",
                "sampled_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Selected default model:",
        run_summary[
            "selected_default_model"
        ],
    )

    print()

    print(
        "Phase 4 validation passed."
    )


if __name__ == "__main__":
    main()
