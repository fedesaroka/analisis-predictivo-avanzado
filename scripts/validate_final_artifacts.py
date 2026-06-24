"""Validate final test outputs and NumPy deployment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from anime_recommender import (
    load_deployment_artifacts,
    load_experiment_config,
    prepare_data,
    recommend_all_known_users,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIRECTORY = (
    ROOT
    / "results"
    / "final_model"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
)


def sha256_file(
    path: Path,
) -> str:
    """Calculate one SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sparse_equal(
    left: sparse.csr_matrix,
    right: sparse.csr_matrix,
) -> bool:
    """Compare sparse matrices exactly."""

    if left.shape != right.shape:
        return False

    return (
        left != right
    ).nnz == 0


def main() -> None:
    """Validate all final evaluation and deployment outputs."""

    required_result_files = {
        "final_model_state.pt",
        "final_recommendations.npz",
        "recommendation_examples.csv",
        "run_summary.json",
        "test_metrics.csv",
        "training_history.csv",
        "validation_test_comparison.csv",
    }

    required_artifact_files = {
        "anime_catalog.parquet",
        "artifact_manifest.json",
        "demo_users.json",
        "feature_names.json",
        "final_recommendations.npz",
        "item_features.npz",
        "item_mapping.json",
        "model_components.npz",
        "seen_final.npz",
        "user_mapping.json",
    }

    missing_results = sorted(
        file_name
        for file_name in required_result_files
        if not (
            RESULTS_DIRECTORY
            / file_name
        ).exists()
    )

    missing_artifacts = sorted(
        file_name
        for file_name in required_artifact_files
        if not (
            ARTIFACT_DIRECTORY
            / file_name
        ).exists()
    )

    if missing_results:
        raise FileNotFoundError(
            "Missing final result files: "
            + ", ".join(missing_results)
        )

    if missing_artifacts:
        raise FileNotFoundError(
            "Missing deployment artifact files: "
            + ", ".join(missing_artifacts)
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

    expected_seen_matrix = (
        prepared.train_matrix
        + prepared.validation_matrix
    )

    expected_seen_matrix = (
        expected_seen_matrix
        .astype(bool)
        .astype(np.float32)
        .tocsr()
    )

    exported_seen_matrix = sparse.load_npz(
        ARTIFACT_DIRECTORY
        / "seen_final.npz"
    ).tocsr()

    if not sparse_equal(
        expected_seen_matrix,
        exported_seen_matrix,
    ):
        raise AssertionError(
            "Exported seen matrix differs from train plus validation."
        )

    if exported_seen_matrix.multiply(
        prepared.test_matrix
    ).nnz:
        raise AssertionError(
            "Exported final seen matrix overlaps the test set."
        )

    artifacts = load_deployment_artifacts(
        ARTIFACT_DIRECTORY
    )

    generated_recommendations = (
        recommend_all_known_users(
            artifacts,
            k=int(
                config[
                    "recommendation_k"
                ]
            ),
        )
    )

    with np.load(
        ARTIFACT_DIRECTORY
        / "final_recommendations.npz",
        allow_pickle=False,
    ) as archive:
        stored_recommendations = archive[
            "recommendations"
        ]

    np.testing.assert_array_equal(
        generated_recommendations,
        stored_recommendations,
    )

    for user_index, item_indices in enumerate(
        stored_recommendations
    ):
        if exported_seen_matrix[
            user_index,
            item_indices,
        ].nnz:
            raise AssertionError(
                f"User {user_index} received a seen recommendation."
            )

    test_metrics = pd.read_csv(
        RESULTS_DIRECTORY
        / "test_metrics.csv"
    )

    expected_models = [
        "popularity",
        "content",
        "default_hybrid_warp_style",
        "tuned_hybrid_warp_style",
    ]

    if test_metrics[
        "model"
    ].tolist() != expected_models:
        raise AssertionError(
            "Unexpected final test model order."
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
        values = test_metrics[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise AssertionError(
                f"{column} contains non-finite values."
            )

        if not np.all(
            (values >= 0)
            & (values <= 1)
        ):
            raise AssertionError(
                f"{column} contains values outside [0, 1]."
            )

    selected_rows = test_metrics[
        test_metrics[
            "selected_before_test"
        ].astype(bool)
    ]

    if selected_rows[
        "model"
    ].tolist() != [
        "tuned_hybrid_warp_style"
    ]:
        raise AssertionError(
            "The frozen selected model is incorrect."
        )

    with (
        RESULTS_DIRECTORY
        / "run_summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        run_summary = json.load(file)

    with (
        ARTIFACT_DIRECTORY
        / "artifact_manifest.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if not bool(
        run_summary[
            "test_set_accessed"
        ]
    ):
        raise AssertionError(
            "Final run summary does not record test access."
        )

    if bool(
        run_summary[
            "selection_changed_after_test"
        ]
    ):
        raise AssertionError(
            "Model selection changed after test access."
        )

    if run_summary[
        "cohort_split_sha256"
    ] != prepared.summary[
        "cohort_split_sha256"
    ]:
        raise AssertionError(
            "Final results refer to another cohort."
        )

    if manifest[
        "cohort_split_sha256"
    ] != prepared.summary[
        "cohort_split_sha256"
    ]:
        raise AssertionError(
            "Artifact manifest refers to another cohort."
        )

    if bool(
        manifest[
            "deployment_requires_pytorch"
        ]
    ):
        raise AssertionError(
            "Deployment artifacts unexpectedly require PyTorch."
        )

    if float(
        manifest[
            "score_parity_max_abs_error"
        ]
    ) > 1e-4:
        raise AssertionError(
            "Deployment score parity error exceeds tolerance."
        )

    for artifact_record in (
        manifest[
            "files"
        ].values()
    ):
        path = (
            ARTIFACT_DIRECTORY
            / artifact_record[
                "path"
            ]
        )

        if int(
            artifact_record[
                "size_bytes"
            ]
        ) != int(
            path.stat().st_size
        ):
            raise AssertionError(
                f"Size mismatch for {path.name}."
            )

        actual_hash = sha256_file(
            path
        )

        if actual_hash != (
            artifact_record[
                "sha256"
            ]
        ):
            raise AssertionError(
                f"Hash mismatch for {path.name}."
            )

    history = pd.read_csv(
        RESULTS_DIRECTORY
        / "training_history.csv"
    )

    expected_history_lengths = {
        "default_hybrid_warp_style": int(
            config[
                "default_training"
            ][
                "epochs"
            ]
        ),
        "tuned_hybrid_warp_style": int(
            run_summary[
                "selected_parameters"
            ][
                "epochs"
            ]
        ),
    }

    for model_name, expected_length in (
        expected_history_lengths.items()
    ):
        model_history = history[
            history[
                "model"
            ].eq(model_name)
        ]

        if len(
            model_history
        ) != expected_length:
            raise AssertionError(
                f"{model_name} has an invalid history length."
            )

    examples = pd.read_csv(
        RESULTS_DIRECTORY
        / "recommendation_examples.csv"
    )

    expected_example_rows = (
        5
        * int(
            config[
                "recommendation_k"
            ]
        )
    )

    if len(examples) != expected_example_rows:
        raise AssertionError(
            "Final recommendation example count is invalid."
        )

    print(
        test_metrics[
            [
                "model",
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "sampled_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Deployment score parity error:",
        manifest[
            "score_parity_max_abs_error"
        ],
    )

    print(
        "Final artifact validation passed."
    )


if __name__ == "__main__":
    main()
