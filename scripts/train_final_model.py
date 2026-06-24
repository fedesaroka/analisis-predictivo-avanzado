"""Train the frozen final recommender and evaluate the test set once."""

from __future__ import annotations

import os


# Configure native numerical runtimes before importing PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# PyTorch must initialize before pandas and SciPy on Windows.
import torch

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from anime_recommender import (
    build_item_features,
    content_score_matrix,
    evaluate_score_matrix,
    load_experiment_config,
    popularity_score_matrix,
    prepare_data,
)
from anime_recommender.training import (
    TrainingSpec,
    score_all_items,
    train_recommender,
)


ROOT = Path(__file__).resolve().parents[1]

OPTUNA_DIRECTORY = (
    ROOT
    / "results"
    / "optuna"
)

DEFAULT_DIRECTORY = (
    ROOT
    / "results"
    / "default_models"
)

DEFAULT_RESULTS_DIRECTORY = (
    ROOT
    / "results"
    / "final_model"
)

DEFAULT_ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
)


def parse_arguments() -> argparse.Namespace:
    """Parse final-training command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--confirm-test-access",
        action="store_true",
        help=(
            "Required confirmation that model selection is frozen "
            "and the test set may now be evaluated."
        ),
    )

    return parser.parse_args()


def prepare_directory(
    path: Path,
    overwrite: bool,
) -> None:
    """Create one clean output directory."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}"
            )

        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=False,
    )


def sha256_file(
    path: Path,
) -> str:
    """Calculate a file's SHA-256 digest."""

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


def json_default(
    value: Any,
) -> Any:
    """Convert scientific Python values for JSON."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    raise TypeError(
        f"Cannot serialize {type(value)}"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one UTF-8 JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_default_specification(
    config: dict[str, Any],
) -> TrainingSpec:
    """Build the untuned final comparator specification."""

    defaults = config[
        "default_training"
    ]

    return TrainingSpec(
        name="default_hybrid_warp_style",
        objective=(
            "warp_style_hard_negative"
        ),
        use_metadata=True,
        latent_dim=int(
            defaults["latent_dim"]
        ),
        epochs=int(
            defaults["epochs"]
        ),
        batch_size=int(
            defaults["batch_size"]
        ),
        learning_rate=float(
            defaults["learning_rate"]
        ),
        weight_decay=float(
            defaults["weight_decay"]
        ),
        metadata_weight=float(
            defaults["metadata_weight"]
        ),
        negatives_per_positive=int(
            defaults[
                "negatives_per_positive"
            ]
        ),
        max_sampled=int(
            defaults["max_sampled"]
        ),
        margin=float(
            defaults["margin"]
        ),
        gradient_clip_norm=float(
            defaults[
                "gradient_clip_norm"
            ]
        ),
        seed=int(
            config["seed"]
        ),
    )


def build_tuned_specification(
    best_params: dict[str, Any],
) -> TrainingSpec:
    """Build the frozen Optuna-selected specification."""

    if str(
        best_params["objective"]
    ) != "warp_style_hard_negative":
        raise ValueError(
            "The selected objective is not WARP-style."
        )

    if not bool(
        best_params["use_metadata"]
    ):
        raise ValueError(
            "The selected model is not hybrid."
        )

    return TrainingSpec(
        name="tuned_hybrid_warp_style",
        objective=str(
            best_params["objective"]
        ),
        use_metadata=True,
        latent_dim=int(
            best_params["latent_dim"]
        ),
        epochs=int(
            best_params["epochs"]
        ),
        batch_size=int(
            best_params["batch_size"]
        ),
        learning_rate=float(
            best_params["learning_rate"]
        ),
        weight_decay=float(
            best_params["weight_decay"]
        ),
        metadata_weight=float(
            best_params["metadata_weight"]
        ),
        negatives_per_positive=1,
        max_sampled=int(
            best_params["max_sampled"]
        ),
        margin=float(
            best_params["margin"]
        ),
        gradient_clip_norm=float(
            best_params[
                "gradient_clip_norm"
            ]
        ),
        seed=int(
            best_params["seed"]
        ),
    )


def evaluate_scores(
    model_name: str,
    variant: str,
    scores: np.ndarray,
    seen_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    normalized_features: sparse.csr_matrix,
    config: dict[str, Any],
    selected_before_test: bool,
) -> tuple[
    dict[str, Any],
    np.ndarray,
]:
    """Evaluate one predetermined model on the final test target."""

    (
        metrics,
        recommendations,
    ) = evaluate_score_matrix(
        scores=scores,
        seen_matrix=seen_matrix,
        target_matrix=test_matrix,
        normalized_item_features=(
            normalized_features
        ),
        k=int(
            config["recommendation_k"]
        ),
        auc_negatives=int(
            config[
                "sampled_auc_negatives"
            ]
        ),
        seed=int(
            config["seed"]
        ),
    )

    row = {
        "model": model_name,
        "variant": variant,
        "selected_before_test": (
            selected_before_test
        ),
        **metrics,
    }

    return row, recommendations


def create_examples(
    prepared,
    final_seen_matrix: sparse.csr_matrix,
    recommendations: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Create deterministic demo users and readable examples."""

    interaction_counts = np.diff(
        final_seen_matrix.indptr
    )

    sort_order = np.lexsort(
        (
            np.arange(
                len(interaction_counts)
            ),
            interaction_counts,
        )
    )

    selected_positions = np.linspace(
        0,
        len(sort_order) - 1,
        num=5,
        dtype=np.int64,
    )

    demo_user_indices = sort_order[
        selected_positions
    ]

    catalog_lookup = (
        prepared.catalog
        .set_index("item_index")
    )

    demo_users: list[
        dict[str, Any]
    ] = []

    example_rows: list[
        dict[str, Any]
    ] = []

    for user_index in demo_user_indices:
        user_index = int(
            user_index
        )

        user_id = int(
            prepared.user_ids[
                user_index
            ]
        )

        recommended_items = (
            recommendations[
                user_index
            ]
        )

        recommendation_records: list[
            dict[str, Any]
        ] = []

        for rank, item_index in enumerate(
            recommended_items,
            start=1,
        ):
            item_index = int(
                item_index
            )

            row = catalog_lookup.loc[
                item_index
            ]

            is_test_hit = bool(
                prepared.test_matrix[
                    user_index,
                    item_index,
                ]
            )

            record = {
                "rank": int(rank),
                "item_index": item_index,
                "anime_id": int(
                    row["anime_id"]
                ),
                "title": str(
                    row["name"]
                ),
                "genre": str(
                    row["genre"]
                ),
                "type": str(
                    row["type"]
                ),
                "test_hit": is_test_hit,
            }

            recommendation_records.append(
                record
            )

            example_rows.append(
                {
                    "user_index": user_index,
                    "user_id": user_id,
                    "positive_history_count": int(
                        interaction_counts[
                            user_index
                        ]
                    ),
                    **record,
                }
            )

        demo_users.append(
            {
                "user_index": user_index,
                "user_id": user_id,
                "label": (
                    f"User {user_id} — "
                    f"{int(interaction_counts[user_index])} "
                    "positive titles"
                ),
                "positive_history_count": int(
                    interaction_counts[
                        user_index
                    ]
                ),
                "recommendations": (
                    recommendation_records
                ),
            }
        )

    return (
        demo_users,
        pd.DataFrame(
            example_rows
        ),
    )


def current_git_commit() -> str | None:
    """Return the current Git commit when available."""

    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def main() -> None:
    """Train, evaluate, and export the frozen final model."""

    arguments = parse_arguments()

    if not arguments.confirm_test_access:
        raise RuntimeError(
            "Final evaluation requires "
            "--confirm-test-access because this phase "
            "uses the held-out test set."
        )

    results_directory = (
        arguments.results_dir.resolve()
    )

    artifact_directory = (
        arguments.artifacts_dir.resolve()
    )

    prepare_directory(
        results_directory,
        overwrite=arguments.overwrite,
    )

    prepare_directory(
        artifact_directory,
        overwrite=arguments.overwrite,
    )

    config = load_experiment_config(
        ROOT
        / "config"
        / "experiment.json"
    )

    final_policy = config.get(
        "final_evaluation"
    )

    if not final_policy:
        raise ValueError(
            "final_evaluation is absent from experiment.json."
        )

    if not bool(
        final_policy[
            "selection_locked_before_test"
        ]
    ):
        raise ValueError(
            "Model selection is not recorded as locked."
        )

    best_params = load_json(
        OPTUNA_DIRECTORY
        / "best_params.json"
    )

    optuna_summary = load_json(
        OPTUNA_DIRECTORY
        / "run_summary.json"
    )

    default_summary = load_json(
        DEFAULT_DIRECTORY
        / "run_summary.json"
    )

    if bool(
        optuna_summary[
            "test_set_accessed"
        ]
    ):
        raise RuntimeError(
            "Optuna results indicate test-set access."
        )

    if best_params[
        "source_model"
    ] != optuna_summary[
        "source_default_model"
    ]:
        raise RuntimeError(
            "The best-parameter source model is inconsistent."
        )

    print(
        "Preparing frozen cohort...",
        flush=True,
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

    expected_cohort_hash = (
        optuna_summary[
            "cohort_split_sha256"
        ]
    )

    if prepared.summary[
        "cohort_split_sha256"
    ] != expected_cohort_hash:
        raise RuntimeError(
            "The final cohort differs from the tuning cohort."
        )

    item_features = build_item_features(
        prepared.catalog
    )

    final_seen_matrix = (
        prepared.train_matrix
        + prepared.validation_matrix
    )

    final_seen_matrix = (
        final_seen_matrix
        .astype(bool)
        .astype(np.float32)
        .tocsr()
    )

    expected_seen_interactions = (
        prepared.train_matrix.nnz
        + prepared.validation_matrix.nnz
    )

    if final_seen_matrix.nnz != (
        expected_seen_interactions
    ):
        raise RuntimeError(
            "Train and validation were not combined correctly."
        )

    if final_seen_matrix.multiply(
        prepared.test_matrix
    ).nnz:
        raise RuntimeError(
            "Final training data overlaps the test set."
        )

    print(
        "Final training interactions:",
        final_seen_matrix.nnz,
        flush=True,
    )

    test_rows: list[
        dict[str, Any]
    ] = []

    recommendations_by_model: dict[
        str,
        np.ndarray,
    ] = {}

    print(
        "Evaluating final popularity baseline...",
        flush=True,
    )

    popularity_scores = (
        popularity_score_matrix(
            final_seen_matrix
        )
    )

    (
        popularity_row,
        popularity_recommendations,
    ) = evaluate_scores(
        model_name="popularity",
        variant="baseline",
        scores=popularity_scores,
        seen_matrix=final_seen_matrix,
        test_matrix=prepared.test_matrix,
        normalized_features=(
            item_features.normalized_matrix
        ),
        config=config,
        selected_before_test=False,
    )

    test_rows.append(
        popularity_row
    )

    recommendations_by_model[
        "popularity"
    ] = popularity_recommendations

    del popularity_scores

    print(
        "Evaluating final content baseline...",
        flush=True,
    )

    content_scores = content_score_matrix(
        final_seen_matrix,
        item_features.normalized_matrix,
    )

    (
        content_row,
        content_recommendations,
    ) = evaluate_scores(
        model_name="content",
        variant="baseline",
        scores=content_scores,
        seen_matrix=final_seen_matrix,
        test_matrix=prepared.test_matrix,
        normalized_features=(
            item_features.normalized_matrix
        ),
        config=config,
        selected_before_test=False,
    )

    test_rows.append(
        content_row
    )

    recommendations_by_model[
        "content"
    ] = content_recommendations

    del content_scores

    default_specification = (
        build_default_specification(
            config
        )
    )

    tuned_specification = (
        build_tuned_specification(
            best_params
        )
    )

    print()
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "TRAINING FINAL DEFAULT COMPARATOR",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )

    default_run = train_recommender(
        train_matrix=final_seen_matrix,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        specification=(
            default_specification
        ),
        verbose=True,
    )

    default_scores = score_all_items(
        model=default_run.model,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        n_users=len(
            prepared.user_ids
        ),
        user_batch_size=int(
            config[
                "default_training"
            ][
                "scoring_user_batch_size"
            ]
        ),
    )

    (
        default_row,
        default_recommendations,
    ) = evaluate_scores(
        model_name=(
            "default_hybrid_warp_style"
        ),
        variant="default",
        scores=default_scores,
        seen_matrix=final_seen_matrix,
        test_matrix=prepared.test_matrix,
        normalized_features=(
            item_features.normalized_matrix
        ),
        config=config,
        selected_before_test=False,
    )

    default_row.update(
        {
            "objective": (
                default_specification.objective
            ),
            "latent_dim": (
                default_specification.latent_dim
            ),
            "epochs": (
                default_specification.epochs
            ),
            "final_train_loss": float(
                default_run.history[
                    -1
                ]["loss"]
            ),
            "parameter_count": int(
                default_run.parameter_count
            ),
        }
    )

    test_rows.append(
        default_row
    )

    recommendations_by_model[
        default_row["model"]
    ] = default_recommendations

    print()
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "TRAINING FROZEN TUNED FINAL MODEL",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )

    tuned_run = train_recommender(
        train_matrix=final_seen_matrix,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        specification=(
            tuned_specification
        ),
        verbose=True,
    )

    tuned_scores = score_all_items(
        model=tuned_run.model,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        n_users=len(
            prepared.user_ids
        ),
        user_batch_size=int(
            best_params[
                "scoring_user_batch_size"
            ]
        ),
    )

    (
        tuned_row,
        tuned_recommendations,
    ) = evaluate_scores(
        model_name=(
            "tuned_hybrid_warp_style"
        ),
        variant="tuned_final",
        scores=tuned_scores,
        seen_matrix=final_seen_matrix,
        test_matrix=prepared.test_matrix,
        normalized_features=(
            item_features.normalized_matrix
        ),
        config=config,
        selected_before_test=True,
    )

    tuned_row.update(
        {
            "objective": (
                tuned_specification.objective
            ),
            "latent_dim": (
                tuned_specification.latent_dim
            ),
            "epochs": (
                tuned_specification.epochs
            ),
            "final_train_loss": float(
                tuned_run.history[
                    -1
                ]["loss"]
            ),
            "parameter_count": int(
                tuned_run.parameter_count
            ),
        }
    )

    test_rows.append(
        tuned_row
    )

    recommendations_by_model[
        tuned_row["model"]
    ] = tuned_recommendations

    test_metrics = pd.DataFrame(
        test_rows
    )

    model_order = {
        "popularity": 0,
        "content": 1,
        "default_hybrid_warp_style": 2,
        "tuned_hybrid_warp_style": 3,
    }

    test_metrics[
        "_model_order"
    ] = test_metrics[
        "model"
    ].map(model_order)

    test_metrics = (
        test_metrics
        .sort_values(
            "_model_order"
        )
        .drop(
            columns="_model_order"
        )
        .reset_index(drop=True)
    )

    validation_metrics = pd.DataFrame(
        [
            {
                "model": (
                    "default_hybrid_warp_style"
                ),
                "validation_precision_at_10": float(
                    default_summary[
                        "selected_default_model_metrics"
                    ][
                        "precision_at_10"
                    ]
                ),
                "validation_recall_at_10": float(
                    default_summary[
                        "selected_default_model_metrics"
                    ][
                        "recall_at_10"
                    ]
                ),
                "validation_ndcg_at_10": float(
                    default_summary[
                        "selected_default_model_metrics"
                    ][
                        "ndcg_at_10"
                    ]
                ),
            },
            {
                "model": (
                    "tuned_hybrid_warp_style"
                ),
                "validation_precision_at_10": float(
                    optuna_summary[
                        "selected_validation_metrics"
                    ][
                        "precision_at_10"
                    ]
                ),
                "validation_recall_at_10": float(
                    optuna_summary[
                        "selected_validation_metrics"
                    ][
                        "recall_at_10"
                    ]
                ),
                "validation_ndcg_at_10": float(
                    optuna_summary[
                        "selected_validation_metrics"
                    ][
                        "ndcg_at_10"
                    ]
                ),
            },
        ]
    )

    validation_test_comparison = (
        test_metrics[
            test_metrics[
                "model"
            ].isin(
                validation_metrics[
                    "model"
                ]
            )
        ]
        .merge(
            validation_metrics,
            on="model",
            how="inner",
            validate="one_to_one",
        )
    )

    validation_test_comparison[
        "precision_generalization_gap"
    ] = (
        validation_test_comparison[
            "precision_at_10"
        ]
        - validation_test_comparison[
            "validation_precision_at_10"
        ]
    )

    default_history = pd.DataFrame(
        default_run.history
    )

    default_history.insert(
        0,
        "model",
        default_specification.name,
    )

    tuned_history = pd.DataFrame(
        tuned_run.history
    )

    tuned_history.insert(
        0,
        "model",
        tuned_specification.name,
    )

    final_training_history = pd.concat(
        [
            default_history,
            tuned_history,
        ],
        ignore_index=True,
    )

    dense_item_features = torch.from_numpy(
        item_features
        .normalized_matrix
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    tuned_run.model.eval()

    with torch.no_grad():
        user_embeddings = (
            tuned_run
            .model
            .user_embedding
            .weight
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=True,
            )
        )

        raw_item_embeddings = (
            tuned_run
            .model
            .item_embedding
            .weight
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=True,
            )
        )

        effective_item_embeddings = (
            tuned_run
            .model
            .effective_item_embeddings(
                dense_item_features
            )
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=True,
            )
        )

        user_biases = (
            tuned_run
            .model
            .user_bias
            .weight
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(
                np.float32,
                copy=True,
            )
        )

        item_biases = (
            tuned_run
            .model
            .item_bias
            .weight
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(
                np.float32,
                copy=True,
            )
        )

        global_bias = (
            tuned_run
            .model
            .global_bias
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(
                np.float32,
                copy=True,
            )
        )

        metadata_projection = (
            tuned_run
            .model
            .metadata_projection
            .weight
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=True,
            )
        )

    numpy_scores = (
        user_embeddings
        @ effective_item_embeddings.T
    )

    numpy_scores = (
        numpy_scores
        + user_biases[:, None]
        + item_biases[None, :]
        + float(global_bias[0])
    )

    score_parity_max_abs_error = float(
        np.max(
            np.abs(
                numpy_scores
                - tuned_scores
            )
        )
    )

    if score_parity_max_abs_error > 1e-4:
        raise RuntimeError(
            "NumPy deployment scores do not reproduce "
            "PyTorch scores. Maximum error: "
            f"{score_parity_max_abs_error}"
        )

    (
        demo_users,
        example_frame,
    ) = create_examples(
        prepared=prepared,
        final_seen_matrix=(
            final_seen_matrix
        ),
        recommendations=(
            tuned_recommendations
        ),
    )

    test_metrics.to_csv(
        results_directory
        / "test_metrics.csv",
        index=False,
        float_format="%.12g",
    )

    validation_test_comparison.to_csv(
        results_directory
        / "validation_test_comparison.csv",
        index=False,
        float_format="%.12g",
    )

    final_training_history.to_csv(
        results_directory
        / "training_history.csv",
        index=False,
        float_format="%.12g",
    )

    example_frame.to_csv(
        results_directory
        / "recommendation_examples.csv",
        index=False,
    )

    np.savez_compressed(
        results_directory
        / "final_recommendations.npz",
        **recommendations_by_model,
    )

    torch.save(
        {
            "state_dict": (
                tuned_run.model.state_dict()
            ),
            "specification": asdict(
                tuned_specification
            ),
            "cohort_split_sha256": (
                expected_cohort_hash
            ),
        },
        results_directory
        / "final_model_state.pt",
    )

    np.savez_compressed(
        artifact_directory
        / "model_components.npz",
        user_embeddings=(
            user_embeddings
        ),
        raw_item_embeddings=(
            raw_item_embeddings
        ),
        effective_item_embeddings=(
            effective_item_embeddings
        ),
        user_biases=user_biases,
        item_biases=item_biases,
        global_bias=global_bias,
        metadata_projection=(
            metadata_projection
        ),
        metadata_weight=np.asarray(
            [
                tuned_specification
                .metadata_weight
            ],
            dtype=np.float32,
        ),
    )

    sparse.save_npz(
        artifact_directory
        / "seen_final.npz",
        final_seen_matrix,
        compressed=True,
    )

    sparse.save_npz(
        artifact_directory
        / "item_features.npz",
        item_features
        .normalized_matrix,
        compressed=True,
    )

    prepared.catalog.to_parquet(
        artifact_directory
        / "anime_catalog.parquet",
        index=False,
    )

    np.savez_compressed(
        artifact_directory
        / "final_recommendations.npz",
        recommendations=(
            tuned_recommendations
        ),
    )

    user_mapping = {
        "user_index_to_id": [
            int(value)
            for value in prepared.user_ids
        ],
        "user_id_to_index": {
            str(
                int(user_id)
            ): int(index)
            for user_id, index
            in prepared.user_to_index.items()
        },
    }

    item_mapping = {
        "item_index_to_id": [
            int(value)
            for value in prepared.item_ids
        ],
        "item_id_to_index": {
            str(
                int(item_id)
            ): int(index)
            for item_id, index
            in prepared.item_to_index.items()
        },
    }

    with (
        artifact_directory
        / "user_mapping.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            user_mapping,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    with (
        artifact_directory
        / "item_mapping.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            item_mapping,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    with (
        artifact_directory
        / "feature_names.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "feature_names": list(
                    item_features
                    .feature_names
                ),
                "genre_names": list(
                    item_features
                    .genre_names
                ),
                "type_names": list(
                    item_features
                    .type_names
                ),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    with (
        artifact_directory
        / "demo_users.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            demo_users,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

        file.write("\n")

    selected_test_metrics = (
        test_metrics[
            test_metrics[
                "model"
            ].eq(
                "tuned_hybrid_warp_style"
            )
        ]
        .iloc[0]
        .to_dict()
    )

    final_run_summary = {
        "experiment_name": config[
            "experiment_name"
        ],
        "framework": config[
            "framework"
        ],
        "python_version": (
            sys.version.split()[0]
        ),
        "platform": platform.platform(),
        "torch_version": (
            torch.__version__
        ),
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "git_commit": (
            current_git_commit()
        ),
        "cohort_split_sha256": (
            expected_cohort_hash
        ),
        "training_data": (
            "train_plus_validation"
        ),
        "training_interactions": int(
            final_seen_matrix.nnz
        ),
        "test_interactions": int(
            prepared.test_matrix.nnz
        ),
        "selected_model_before_test": (
            "tuned_hybrid_warp_style"
        ),
        "selected_parameters": (
            asdict(
                tuned_specification
            )
        ),
        "selected_test_metrics": (
            selected_test_metrics
        ),
        "score_parity_max_abs_error": (
            score_parity_max_abs_error
        ),
        "selection_changed_after_test": (
            False
        ),
        "test_set_accessed": True,
        "test_access_policy": (
            "single final evaluation phase "
            "after model and hyperparameter freeze"
        ),
    }

    with (
        results_directory
        / "run_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_run_summary,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )

        file.write("\n")

    artifact_files = {
        "model_components": (
            "model_components.npz"
        ),
        "seen_final": (
            "seen_final.npz"
        ),
        "item_features": (
            "item_features.npz"
        ),
        "anime_catalog": (
            "anime_catalog.parquet"
        ),
        "final_recommendations": (
            "final_recommendations.npz"
        ),
        "user_mapping": (
            "user_mapping.json"
        ),
        "item_mapping": (
            "item_mapping.json"
        ),
        "feature_names": (
            "feature_names.json"
        ),
        "demo_users": (
            "demo_users.json"
        ),
    }

    artifact_records = {}

    for artifact_name, file_name in (
        artifact_files.items()
    ):
        path = (
            artifact_directory
            / file_name
        )

        artifact_records[
            artifact_name
        ] = {
            "path": file_name,
            "size_bytes": int(
                path.stat().st_size
            ),
            "sha256": sha256_file(
                path
            ),
        }

    artifact_manifest = {
        "schema_version": int(
            final_policy[
                "deployment_artifact_schema_version"
            ]
        ),
        "generated_at_utc": (
            final_run_summary[
                "generated_at_utc"
            ]
        ),
        "git_commit": (
            final_run_summary[
                "git_commit"
            ]
        ),
        "deployment_requires_pytorch": (
            False
        ),
        "model_name": (
            "tuned_hybrid_warp_style"
        ),
        "objective": (
            tuned_specification.objective
        ),
        "cohort_split_sha256": (
            expected_cohort_hash
        ),
        "anime_sha256": (
            prepared.summary[
                "anime_sha256"
            ]
        ),
        "ratings_sha256": (
            prepared.summary[
                "ratings_sha256"
            ]
        ),
        "n_users": int(
            len(
                prepared.user_ids
            )
        ),
        "n_items": int(
            len(
                prepared.item_ids
            )
        ),
        "latent_dim": int(
            tuned_specification.latent_dim
        ),
        "n_item_features": int(
            item_features
            .matrix
            .shape[1]
        ),
        "seen_interactions": int(
            final_seen_matrix.nnz
        ),
        "recommendation_k": int(
            config[
                "recommendation_k"
            ]
        ),
        "score_parity_max_abs_error": (
            score_parity_max_abs_error
        ),
        "selected_parameters": (
            asdict(
                tuned_specification
            )
        ),
        "selected_test_metrics": (
            selected_test_metrics
        ),
        "files": artifact_records,
    }

    with (
        artifact_directory
        / "artifact_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact_manifest,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )

        file.write("\n")

    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "FINAL TEST EVALUATION COMPLETE",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
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
                "catalog_coverage_at_10",
                "intra_list_diversity_at_10",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print()

    print(
        "Frozen selected model:",
        "tuned_hybrid_warp_style",
        flush=True,
    )

    print(
        "NumPy score parity maximum error:",
        score_parity_max_abs_error,
        flush=True,
    )

    print(
        "Results directory:",
        results_directory,
        flush=True,
    )

    print(
        "Artifact directory:",
        artifact_directory,
        flush=True,
    )


if __name__ == "__main__":
    main()
