"""Train and evaluate the six default PyTorch recommenders."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


# Set single-threaded numerical execution before importing PyTorch.
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
)


# PyTorch must be initialized before pandas and other libraries that
# may load native numerical runtimes on Windows.
import torch

import numpy as np
import pandas as pd

from anime_recommender import (
    build_item_features,
    evaluate_baselines,
    evaluate_score_matrix,
    load_experiment_config,
    prepare_data,
    recommendation_examples,
)
from anime_recommender.training import (
    TrainingSpec,
    score_all_items,
    train_recommender,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODEL_ORDER = [
    "popularity",
    "content",
    "collaborative_logistic",
    "collaborative_bpr",
    "collaborative_warp_style",
    "hybrid_logistic",
    "hybrid_bpr",
    "hybrid_warp_style",
]


MODEL_SPECIFICATIONS = [
    {
        "name": "collaborative_logistic",
        "objective": "logistic",
        "use_metadata": False,
    },
    {
        "name": "collaborative_bpr",
        "objective": "bpr",
        "use_metadata": False,
    },
    {
        "name": "collaborative_warp_style",
        "objective": "warp_style_hard_negative",
        "use_metadata": False,
    },
    {
        "name": "hybrid_logistic",
        "objective": "logistic",
        "use_metadata": True,
    },
    {
        "name": "hybrid_bpr",
        "objective": "bpr",
        "use_metadata": True,
    },
    {
        "name": "hybrid_warp_style",
        "objective": "warp_style_hard_negative",
        "use_metadata": True,
    },
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "default_models"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def json_default(
    value: Any,
) -> Any:
    """Convert NumPy values to JSON-compatible Python values."""

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    raise TypeError(
        f"Cannot serialize object of type {type(value)}"
    )


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Create one fresh output directory."""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to recreate it."
            )

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


def build_training_spec(
    model_specification: dict[str, Any],
    config: dict[str, Any],
) -> TrainingSpec:
    """Combine model identity with shared default hyperparameters."""

    defaults = config[
        "default_training"
    ]

    return TrainingSpec(
        name=str(
            model_specification["name"]
        ),
        objective=str(
            model_specification[
                "objective"
            ]
        ),
        use_metadata=bool(
            model_specification[
                "use_metadata"
            ]
        ),
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


def main() -> None:
    """Run all default models and export validation results."""

    arguments = parse_arguments()

    output_dir = (
        arguments.output_dir.resolve()
    )

    prepare_output_directory(
        output_dir=output_dir,
        overwrite=arguments.overwrite,
    )

    config = load_experiment_config(
        ROOT
        / "config"
        / "experiment.json"
    )

    print(
        "Preparing deterministic cohort...",
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

    print(
        "Building genre/type metadata...",
        flush=True,
    )

    item_features = build_item_features(
        prepared.catalog
    )

    print(
        "Evaluating baselines...",
        flush=True,
    )

    (
        baseline_results,
        recommendations_by_model,
    ) = evaluate_baselines(
        prepared=prepared,
        item_features=item_features,
        config=config,
    )

    metric_rows: list[
        dict[str, Any]
    ] = []

    training_history_rows: list[
        dict[str, Any]
    ] = []

    for baseline_row in baseline_results.to_dict(
        orient="records"
    ):
        model_name = str(
            baseline_row["model"]
        )

        metric_rows.append(
            {
                "model": model_name,
                "model_family": "baseline",
                "objective": model_name,
                "uses_metadata": (
                    model_name == "content"
                ),
                "latent_dim": 0,
                "epochs": 0,
                "final_train_loss": np.nan,
                "parameter_count": 0,
                **{
                    key: value
                    for key, value
                    in baseline_row.items()
                    if key != "model"
                },
            }
        )

    for model_specification in (
        MODEL_SPECIFICATIONS
    ):
        training_spec = build_training_spec(
            model_specification=(
                model_specification
            ),
            config=config,
        )

        print()
        print(
            "=" * 72,
            flush=True,
        )
        print(
            f"Training {training_spec.name}",
            flush=True,
        )
        print(
            f"Objective: {training_spec.objective}",
            flush=True,
        )
        print(
            f"Uses metadata: "
            f"{training_spec.use_metadata}",
            flush=True,
        )
        print(
            "=" * 72,
            flush=True,
        )

        training_run = train_recommender(
            train_matrix=(
                prepared.train_matrix
            ),
            item_feature_matrix=(
                item_features
                .normalized_matrix
            ),
            specification=training_spec,
            verbose=True,
        )

        print(
            f"Scoring complete catalog for "
            f"{training_spec.name}...",
            flush=True,
        )

        score_matrix = score_all_items(
            model=training_run.model,
            item_feature_matrix=(
                item_features
                .normalized_matrix
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
            validation_metrics,
            recommendations,
        ) = evaluate_score_matrix(
            scores=score_matrix,
            seen_matrix=(
                prepared.train_matrix
            ),
            target_matrix=(
                prepared.validation_matrix
            ),
            normalized_item_features=(
                item_features
                .normalized_matrix
            ),
            k=int(
                config[
                    "recommendation_k"
                ]
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

        recommendations_by_model[
            training_spec.name
        ] = recommendations

        final_train_loss = float(
            training_run.history[
                -1
            ]["loss"]
        )

        metric_row = {
            "model": training_spec.name,
            "model_family": "pytorch",
            "objective": (
                training_spec.objective
            ),
            "uses_metadata": (
                training_spec.use_metadata
            ),
            "latent_dim": (
                training_spec.latent_dim
            ),
            "epochs": (
                training_spec.epochs
            ),
            "final_train_loss": (
                final_train_loss
            ),
            "parameter_count": (
                training_run.parameter_count
            ),
            **validation_metrics,
        }

        metric_rows.append(
            metric_row
        )

        for history_row in (
            training_run.history
        ):
            training_history_rows.append(
                {
                    "model": (
                        training_spec.name
                    ),
                    "objective": (
                        training_spec.objective
                    ),
                    "uses_metadata": (
                        training_spec.use_metadata
                    ),
                    **history_row,
                }
            )

        print(
            f"Validation Precision@10: "
            f"{validation_metrics['precision_at_10']:.6f}",
            flush=True,
        )

        print(
            f"Validation NDCG@10: "
            f"{validation_metrics['ndcg_at_10']:.6f}",
            flush=True,
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    model_order = {
        model_name: order
        for order, model_name
        in enumerate(
            EXPECTED_MODEL_ORDER
        )
    }

    metrics["_model_order"] = (
        metrics["model"]
        .map(model_order)
    )

    if metrics[
        "_model_order"
    ].isna().any():
        unknown_models = metrics.loc[
            metrics[
                "_model_order"
            ].isna(),
            "model",
        ].tolist()

        raise RuntimeError(
            "Unknown models were generated: "
            f"{unknown_models}"
        )

    metrics = (
        metrics
        .sort_values(
            "_model_order"
        )
        .drop(
            columns="_model_order"
        )
        .reset_index(drop=True)
    )

    training_history = pd.DataFrame(
        training_history_rows
    )

    neural_metrics = metrics[
        metrics[
            "model_family"
        ].eq("pytorch")
    ].copy()

    selected_row = (
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
        .iloc[0]
    )

    selected_default_model = str(
        selected_row["model"]
    )

    overall_best_row = (
        metrics
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
        .iloc[0]
    )

    selected_user_indices = [
        0,
        len(
            prepared.user_ids
        )
        // 2,
        len(
            prepared.user_ids
        )
        - 1,
    ]

    examples = recommendation_examples(
        prepared=prepared,
        recommendations_by_model=(
            recommendations_by_model
        ),
        user_indices=(
            selected_user_indices
        ),
    )

    metrics.to_csv(
        output_dir
        / "validation_metrics.csv",
        index=False,
        float_format="%.10f",
    )

    training_history.to_csv(
        output_dir
        / "training_history.csv",
        index=False,
        float_format="%.10f",
    )

    examples.to_csv(
        output_dir
        / "recommendation_examples.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir
        / "recommendations.npz",
        **recommendations_by_model,
    )

    run_summary = {
        "experiment_name": (
            config["experiment_name"]
        ),
        "framework": (
            config["framework"]
        ),
        "python_version": (
            sys.version.split()[0]
        ),
        "platform": platform.platform(),
        "torch_version": (
            torch.__version__
        ),
        "numpy_version": (
            np.__version__
        ),
        "pandas_version": (
            pd.__version__
        ),
        "cohort_split_sha256": (
            prepared.summary[
                "cohort_split_sha256"
            ]
        ),
        "final_users": int(
            prepared.summary[
                "final_users"
            ]
        ),
        "final_items": int(
            prepared.summary[
                "final_items"
            ]
        ),
        "train_interactions": int(
            prepared.summary[
                "train_interactions"
            ]
        ),
        "validation_interactions": int(
            prepared.summary[
                "validation_interactions"
            ]
        ),
        "item_feature_summary": (
            item_features.summary
        ),
        "default_training": (
            config["default_training"]
        ),
        "model_specifications": (
            MODEL_SPECIFICATIONS
        ),
        "selected_default_model": (
            selected_default_model
        ),
        "selected_default_model_metrics": {
            key: value
            for key, value
            in selected_row.to_dict().items()
            if key not in {
                "final_train_loss",
            }
        },
        "overall_best_validation_model": str(
            overall_best_row[
                "model"
            ]
        ),
        "recommendation_k": int(
            config[
                "recommendation_k"
            ]
        ),
        "test_set_accessed": False,
    }

    with (
        output_dir
        / "run_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_summary,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "DEFAULT MODEL COMPARISON COMPLETE",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )

    print(
        metrics[
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
        "Selected default neural model:",
        selected_default_model,
        flush=True,
    )

    print(
        "Results written to:",
        output_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()
