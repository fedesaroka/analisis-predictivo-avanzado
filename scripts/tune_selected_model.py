"""Tune the selected default neural recommender with Optuna TPE."""

from __future__ import annotations

import os


# Keep native numerical execution deterministic and avoid runtime
# conflicts on Windows before importing PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# PyTorch must initialize before pandas, SciPy, and Optuna-related
# dependencies that may load native numerical libraries.
import torch

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.trial import TrialState

from anime_recommender import (
    build_item_features,
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

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
    / "default_models"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "optuna"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def parse_boolean(value: Any) -> bool:
    """Convert common CSV Boolean representations safely."""

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    normalized = str(value).strip().casefold()

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot interpret Boolean value: {value!r}"
    )


def json_default(value: Any) -> Any:
    """Convert NumPy values to JSON-compatible values."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    raise TypeError(
        f"Cannot serialize object of type {type(value)}"
    )


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Create a clean result directory."""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to recreate it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


def validate_search_configuration(
    config: dict[str, Any],
) -> None:
    """Validate the complete Optuna search configuration."""

    if "optuna_search" not in config:
        raise ValueError(
            "experiment.json does not contain optuna_search."
        )

    search = config["optuna_search"]

    required_keys = {
        "sampler",
        "n_startup_trials",
        "latent_dim_choices",
        "epochs_choices",
        "batch_size_choices",
        "learning_rate_min",
        "learning_rate_max",
        "weight_decay_min",
        "weight_decay_max",
        "metadata_weight_min",
        "metadata_weight_max",
        "metadata_weight_step",
        "negatives_per_positive_choices",
        "max_sampled_choices",
        "margin_min",
        "margin_max",
        "margin_step",
    }

    missing_keys = sorted(
        required_keys.difference(search)
    )

    if missing_keys:
        raise ValueError(
            "optuna_search is missing keys: "
            + ", ".join(missing_keys)
        )

    if str(search["sampler"]).casefold() != "tpe":
        raise ValueError(
            "This phase requires the TPE sampler."
        )

    if int(config["optuna_trials"]) <= 0:
        raise ValueError(
            "optuna_trials must be positive."
        )

    if int(search["n_startup_trials"]) <= 0:
        raise ValueError(
            "n_startup_trials must be positive."
        )

    choice_keys = [
        "latent_dim_choices",
        "epochs_choices",
        "batch_size_choices",
        "negatives_per_positive_choices",
        "max_sampled_choices",
    ]

    for key in choice_keys:
        values = search[key]

        if not values:
            raise ValueError(
                f"{key} cannot be empty."
            )

        if any(int(value) <= 0 for value in values):
            raise ValueError(
                f"{key} must contain positive integers."
            )

    bounded_ranges = [
        (
            "learning_rate_min",
            "learning_rate_max",
        ),
        (
            "weight_decay_min",
            "weight_decay_max",
        ),
        (
            "metadata_weight_min",
            "metadata_weight_max",
        ),
        (
            "margin_min",
            "margin_max",
        ),
    ]

    for minimum_key, maximum_key in bounded_ranges:
        minimum = float(search[minimum_key])
        maximum = float(search[maximum_key])

        if minimum <= 0:
            raise ValueError(
                f"{minimum_key} must be positive."
            )

        if maximum < minimum:
            raise ValueError(
                f"{maximum_key} must be at least {minimum_key}."
            )


def load_default_selection() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.Series,
]:
    """Load and validate the Phase 4 selected model."""

    summary_path = (
        DEFAULT_RESULTS_DIR
        / "run_summary.json"
    )

    metrics_path = (
        DEFAULT_RESULTS_DIR
        / "validation_metrics.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing default run summary: {summary_path}"
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing default metrics: {metrics_path}"
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    metrics = pd.read_csv(metrics_path)

    selected_model = str(
        summary["selected_default_model"]
    )

    matching_rows = metrics[
        metrics["model"].eq(selected_model)
    ]

    if len(matching_rows) != 1:
        raise RuntimeError(
            "The selected default model could not be resolved "
            "to exactly one metrics row."
        )

    selected_row = matching_rows.iloc[0]

    if str(
        selected_row["model_family"]
    ) != "pytorch":
        raise RuntimeError(
            "The selected model is not a PyTorch model."
        )

    if bool(summary["test_set_accessed"]):
        raise RuntimeError(
            "Phase 4 unexpectedly accessed the test set."
        )

    return (
        summary,
        metrics,
        selected_row,
    )


def default_trial_parameters(
    config: dict[str, Any],
    objective: str,
    use_metadata: bool,
) -> dict[str, Any]:
    """Create trial zero from the default Phase 4 configuration."""

    defaults = config["default_training"]

    parameters: dict[str, Any] = {
        "latent_dim": int(
            defaults["latent_dim"]
        ),
        "epochs": int(
            defaults["epochs"]
        ),
        "batch_size": int(
            defaults["batch_size"]
        ),
        "learning_rate": float(
            defaults["learning_rate"]
        ),
        "weight_decay": float(
            defaults["weight_decay"]
        ),
    }

    if use_metadata:
        parameters["metadata_weight"] = float(
            defaults["metadata_weight"]
        )

    if objective in {
        "logistic",
        "bpr",
    }:
        parameters["negatives_per_positive"] = int(
            defaults["negatives_per_positive"]
        )

    if objective == "warp_style_hard_negative":
        parameters["max_sampled"] = int(
            defaults["max_sampled"]
        )

        parameters["margin"] = float(
            defaults["margin"]
        )

    return parameters


def suggest_parameters(
    trial: optuna.Trial,
    config: dict[str, Any],
    objective: str,
    use_metadata: bool,
) -> dict[str, Any]:
    """Sample one conditional hyperparameter configuration."""

    search = config["optuna_search"]

    parameters: dict[str, Any] = {
        "latent_dim": trial.suggest_categorical(
            "latent_dim",
            [
                int(value)
                for value
                in search["latent_dim_choices"]
            ],
        ),
        "epochs": trial.suggest_categorical(
            "epochs",
            [
                int(value)
                for value
                in search["epochs_choices"]
            ],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [
                int(value)
                for value
                in search["batch_size_choices"]
            ],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(search["learning_rate_min"]),
            float(search["learning_rate_max"]),
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            float(search["weight_decay_min"]),
            float(search["weight_decay_max"]),
            log=True,
        ),
    }

    if use_metadata:
        parameters[
            "metadata_weight"
        ] = trial.suggest_float(
            "metadata_weight",
            float(search["metadata_weight_min"]),
            float(search["metadata_weight_max"]),
            step=float(
                search["metadata_weight_step"]
            ),
        )

    if objective in {
        "logistic",
        "bpr",
    }:
        parameters[
            "negatives_per_positive"
        ] = trial.suggest_categorical(
            "negatives_per_positive",
            [
                int(value)
                for value
                in search[
                    "negatives_per_positive_choices"
                ]
            ],
        )

    if objective == "warp_style_hard_negative":
        parameters[
            "max_sampled"
        ] = trial.suggest_categorical(
            "max_sampled",
            [
                int(value)
                for value
                in search["max_sampled_choices"]
            ],
        )

        parameters["margin"] = trial.suggest_float(
            "margin",
            float(search["margin_min"]),
            float(search["margin_max"]),
            step=float(search["margin_step"]),
        )

    return parameters


def build_training_specification(
    name: str,
    parameters: dict[str, Any],
    config: dict[str, Any],
    objective: str,
    use_metadata: bool,
) -> TrainingSpec:
    """Resolve sampled and fixed parameters into TrainingSpec."""

    defaults = config["default_training"]

    return TrainingSpec(
        name=name,
        objective=objective,
        use_metadata=use_metadata,
        latent_dim=int(
            parameters["latent_dim"]
        ),
        epochs=int(
            parameters["epochs"]
        ),
        batch_size=int(
            parameters["batch_size"]
        ),
        learning_rate=float(
            parameters["learning_rate"]
        ),
        weight_decay=float(
            parameters["weight_decay"]
        ),
        metadata_weight=float(
            parameters.get(
                "metadata_weight",
                defaults["metadata_weight"],
            )
        ),
        negatives_per_positive=int(
            parameters.get(
                "negatives_per_positive",
                defaults["negatives_per_positive"],
            )
        ),
        max_sampled=int(
            parameters.get(
                "max_sampled",
                defaults["max_sampled"],
            )
        ),
        margin=float(
            parameters.get(
                "margin",
                defaults["margin"],
            )
        ),
        gradient_clip_norm=float(
            defaults["gradient_clip_norm"]
        ),
        seed=int(config["seed"]),
    )


def select_best_trial(
    completed_trials: list[
        optuna.trial.FrozenTrial
    ],
) -> optuna.trial.FrozenTrial:
    """Select using Precision@10 and deterministic tie-breakers."""

    if not completed_trials:
        raise RuntimeError(
            "No completed Optuna trials are available."
        )

    return sorted(
        completed_trials,
        key=lambda trial: (
            -float(trial.value),
            -float(
                trial.user_attrs["ndcg_at_10"]
            ),
            -float(
                trial.user_attrs["hit_rate_at_10"]
            ),
            int(trial.number),
        ),
    )[0]


def trials_to_dataframe(
    study: optuna.Study,
) -> pd.DataFrame:
    """Convert trials, parameters, and metrics into one table."""

    parameter_names = sorted(
        {
            parameter_name
            for trial in study.trials
            for parameter_name in trial.params
        }
    )

    user_attribute_names = sorted(
        {
            attribute_name
            for trial in study.trials
            for attribute_name in trial.user_attrs
        }
    )

    rows: list[dict[str, Any]] = []

    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "objective_value": (
                None
                if trial.value is None
                else float(trial.value)
            ),
        }

        for parameter_name in parameter_names:
            row[parameter_name] = trial.params.get(
                parameter_name
            )

        for attribute_name in user_attribute_names:
            row[attribute_name] = trial.user_attrs.get(
                attribute_name
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("trial_number")
        .reset_index(drop=True)
    )


def main() -> None:
    """Execute Bayesian optimization and export selected results."""

    arguments = parse_arguments()

    output_dir = arguments.output_dir.resolve()

    prepare_output_directory(
        output_dir=output_dir,
        overwrite=arguments.overwrite,
    )

    config = load_experiment_config(
        ROOT
        / "config"
        / "experiment.json"
    )

    validate_search_configuration(config)

    (
        default_summary,
        default_metrics,
        selected_default_row,
    ) = load_default_selection()

    selected_default_model = str(
        default_summary[
            "selected_default_model"
        ]
    )

    objective_name = str(
        selected_default_row["objective"]
    )

    use_metadata = parse_boolean(
        selected_default_row["uses_metadata"]
    )

    print(
        "Selected Phase 4 model:",
        selected_default_model,
        flush=True,
    )

    print(
        "Objective:",
        objective_name,
        flush=True,
    )

    print(
        "Uses metadata:",
        use_metadata,
        flush=True,
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

    if (
        prepared.summary[
            "cohort_split_sha256"
        ]
        != default_summary[
            "cohort_split_sha256"
        ]
    ):
        raise RuntimeError(
            "The current cohort differs from the Phase 4 cohort."
        )

    print(
        "Building deterministic item metadata...",
        flush=True,
    )

    item_features = build_item_features(
        prepared.catalog
    )

    sampler = optuna.samplers.TPESampler(
        seed=int(config["seed"]),
        n_startup_trials=int(
            config["optuna_search"][
                "n_startup_trials"
            ]
        ),
    )

    study = optuna.create_study(
        study_name=(
            "anime_selected_model_precision_at_10"
        ),
        direction="maximize",
        sampler=sampler,
    )

    default_parameters = default_trial_parameters(
        config=config,
        objective=objective_name,
        use_metadata=use_metadata,
    )

    # Trial zero reproduces the selected Phase 4 configuration.
    # This guarantees the search contains the existing benchmark.
    study.enqueue_trial(
        default_parameters
    )

    def objective_function(
        trial: optuna.Trial,
    ) -> float:
        """Train and evaluate one validation-only trial."""

        parameters = suggest_parameters(
            trial=trial,
            config=config,
            objective=objective_name,
            use_metadata=use_metadata,
        )

        specification = build_training_specification(
            name=(
                f"optuna_trial_{trial.number:03d}"
            ),
            parameters=parameters,
            config=config,
            objective=objective_name,
            use_metadata=use_metadata,
        )

        print()
        print(
            "=" * 72,
            flush=True,
        )

        print(
            f"OPTUNA TRIAL {trial.number}",
            flush=True,
        )

        print(
            json.dumps(
                parameters,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        started = perf_counter()

        training_run = train_recommender(
            train_matrix=prepared.train_matrix,
            item_feature_matrix=(
                item_features.normalized_matrix
            ),
            specification=specification,
            verbose=True,
        )

        score_matrix = score_all_items(
            model=training_run.model,
            item_feature_matrix=(
                item_features.normalized_matrix
            ),
            n_users=len(prepared.user_ids),
            user_batch_size=int(
                config["default_training"][
                    "scoring_user_batch_size"
                ]
            ),
        )

        (
            validation_metrics,
            recommendations,
        ) = evaluate_score_matrix(
            scores=score_matrix,
            seen_matrix=prepared.train_matrix,
            target_matrix=(
                prepared.validation_matrix
            ),
            normalized_item_features=(
                item_features.normalized_matrix
            ),
            k=int(config["recommendation_k"]),
            auc_negatives=int(
                config["sampled_auc_negatives"]
            ),
            seed=int(config["seed"]),
        )

        elapsed_seconds = (
            perf_counter()
            - started
        )

        for metric_name, metric_value in (
            validation_metrics.items()
        ):
            trial.set_user_attr(
                metric_name,
                float(metric_value),
            )

        trial.set_user_attr(
            "final_train_loss",
            float(
                training_run.history[-1]["loss"]
            ),
        )

        trial.set_user_attr(
            "parameter_count",
            int(training_run.parameter_count),
        )

        trial.set_user_attr(
            "duration_seconds",
            float(elapsed_seconds),
        )

        trial.set_user_attr(
            "is_default_trial",
            bool(trial.number == 0),
        )

        trial.set_user_attr(
            "recommendation_shape",
            list(recommendations.shape),
        )

        precision = float(
            validation_metrics[
                "precision_at_10"
            ]
        )

        print(
            f"Trial {trial.number} "
            f"Precision@10={precision:.10f}",
            flush=True,
        )

        print(
            f"Trial {trial.number} "
            f"NDCG@10="
            f"{validation_metrics['ndcg_at_10']:.10f}",
            flush=True,
        )

        print(
            f"Duration: {elapsed_seconds:.1f} seconds",
            flush=True,
        )

        return precision

    study.optimize(
        objective_function,
        n_trials=int(config["optuna_trials"]),
        n_jobs=1,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]

    expected_trial_count = int(
        config["optuna_trials"]
    )

    if len(completed_trials) != expected_trial_count:
        raise RuntimeError(
            f"Expected {expected_trial_count} completed trials, "
            f"found {len(completed_trials)}."
        )

    trial_zero = study.trials[0]

    default_precision = float(
        selected_default_row[
            "precision_at_10"
        ]
    )

    trial_zero_precision = float(
        trial_zero.user_attrs[
            "precision_at_10"
        ]
    )

    if not np.isclose(
        trial_zero_precision,
        default_precision,
        rtol=0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Trial zero did not reproduce the Phase 4 "
            "default validation Precision@10."
        )

    selected_trial = select_best_trial(
        completed_trials
    )

    selected_parameters = dict(
        selected_trial.params
    )

    selected_model_name = (
        f"tuned_{selected_default_model}"
    )

    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "RETRAINING SELECTED CONFIGURATION",
        flush=True,
    )

    print(
        f"Selected trial: {selected_trial.number}",
        flush=True,
    )

    print(
        json.dumps(
            selected_parameters,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    selected_specification = (
        build_training_specification(
            name=selected_model_name,
            parameters=selected_parameters,
            config=config,
            objective=objective_name,
            use_metadata=use_metadata,
        )
    )

    selected_training_run = train_recommender(
        train_matrix=prepared.train_matrix,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        specification=selected_specification,
        verbose=True,
    )

    selected_score_matrix = score_all_items(
        model=selected_training_run.model,
        item_feature_matrix=(
            item_features.normalized_matrix
        ),
        n_users=len(prepared.user_ids),
        user_batch_size=int(
            config["default_training"][
                "scoring_user_batch_size"
            ]
        ),
    )

    (
        selected_validation_metrics,
        selected_recommendations,
    ) = evaluate_score_matrix(
        scores=selected_score_matrix,
        seen_matrix=prepared.train_matrix,
        target_matrix=prepared.validation_matrix,
        normalized_item_features=(
            item_features.normalized_matrix
        ),
        k=int(config["recommendation_k"]),
        auc_negatives=int(
            config["sampled_auc_negatives"]
        ),
        seed=int(config["seed"]),
    )

    selected_trial_precision = float(
        selected_trial.value
    )

    retrained_precision = float(
        selected_validation_metrics[
            "precision_at_10"
        ]
    )

    if not np.isclose(
        selected_trial_precision,
        retrained_precision,
        rtol=0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "The selected trial could not be reproduced "
            "during clean retraining."
        )

    if (
        retrained_precision
        + 1e-12
        < default_precision
    ):
        raise RuntimeError(
            "The selected tuned model performs worse than "
            "the included default trial."
        )

    trial_table = trials_to_dataframe(
        study
    )

    importance_mapping = (
        get_param_importances(
            study
        )
    )

    parameter_importances = pd.DataFrame(
        [
            {
                "parameter": parameter,
                "importance": float(importance),
            }
            for parameter, importance
            in importance_mapping.items()
        ]
    )

    parameter_importances = (
        parameter_importances
        .sort_values(
            [
                "importance",
                "parameter",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    selected_history = pd.DataFrame(
        selected_training_run.history
    )

    selected_history.insert(
        0,
        "model",
        selected_model_name,
    )

    selected_metrics_row = {
        "model": selected_model_name,
        "source_model": selected_default_model,
        "objective": objective_name,
        "uses_metadata": use_metadata,
        "selected_trial_number": int(
            selected_trial.number
        ),
        "parameter_count": int(
            selected_training_run.parameter_count
        ),
        "final_train_loss": float(
            selected_training_run.history[
                -1
            ]["loss"]
        ),
        **selected_validation_metrics,
    }

    selected_metrics = pd.DataFrame(
        [selected_metrics_row]
    )

    default_comparison_row = {
        "model": selected_default_model,
        "variant": "default",
        **{
            metric_name: float(
                selected_default_row[
                    metric_name
                ]
            )
            for metric_name in [
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "catalog_coverage_at_10",
                "intra_list_diversity_at_10",
                "sampled_auc",
            ]
        },
    }

    tuned_comparison_row = {
        "model": selected_model_name,
        "variant": "tuned",
        **{
            metric_name: float(
                selected_validation_metrics[
                    metric_name
                ]
            )
            for metric_name in [
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "catalog_coverage_at_10",
                "intra_list_diversity_at_10",
                "sampled_auc",
            ]
        },
    }

    validation_comparison = pd.DataFrame(
        [
            default_comparison_row,
            tuned_comparison_row,
        ]
    )

    validation_comparison[
        "precision_lift_vs_default"
    ] = (
        validation_comparison[
            "precision_at_10"
        ]
        - default_precision
    )

    selected_user_indices = [
        0,
        len(prepared.user_ids) // 2,
        len(prepared.user_ids) - 1,
    ]

    examples = recommendation_examples(
        prepared=prepared,
        recommendations_by_model={
            selected_model_name: (
                selected_recommendations
            )
        },
        user_indices=selected_user_indices,
    )

    resolved_best_parameters = {
        "source_model": selected_default_model,
        "objective": objective_name,
        "use_metadata": use_metadata,
        **selected_parameters,
        "gradient_clip_norm": float(
            config["default_training"][
                "gradient_clip_norm"
            ]
        ),
        "scoring_user_batch_size": int(
            config["default_training"][
                "scoring_user_batch_size"
            ]
        ),
        "seed": int(config["seed"]),
    }

    trial_table.to_csv(
        output_dir
        / "trials.csv",
        index=False,
        float_format="%.12g",
    )

    parameter_importances.to_csv(
        output_dir
        / "parameter_importances.csv",
        index=False,
        float_format="%.12g",
    )

    selected_history.to_csv(
        output_dir
        / "best_training_history.csv",
        index=False,
        float_format="%.12g",
    )

    selected_metrics.to_csv(
        output_dir
        / "best_validation_metrics.csv",
        index=False,
        float_format="%.12g",
    )

    validation_comparison.to_csv(
        output_dir
        / "validation_comparison.csv",
        index=False,
        float_format="%.12g",
    )

    examples.to_csv(
        output_dir
        / "recommendation_examples.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir
        / "best_recommendations.npz",
        **{
            selected_model_name: (
                selected_recommendations
            )
        },
    )

    with (
        output_dir
        / "best_params.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            resolved_best_parameters,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )

        file.write("\n")

    run_summary = {
        "experiment_name": config[
            "experiment_name"
        ],
        "framework": config["framework"],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "optuna_version": optuna.__version__,
        "sampler": "TPESampler",
        "sampler_seed": int(config["seed"]),
        "n_startup_trials": int(
            config["optuna_search"][
                "n_startup_trials"
            ]
        ),
        "requested_trials": int(
            config["optuna_trials"]
        ),
        "completed_trials": int(
            len(completed_trials)
        ),
        "study_direction": "maximize",
        "primary_metric": "precision_at_10",
        "tie_breakers": [
            "ndcg_at_10",
            "hit_rate_at_10",
            "trial_number"
        ],
        "cohort_split_sha256": prepared.summary[
            "cohort_split_sha256"
        ],
        "source_default_model": (
            selected_default_model
        ),
        "source_objective": objective_name,
        "source_uses_metadata": use_metadata,
        "default_trial_parameters": (
            default_parameters
        ),
        "default_validation_precision_at_10": (
            default_precision
        ),
        "selected_trial_number": int(
            selected_trial.number
        ),
        "selected_model_name": (
            selected_model_name
        ),
        "selected_parameters": (
            resolved_best_parameters
        ),
        "selected_validation_metrics": (
            selected_validation_metrics
        ),
        "precision_lift_vs_default": float(
            retrained_precision
            - default_precision
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

        file.write("\n")

    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "OPTUNA SEARCH COMPLETE",
        flush=True,
    )

    print(
        "Selected trial:",
        selected_trial.number,
        flush=True,
    )

    print(
        "Default Precision@10:",
        f"{default_precision:.10f}",
        flush=True,
    )

    print(
        "Tuned Precision@10:",
        f"{retrained_precision:.10f}",
        flush=True,
    )

    print(
        "Precision lift:",
        f"{retrained_precision - default_precision:.10f}",
        flush=True,
    )

    print(
        "Results written to:",
        output_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()
