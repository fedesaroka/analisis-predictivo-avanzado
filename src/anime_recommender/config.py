"""Experiment configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "experiment_name",
    "framework",
    "seed",
    "positive_rating_threshold",
    "target_user_count",
    "minimum_user_positives_before_sampling",
    "maximum_user_positives_before_sampling",
    "minimum_user_positives_after_filtering",
    "minimum_item_users",
    "validation_fraction",
    "test_fraction",
    "recommendation_k",
    "optuna_trials",
    "device",
    "candidate_objectives",
    "item_metadata",
    "primary_metric",
}


def load_experiment_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the central experiment configuration."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = json.load(file)

    missing_keys = sorted(
        REQUIRED_KEYS.difference(config)
    )

    if missing_keys:
        raise ValueError(
            "The experiment configuration is missing keys: "
            + ", ".join(missing_keys)
        )

    seed = int(config["seed"])
    positive_threshold = int(
        config["positive_rating_threshold"]
    )
    target_user_count = int(
        config["target_user_count"]
    )
    minimum_before = int(
        config[
            "minimum_user_positives_before_sampling"
        ]
    )
    maximum_before = int(
        config[
            "maximum_user_positives_before_sampling"
        ]
    )
    minimum_after = int(
        config[
            "minimum_user_positives_after_filtering"
        ]
    )
    minimum_item_users = int(
        config["minimum_item_users"]
    )
    validation_fraction = float(
        config["validation_fraction"]
    )
    test_fraction = float(
        config["test_fraction"]
    )

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    if not 1 <= positive_threshold <= 10:
        raise ValueError(
            "positive_rating_threshold must be between 1 and 10."
        )

    if target_user_count <= 0:
        raise ValueError(
            "target_user_count must be positive."
        )

    if minimum_before <= 0:
        raise ValueError(
            "minimum_user_positives_before_sampling "
            "must be positive."
        )

    if maximum_before < minimum_before:
        raise ValueError(
            "maximum_user_positives_before_sampling "
            "must be at least the minimum."
        )

    if minimum_after < 3:
        raise ValueError(
            "minimum_user_positives_after_filtering "
            "must be at least 3."
        )

    if minimum_after > minimum_before:
        raise ValueError(
            "The post-filter minimum cannot exceed "
            "the pre-sampling minimum."
        )

    if minimum_item_users < 2:
        raise ValueError(
            "minimum_item_users must be at least 2."
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must lie between 0 and 1."
        )

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must lie between 0 and 1."
        )

    if validation_fraction + test_fraction >= 1:
        raise ValueError(
            "Validation and test fractions must sum to less than 1."
        )

    return config
