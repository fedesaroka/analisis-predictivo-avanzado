"""Run and verify the deterministic data pipeline twice."""

from __future__ import annotations

import json
from pathlib import Path

from anime_recommender import (
    load_experiment_config,
    prepare_data,
)


ROOT = Path(__file__).resolve().parents[1]


def matrices_are_equal(left, right) -> bool:
    """Compare two sparse matrices exactly."""

    if left.shape != right.shape:
        return False

    difference = left != right

    return difference.nnz == 0


def main() -> None:
    config = load_experiment_config(
        ROOT / "config" / "experiment.json"
    )

    arguments = {
        "anime_path": (
            ROOT / "data" / "anime.csv"
        ),
        "ratings_path": (
            ROOT / "data" / "rating.parquet"
        ),
        "config": config,
    }

    first = prepare_data(**arguments)
    second = prepare_data(**arguments)

    if (
        first.summary[
            "cohort_split_sha256"
        ]
        != second.summary[
            "cohort_split_sha256"
        ]
    ):
        raise AssertionError(
            "The cohort hash changed between executions."
        )

    matrix_pairs = [
        (
            "train",
            first.train_matrix,
            second.train_matrix,
        ),
        (
            "validation",
            first.validation_matrix,
            second.validation_matrix,
        ),
        (
            "test",
            first.test_matrix,
            second.test_matrix,
        ),
        (
            "all positives",
            first.all_positive_matrix,
            second.all_positive_matrix,
        ),
    ]

    for name, first_matrix, second_matrix in (
        matrix_pairs
    ):
        if not matrices_are_equal(
            first_matrix,
            second_matrix,
        ):
            raise AssertionError(
                f"The {name} matrix changed "
                "between executions."
            )

    if first.user_to_index != (
        second.user_to_index
    ):
        raise AssertionError(
            "User mappings changed between executions."
        )

    if first.item_to_index != (
        second.item_to_index
    ):
        raise AssertionError(
            "Item mappings changed between executions."
        )

    print(
        json.dumps(
            first.summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Deterministic data pipeline validation passed."
    )


if __name__ == "__main__":
    main()
