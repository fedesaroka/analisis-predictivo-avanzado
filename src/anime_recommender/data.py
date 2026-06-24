"""Deterministic data preparation for the anime recommender."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


CATALOG_COLUMNS = {
    "anime_id",
    "name",
    "genre",
    "type",
}

RATING_COLUMNS = {
    "user_id",
    "anime_id",
    "rating",
}


@dataclass(frozen=True)
class PreparedData:
    """All deterministic outputs of data preparation."""

    catalog: pd.DataFrame
    interactions: pd.DataFrame
    split_interactions: pd.DataFrame

    user_ids: np.ndarray
    item_ids: np.ndarray

    user_to_index: dict[int, int]
    item_to_index: dict[int, int]

    train_matrix: sparse.csr_matrix
    validation_matrix: sparse.csr_matrix
    test_matrix: sparse.csr_matrix
    all_positive_matrix: sparse.csr_matrix

    summary: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    """Calculate a SHA-256 hash without loading a file at once."""

    file_path = Path(path)
    digest = sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """Fail explicitly when an expected input column is absent."""

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def _load_catalog(
    path: str | Path,
) -> tuple[pd.DataFrame, int]:
    """Load and normalize the anime catalog."""

    catalog_path = Path(path)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Anime catalog not found: {catalog_path}"
        )

    raw = pd.read_csv(catalog_path)
    raw_row_count = len(raw)

    _require_columns(
        raw,
        CATALOG_COLUMNS,
        "anime.csv",
    )

    catalog = raw.copy()

    catalog["anime_id"] = pd.to_numeric(
        catalog["anime_id"],
        errors="coerce",
    )

    catalog = catalog.dropna(
        subset=["anime_id"]
    ).copy()

    catalog["anime_id"] = (
        catalog["anime_id"]
        .astype(np.int64)
    )

    if catalog["anime_id"].duplicated().any():
        duplicate_ids = (
            catalog.loc[
                catalog["anime_id"].duplicated(
                    keep=False
                ),
                "anime_id",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "anime.csv contains duplicate anime_id values. "
            f"Examples: {duplicate_ids[:10]}"
        )

    catalog["name"] = (
        catalog["name"]
        .fillna("Unknown title")
        .astype(str)
        .map(unescape)
        .str.strip()
    )

    catalog["genre"] = (
        catalog["genre"]
        .fillna("Unknown")
        .astype(str)
        .map(unescape)
        .str.strip()
        .replace("", "Unknown")
    )

    catalog["type"] = (
        catalog["type"]
        .fillna("Unknown")
        .astype(str)
        .map(unescape)
        .str.strip()
        .replace("", "Unknown")
    )

    catalog = (
        catalog
        .sort_values("anime_id")
        .reset_index(drop=True)
    )

    if catalog.empty:
        raise ValueError(
            "The normalized anime catalog is empty."
        )

    return catalog, raw_row_count


def _load_ratings(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, validate, and deterministically deduplicate ratings."""

    ratings_path = Path(path)

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings dataset not found: {ratings_path}"
        )

    raw = pd.read_parquet(ratings_path)
    raw_row_count = len(raw)

    _require_columns(
        raw,
        RATING_COLUMNS,
        "rating.parquet",
    )

    ratings = raw.loc[
        :,
        ["user_id", "anime_id", "rating"],
    ].copy()

    for column in [
        "user_id",
        "anime_id",
        "rating",
    ]:
        ratings[column] = pd.to_numeric(
            ratings[column],
            errors="coerce",
        )

    rows_with_missing_required_values = int(
        ratings[
            [
                "user_id",
                "anime_id",
                "rating",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    ratings = ratings.dropna(
        subset=[
            "user_id",
            "anime_id",
            "rating",
        ]
    ).copy()

    ratings[
        [
            "user_id",
            "anime_id",
            "rating",
        ]
    ] = ratings[
        [
            "user_id",
            "anime_id",
            "rating",
        ]
    ].astype(np.int64)

    valid_rating_mask = (
        ratings["rating"].eq(-1)
        | ratings["rating"].between(1, 10)
    )

    if not valid_rating_mask.all():
        invalid_values = sorted(
            ratings.loc[
                ~valid_rating_mask,
                "rating",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected rating values were found: "
            f"{invalid_values}"
        )

    duplicate_pair_rows = int(
        ratings.duplicated(
            subset=[
                "user_id",
                "anime_id",
            ],
            keep=False,
        ).sum()
    )

    duplicate_pair_count = int(
        ratings.duplicated(
            subset=[
                "user_id",
                "anime_id",
            ],
            keep="first",
        ).sum()
    )

    # There are no timestamps that identify the latest record.
    # For duplicate user-anime pairs, the maximum recorded rating
    # is retained deterministically.
    ratings = (
        ratings
        .groupby(
            [
                "user_id",
                "anime_id",
            ],
            as_index=False,
            sort=True,
        )["rating"]
        .max()
    )

    ratings = (
        ratings
        .sort_values(
            [
                "user_id",
                "anime_id",
            ]
        )
        .reset_index(drop=True)
    )

    diagnostics = {
        "raw_rating_rows": int(raw_row_count),
        "rows_with_missing_required_values": (
            rows_with_missing_required_values
        ),
        "rows_participating_in_duplicate_pairs": (
            duplicate_pair_rows
        ),
        "duplicate_pair_records_removed": (
            duplicate_pair_count
        ),
        "deduplicated_rating_rows": int(
            len(ratings)
        ),
    }

    return ratings, diagnostics


def _iterative_filter(
    interactions: pd.DataFrame,
    minimum_user_positives: int,
    minimum_item_users: int,
) -> tuple[pd.DataFrame, int]:
    """Repeatedly enforce user and item support constraints."""

    filtered = interactions.copy()
    iteration_count = 0

    while True:
        iteration_count += 1

        previous_pair_count = len(filtered)
        previous_user_count = (
            filtered["user_id"].nunique()
        )
        previous_item_count = (
            filtered["anime_id"].nunique()
        )

        item_counts = (
            filtered
            .groupby("anime_id")["user_id"]
            .nunique()
        )

        retained_items = item_counts[
            item_counts >= minimum_item_users
        ].index

        filtered = filtered[
            filtered["anime_id"].isin(
                retained_items
            )
        ].copy()

        user_counts = (
            filtered
            .groupby("user_id")["anime_id"]
            .nunique()
        )

        retained_users = user_counts[
            user_counts >= minimum_user_positives
        ].index

        filtered = filtered[
            filtered["user_id"].isin(
                retained_users
            )
        ].copy()

        current_state = (
            len(filtered),
            filtered["user_id"].nunique(),
            filtered["anime_id"].nunique(),
        )

        previous_state = (
            previous_pair_count,
            previous_user_count,
            previous_item_count,
        )

        if current_state == previous_state:
            break

        if filtered.empty:
            raise ValueError(
                "Iterative filtering removed every interaction."
            )

    filtered = (
        filtered
        .sort_values(
            [
                "user_id",
                "anime_id",
            ]
        )
        .reset_index(drop=True)
    )

    return filtered, iteration_count


def _initial_user_split(
    interactions: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    """Create deterministic user-level train, validation, and test splits."""

    records: list[dict[str, int | str]] = []

    for user_id, user_frame in interactions.groupby(
        "user_id",
        sort=True,
    ):
        user_frame = (
            user_frame
            .sort_values("anime_id")
            .reset_index(drop=True)
        )

        item_ids = user_frame[
            "anime_id"
        ].to_numpy(dtype=np.int64)

        rating_lookup = (
            user_frame
            .set_index("anime_id")["rating"]
            .to_dict()
        )

        user_seed = np.random.SeedSequence(
            [
                int(seed),
                int(user_id),
            ]
        )

        rng = np.random.default_rng(
            user_seed
        )

        shuffled_items = rng.permutation(
            item_ids
        )

        n_interactions = len(
            shuffled_items
        )

        n_validation = max(
            1,
            int(
                np.floor(
                    n_interactions
                    * validation_fraction
                )
            ),
        )

        n_test = max(
            1,
            int(
                np.floor(
                    n_interactions
                    * test_fraction
                )
            ),
        )

        while (
            n_validation
            + n_test
            >= n_interactions
        ):
            if n_test >= n_validation:
                n_test -= 1
            else:
                n_validation -= 1

        if n_validation < 1 or n_test < 1:
            raise ValueError(
                f"User {user_id} cannot receive "
                "both validation and test interactions."
            )

        test_items = shuffled_items[
            :n_test
        ]

        validation_items = shuffled_items[
            n_test : n_test + n_validation
        ]

        train_items = shuffled_items[
            n_test + n_validation :
        ]

        split_assignments = (
            ("train", train_items),
            ("validation", validation_items),
            ("test", test_items),
        )

        for split_name, split_items in (
            split_assignments
        ):
            for anime_id in split_items:
                records.append(
                    {
                        "user_id": int(user_id),
                        "anime_id": int(anime_id),
                        "rating": int(
                            rating_lookup[
                                int(anime_id)
                            ]
                        ),
                        "split": split_name,
                    }
                )

    split_frame = pd.DataFrame.from_records(
        records
    )

    return split_frame


def _ensure_train_item_support(
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure every retained item has at least one train interaction.

    When an item appears only in validation or test, one held-out
    interaction is exchanged with a train interaction belonging to
    the same user. The replacement item must retain at least one
    training interaction elsewhere.
    """

    repaired = split_frame.copy()

    while True:
        all_items = set(
            repaired["anime_id"].unique()
        )

        train_items = set(
            repaired.loc[
                repaired["split"].eq("train"),
                "anime_id",
            ].unique()
        )

        missing_items = sorted(
            all_items.difference(
                train_items
            )
        )

        if not missing_items:
            break

        missing_item = int(
            missing_items[0]
        )

        candidate_rows = repaired[
            repaired["anime_id"].eq(
                missing_item
            )
            & repaired["split"].isin(
                [
                    "validation",
                    "test",
                ]
            )
        ].copy()

        candidate_rows[
            "_split_priority"
        ] = candidate_rows[
            "split"
        ].map(
            {
                "validation": 0,
                "test": 1,
            }
        )

        candidate_rows = candidate_rows.sort_values(
            [
                "user_id",
                "_split_priority",
            ]
        )

        repair_completed = False

        for candidate_index, candidate in (
            candidate_rows.iterrows()
        ):
            user_id = int(
                candidate["user_id"]
            )

            original_split = str(
                candidate["split"]
            )

            train_support = (
                repaired.loc[
                    repaired["split"].eq(
                        "train"
                    )
                ]
                .groupby("anime_id")
                .size()
            )

            replacement_rows = repaired[
                repaired["user_id"].eq(
                    user_id
                )
                & repaired["split"].eq(
                    "train"
                )
            ].copy()

            replacement_rows[
                "_train_support"
            ] = replacement_rows[
                "anime_id"
            ].map(train_support)

            replacement_rows = (
                replacement_rows[
                    replacement_rows[
                        "_train_support"
                    ].gt(1)
                ]
                .sort_values(
                    [
                        "_train_support",
                        "anime_id",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

            if replacement_rows.empty:
                continue

            replacement_index = (
                replacement_rows.index[0]
            )

            repaired.at[
                candidate_index,
                "split",
            ] = "train"

            repaired.at[
                replacement_index,
                "split",
            ] = original_split

            repair_completed = True
            break

        if not repair_completed:
            raise RuntimeError(
                "Unable to guarantee train support for "
                f"anime_id={missing_item} without "
                "invalidating a user's held-out split."
            )

    return repaired


def _sort_split_frame(
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Sort split records in one canonical order."""

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    result = split_frame.copy()

    result["_split_order"] = (
        result["split"]
        .map(split_order)
    )

    result = (
        result
        .sort_values(
            [
                "user_id",
                "_split_order",
                "anime_id",
            ]
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    return result


def _build_binary_matrix(
    frame: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> sparse.csr_matrix:
    """Build a binary implicit-feedback matrix."""

    values = np.ones(
        len(frame),
        dtype=np.float32,
    )

    matrix = sparse.coo_matrix(
        (
            values,
            (
                frame["user_index"].to_numpy(
                    dtype=np.int64
                ),
                frame["item_index"].to_numpy(
                    dtype=np.int64
                ),
            ),
        ),
        shape=(
            n_users,
            n_items,
        ),
        dtype=np.float32,
    )

    return matrix.tocsr()


def _cohort_hash(
    split_frame: pd.DataFrame,
) -> str:
    """Hash the exact final split assignment."""

    canonical = (
        split_frame[
            [
                "user_id",
                "anime_id",
                "rating",
                "split",
            ]
        ]
        .sort_values(
            [
                "user_id",
                "split",
                "anime_id",
            ]
        )
        .to_csv(
            index=False,
            lineterminator="\n",
        )
        .encode("utf-8")
    )

    return sha256(
        canonical
    ).hexdigest()


def prepare_data(
    anime_path: str | Path,
    ratings_path: str | Path,
    config: dict[str, Any],
) -> PreparedData:
    """Create the final deterministic recommendation dataset."""

    anime_path = Path(anime_path)
    ratings_path = Path(ratings_path)

    catalog, raw_catalog_rows = (
        _load_catalog(anime_path)
    )

    ratings, rating_diagnostics = (
        _load_ratings(ratings_path)
    )

    valid_catalog_ids = set(
        catalog["anime_id"].tolist()
    )

    orphan_rating_rows = int(
        (
            ~ratings["anime_id"].isin(
                valid_catalog_ids
            )
        ).sum()
    )

    ratings = ratings[
        ratings["anime_id"].isin(
            valid_catalog_ids
        )
    ].copy()

    positive_threshold = int(
        config[
            "positive_rating_threshold"
        ]
    )

    all_positive_interactions = (
        ratings[
            ratings["rating"].ge(
                positive_threshold
            )
        ][
            [
                "user_id",
                "anime_id",
                "rating",
            ]
        ]
        .drop_duplicates(
            [
                "user_id",
                "anime_id",
            ]
        )
        .sort_values(
            [
                "user_id",
                "anime_id",
            ]
        )
        .reset_index(drop=True)
    )

    positive_counts = (
        all_positive_interactions
        .groupby("user_id")
        .size()
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

    eligible_user_ids = (
        positive_counts[
            positive_counts.between(
                minimum_before,
                maximum_before,
            )
        ]
        .index
        .to_numpy(dtype=np.int64)
    )

    eligible_user_ids = np.sort(
        eligible_user_ids
    )

    target_user_count = int(
        config["target_user_count"]
    )

    seed = int(
        config["seed"]
    )

    if len(eligible_user_ids) > target_user_count:
        rng = np.random.default_rng(
            seed
        )

        selected_user_ids = rng.choice(
            eligible_user_ids,
            size=target_user_count,
            replace=False,
        )

        selected_user_ids = np.sort(
            selected_user_ids
        )
    else:
        selected_user_ids = (
            eligible_user_ids.copy()
        )

    sampled_interactions = (
        all_positive_interactions[
            all_positive_interactions[
                "user_id"
            ].isin(selected_user_ids)
        ]
        .copy()
    )

    filtered_interactions, filter_iterations = (
        _iterative_filter(
            interactions=sampled_interactions,
            minimum_user_positives=int(
                config[
                    "minimum_user_positives_after_filtering"
                ]
            ),
            minimum_item_users=int(
                config[
                    "minimum_item_users"
                ]
            ),
        )
    )

    if filtered_interactions.empty:
        raise ValueError(
            "No interactions remain after filtering."
        )

    final_user_ids = np.sort(
        filtered_interactions[
            "user_id"
        ].unique()
    ).astype(np.int64)

    final_item_ids = np.sort(
        filtered_interactions[
            "anime_id"
        ].unique()
    ).astype(np.int64)

    user_to_index = {
        int(user_id): index
        for index, user_id
        in enumerate(final_user_ids)
    }

    item_to_index = {
        int(item_id): index
        for index, item_id
        in enumerate(final_item_ids)
    }

    split_interactions = (
        _initial_user_split(
            interactions=filtered_interactions,
            seed=seed,
            validation_fraction=float(
                config[
                    "validation_fraction"
                ]
            ),
            test_fraction=float(
                config[
                    "test_fraction"
                ]
            ),
        )
    )

    split_interactions = (
        _ensure_train_item_support(
            split_interactions
        )
    )

    split_interactions = (
        _sort_split_frame(
            split_interactions
        )
    )

    split_interactions[
        "user_index"
    ] = split_interactions[
        "user_id"
    ].map(user_to_index)

    split_interactions[
        "item_index"
    ] = split_interactions[
        "anime_id"
    ].map(item_to_index)

    if (
        split_interactions[
            [
                "user_index",
                "item_index",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "At least one split interaction could not be indexed."
        )

    split_interactions[
        [
            "user_index",
            "item_index",
        ]
    ] = split_interactions[
        [
            "user_index",
            "item_index",
        ]
    ].astype(np.int64)

    indexed_interactions = (
        filtered_interactions.copy()
    )

    indexed_interactions[
        "user_index"
    ] = indexed_interactions[
        "user_id"
    ].map(user_to_index).astype(np.int64)

    indexed_interactions[
        "item_index"
    ] = indexed_interactions[
        "anime_id"
    ].map(item_to_index).astype(np.int64)

    final_catalog = (
        catalog[
            catalog["anime_id"].isin(
                final_item_ids
            )
        ]
        .copy()
    )

    final_catalog[
        "item_index"
    ] = final_catalog[
        "anime_id"
    ].map(item_to_index).astype(np.int64)

    final_catalog = (
        final_catalog
        .sort_values("item_index")
        .reset_index(drop=True)
    )

    n_users = len(
        final_user_ids
    )

    n_items = len(
        final_item_ids
    )

    train_frame = split_interactions[
        split_interactions[
            "split"
        ].eq("train")
    ]

    validation_frame = split_interactions[
        split_interactions[
            "split"
        ].eq("validation")
    ]

    test_frame = split_interactions[
        split_interactions[
            "split"
        ].eq("test")
    ]

    train_matrix = _build_binary_matrix(
        train_frame,
        n_users,
        n_items,
    )

    validation_matrix = _build_binary_matrix(
        validation_frame,
        n_users,
        n_items,
    )

    test_matrix = _build_binary_matrix(
        test_frame,
        n_users,
        n_items,
    )

    all_positive_matrix = _build_binary_matrix(
        split_interactions,
        n_users,
        n_items,
    )

    if len(split_interactions) != len(
        indexed_interactions
    ):
        raise RuntimeError(
            "Split rows do not equal final positive interactions."
        )

    if split_interactions.duplicated(
        [
            "user_id",
            "anime_id",
        ]
    ).any():
        raise RuntimeError(
            "A user-item pair appears in more than one split."
        )

    if train_matrix.multiply(
        validation_matrix
    ).nnz:
        raise RuntimeError(
            "Train and validation matrices overlap."
        )

    if train_matrix.multiply(
        test_matrix
    ).nnz:
        raise RuntimeError(
            "Train and test matrices overlap."
        )

    if validation_matrix.multiply(
        test_matrix
    ).nnz:
        raise RuntimeError(
            "Validation and test matrices overlap."
        )

    user_split_counts = (
        split_interactions
        .groupby(
            [
                "user_id",
                "split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if not required_splits.issubset(
        user_split_counts.columns
    ):
        raise RuntimeError(
            "At least one required split is absent."
        )

    if (
        user_split_counts[
            [
                "train",
                "validation",
                "test",
            ]
        ]
        .le(0)
        .any()
        .any()
    ):
        raise RuntimeError(
            "At least one user lacks a required split."
        )

    items_without_training_support = int(
        n_items
        - train_frame[
            "anime_id"
        ].nunique()
    )

    if items_without_training_support != 0:
        raise RuntimeError(
            "At least one retained item lacks train support."
        )

    final_user_positive_counts = (
        indexed_interactions
        .groupby("user_id")
        .size()
    )

    split_counts = (
        split_interactions[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    summary: dict[str, Any] = {
        "anime_sha256": sha256_file(
            anime_path
        ),
        "ratings_sha256": sha256_file(
            ratings_path
        ),
        "raw_catalog_rows": int(
            raw_catalog_rows
        ),
        **rating_diagnostics,
        "orphan_rating_rows_removed": int(
            orphan_rating_rows
        ),
        "positive_interactions_before_sampling": int(
            len(
                all_positive_interactions
            )
        ),
        "eligible_users_before_sampling": int(
            len(eligible_user_ids)
        ),
        "selected_users_before_filtering": int(
            len(selected_user_ids)
        ),
        "iterative_filter_iterations": int(
            filter_iterations
        ),
        "final_users": int(n_users),
        "final_items": int(n_items),
        "final_positive_interactions": int(
            len(indexed_interactions)
        ),
        "train_interactions": int(
            split_counts.get(
                "train",
                0,
            )
        ),
        "validation_interactions": int(
            split_counts.get(
                "validation",
                0,
            )
        ),
        "test_interactions": int(
            split_counts.get(
                "test",
                0,
            )
        ),
        "minimum_positives_per_user": int(
            final_user_positive_counts.min()
        ),
        "median_positives_per_user": float(
            final_user_positive_counts.median()
        ),
        "maximum_positives_per_user": int(
            final_user_positive_counts.max()
        ),
        "items_without_training_support": int(
            items_without_training_support
        ),
        "cohort_split_sha256": (
            _cohort_hash(
                split_interactions
            )
        ),
    }

    return PreparedData(
        catalog=final_catalog,
        interactions=indexed_interactions,
        split_interactions=split_interactions,
        user_ids=final_user_ids,
        item_ids=final_item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        test_matrix=test_matrix,
        all_positive_matrix=all_positive_matrix,
        summary=summary,
    )
