"""Popularity and content-based recommendation baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from anime_recommender.data import PreparedData
from anime_recommender.evaluation import (
    evaluate_score_matrix,
    validate_metric_ranges,
)
from anime_recommender.features import (
    ItemFeatureData,
)


def popularity_score_matrix(
    train_matrix: sparse.csr_matrix,
) -> np.ndarray:
    """Score every item using positive training-interaction count."""

    item_popularity = np.asarray(
        train_matrix.sum(axis=0)
    ).ravel().astype(
        np.float32
    )

    # log1p preserves the popularity ordering while reducing
    # large numerical differences between head and tail items.
    item_scores = np.log1p(
        item_popularity
    ).astype(
        np.float32
    )

    return np.broadcast_to(
        item_scores,
        train_matrix.shape,
    ).copy()


def content_score_matrix(
    train_matrix: sparse.csr_matrix,
    normalized_item_features: sparse.csr_matrix,
) -> np.ndarray:
    """Score items by cosine similarity to each user's content profile."""

    train_matrix = (
        train_matrix
        .astype(np.float32)
        .tocsr()
    )

    user_profiles = (
        train_matrix
        @ normalized_item_features
    ).tocsr()

    profile_squared_norms = np.asarray(
        user_profiles.multiply(
            user_profiles
        ).sum(axis=1)
    ).ravel()

    profile_norms = np.sqrt(
        profile_squared_norms
    )

    if np.any(
        profile_norms <= 0
    ):
        zero_profile_users = np.flatnonzero(
            profile_norms <= 0
        ).tolist()

        raise RuntimeError(
            "At least one user has a zero content profile: "
            f"{zero_profile_users[:10]}"
        )

    normalized_user_profiles = (
        sparse.diags(
            (
                1.0
                / profile_norms
            ).astype(
                np.float32
            ),
            format="csr",
        )
        @ user_profiles
    ).tocsr()

    score_matrix = (
        normalized_user_profiles
        @ normalized_item_features.T
    ).toarray().astype(
        np.float32
    )

    if not np.isfinite(
        score_matrix
    ).all():
        raise RuntimeError(
            "The content baseline generated non-finite scores."
        )

    return score_matrix


def evaluate_baselines(
    prepared: PreparedData,
    item_features: ItemFeatureData,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    """Evaluate popularity and content baselines on validation data."""

    score_matrices = {
        "popularity": popularity_score_matrix(
            prepared.train_matrix
        ),
        "content": content_score_matrix(
            prepared.train_matrix,
            item_features.normalized_matrix,
        ),
    }

    result_rows: list[
        dict[str, Any]
    ] = []

    recommendations_by_model: dict[
        str,
        np.ndarray,
    ] = {}

    for model_name, score_matrix in (
        score_matrices.items()
    ):
        metrics, recommendations = (
            evaluate_score_matrix(
                scores=score_matrix,
                seen_matrix=prepared.train_matrix,
                target_matrix=prepared.validation_matrix,
                normalized_item_features=(
                    item_features.normalized_matrix
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
                    config[
                        "seed"
                    ]
                ),
            )
        )

        validate_metric_ranges(
            metrics
        )

        result_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        recommendations_by_model[
            model_name
        ] = recommendations

    results = pd.DataFrame(
        result_rows
    )

    model_order = {
        "popularity": 0,
        "content": 1,
    }

    results["_model_order"] = (
        results["model"]
        .map(model_order)
    )

    results = (
        results
        .sort_values(
            "_model_order"
        )
        .drop(
            columns="_model_order"
        )
        .reset_index(drop=True)
    )

    return (
        results,
        recommendations_by_model,
    )


def recommendation_examples(
    prepared: PreparedData,
    recommendations_by_model: dict[
        str,
        np.ndarray,
    ],
    user_indices: list[int],
) -> pd.DataFrame:
    """Build readable recommendation examples for selected users."""

    catalog_lookup = (
        prepared.catalog
        .set_index("item_index")
    )

    rows: list[
        dict[str, Any]
    ] = []

    for model_name, recommendations in (
        recommendations_by_model.items()
    ):
        for user_index in user_indices:
            if not 0 <= user_index < len(
                prepared.user_ids
            ):
                raise IndexError(
                    f"Invalid user_index: {user_index}"
                )

            user_id = int(
                prepared.user_ids[
                    user_index
                ]
            )

            for rank, item_index in enumerate(
                recommendations[
                    user_index
                ],
                start=1,
            ):
                item_index = int(
                    item_index
                )

                catalog_row = catalog_lookup.loc[
                    item_index
                ]

                is_validation_positive = bool(
                    prepared.validation_matrix[
                        user_index,
                        item_index,
                    ]
                )

                rows.append(
                    {
                        "model": model_name,
                        "user_index": int(
                            user_index
                        ),
                        "user_id": user_id,
                        "rank": int(rank),
                        "item_index": item_index,
                        "anime_id": int(
                            catalog_row[
                                "anime_id"
                            ]
                        ),
                        "title": str(
                            catalog_row[
                                "name"
                            ]
                        ),
                        "genre": str(
                            catalog_row[
                                "genre"
                            ]
                        ),
                        "type": str(
                            catalog_row[
                                "type"
                            ]
                        ),
                        "validation_hit": (
                            is_validation_positive
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )
