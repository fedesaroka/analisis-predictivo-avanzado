"""Ranking and evaluation utilities for recommendation models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def rank_top_k(
    scores: np.ndarray,
    seen_matrix: sparse.csr_matrix,
    k: int,
) -> np.ndarray:
    """Rank unseen items with deterministic item-index tie breaking."""

    score_matrix = np.asarray(
        scores,
        dtype=np.float32,
    )

    if score_matrix.ndim != 2:
        raise ValueError(
            "scores must be a two-dimensional matrix."
        )

    if score_matrix.shape != (
        seen_matrix.shape
    ):
        raise ValueError(
            "scores and seen_matrix must have identical shapes."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if not np.isfinite(
        score_matrix
    ).all():
        raise ValueError(
            "scores contains non-finite values."
        )

    seen_matrix = (
        seen_matrix.tocsr()
    )

    n_users, n_items = (
        score_matrix.shape
    )

    if k >= n_items:
        raise ValueError(
            "k must be smaller than the catalog size."
        )

    item_indices = np.arange(
        n_items,
        dtype=np.int64,
    )

    recommendations = np.empty(
        (
            n_users,
            k,
        ),
        dtype=np.int64,
    )

    for user_index in range(
        n_users
    ):
        user_scores = (
            score_matrix[
                user_index
            ]
            .copy()
        )

        start = seen_matrix.indptr[
            user_index
        ]

        end = seen_matrix.indptr[
            user_index + 1
        ]

        seen_items = seen_matrix.indices[
            start:end
        ]

        user_scores[
            seen_items
        ] = -np.inf

        available_count = int(
            np.isfinite(
                user_scores
            ).sum()
        )

        if available_count < k:
            raise RuntimeError(
                f"User {user_index} has only "
                f"{available_count} candidate items."
            )

        # np.lexsort uses the final key as the primary key.
        # Scores are sorted descending. Item indices are the
        # deterministic secondary key for equal scores.
        ranking = np.lexsort(
            (
                item_indices,
                -user_scores,
            )
        )

        recommendations[
            user_index
        ] = ranking[:k]

    return recommendations


def _discounted_gain(
    hit_vector: np.ndarray,
) -> float:
    """Calculate discounted gain for one ranked hit vector."""

    discounts = 1.0 / np.log2(
        np.arange(
            2,
            len(hit_vector) + 2,
            dtype=np.float64,
        )
    )

    return float(
        np.sum(
            hit_vector
            * discounts
        )
    )


def _mean_intra_list_diversity(
    recommendations: np.ndarray,
    normalized_item_features: sparse.csr_matrix,
) -> float:
    """Calculate mean pairwise cosine distance within each list."""

    if recommendations.shape[1] < 2:
        return 0.0

    pair_rows, pair_columns = (
        np.triu_indices(
            recommendations.shape[1],
            k=1,
        )
    )

    user_diversities: list[float] = []

    for item_indices in recommendations:
        vectors = normalized_item_features[
            item_indices
        ]

        similarities = (
            vectors
            @ vectors.T
        ).toarray()

        pairwise_similarities = (
            similarities[
                pair_rows,
                pair_columns,
            ]
        )

        pairwise_distances = (
            1.0
            - np.clip(
                pairwise_similarities,
                0.0,
                1.0,
            )
        )

        user_diversities.append(
            float(
                pairwise_distances.mean()
            )
        )

    return float(
        np.mean(
            user_diversities
        )
    )


def evaluate_recommendations(
    recommendations: np.ndarray,
    target_matrix: sparse.csr_matrix,
    normalized_item_features: sparse.csr_matrix,
    catalog_size: int,
) -> dict[str, float]:
    """Calculate top-k ranking metrics from recommendation lists."""

    target_matrix = (
        target_matrix.tocsr()
    )

    if recommendations.ndim != 2:
        raise ValueError(
            "recommendations must be two-dimensional."
        )

    if recommendations.shape[0] != (
        target_matrix.shape[0]
    ):
        raise ValueError(
            "Recommendations and targets must have "
            "the same number of users."
        )

    k = int(
        recommendations.shape[1]
    )

    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    hit_rate_values: list[float] = []

    for user_index, recommended_items in enumerate(
        recommendations
    ):
        start = target_matrix.indptr[
            user_index
        ]

        end = target_matrix.indptr[
            user_index + 1
        ]

        relevant_items = target_matrix.indices[
            start:end
        ]

        if len(relevant_items) == 0:
            raise RuntimeError(
                f"User {user_index} has no target items."
            )

        hit_vector = np.isin(
            recommended_items,
            relevant_items,
            assume_unique=False,
        ).astype(np.float64)

        hit_count = float(
            hit_vector.sum()
        )

        precision_values.append(
            hit_count / k
        )

        recall_values.append(
            hit_count
            / len(relevant_items)
        )

        ideal_hit_count = min(
            k,
            len(relevant_items),
        )

        ideal_vector = np.ones(
            ideal_hit_count,
            dtype=np.float64,
        )

        ideal_dcg = _discounted_gain(
            ideal_vector
        )

        actual_dcg = _discounted_gain(
            hit_vector
        )

        ndcg_values.append(
            actual_dcg / ideal_dcg
        )

        hit_rate_values.append(
            float(
                hit_count > 0
            )
        )

    unique_recommended_items = np.unique(
        recommendations
    )

    coverage = (
        len(
            unique_recommended_items
        )
        / catalog_size
    )

    diversity = (
        _mean_intra_list_diversity(
            recommendations,
            normalized_item_features,
        )
    )

    return {
        f"precision_at_{k}": float(
            np.mean(
                precision_values
            )
        ),
        f"recall_at_{k}": float(
            np.mean(
                recall_values
            )
        ),
        f"ndcg_at_{k}": float(
            np.mean(
                ndcg_values
            )
        ),
        f"hit_rate_at_{k}": float(
            np.mean(
                hit_rate_values
            )
        ),
        f"catalog_coverage_at_{k}": float(
            coverage
        ),
        f"intra_list_diversity_at_{k}": float(
            diversity
        ),
    }


def sampled_auc(
    scores: np.ndarray,
    seen_matrix: sparse.csr_matrix,
    target_matrix: sparse.csr_matrix,
    negatives_per_user: int,
    seed: int,
) -> float:
    """Estimate AUC using deterministic negative samples per user."""

    score_matrix = np.asarray(
        scores,
        dtype=np.float32,
    )

    seen_matrix = (
        seen_matrix.tocsr()
    )

    target_matrix = (
        target_matrix.tocsr()
    )

    if score_matrix.shape != seen_matrix.shape:
        raise ValueError(
            "scores and seen_matrix must have identical shapes."
        )

    if target_matrix.shape != seen_matrix.shape:
        raise ValueError(
            "target_matrix and seen_matrix must have identical shapes."
        )

    if negatives_per_user <= 0:
        raise ValueError(
            "negatives_per_user must be positive."
        )

    n_users, n_items = (
        score_matrix.shape
    )

    user_auc_values: list[float] = []

    for user_index in range(
        n_users
    ):
        seen_start = seen_matrix.indptr[
            user_index
        ]

        seen_end = seen_matrix.indptr[
            user_index + 1
        ]

        seen_items = seen_matrix.indices[
            seen_start:seen_end
        ]

        target_start = target_matrix.indptr[
            user_index
        ]

        target_end = target_matrix.indptr[
            user_index + 1
        ]

        positive_items = target_matrix.indices[
            target_start:target_end
        ]

        if len(positive_items) == 0:
            raise RuntimeError(
                f"User {user_index} has no positive target items."
            )

        forbidden = np.zeros(
            n_items,
            dtype=bool,
        )

        forbidden[
            seen_items
        ] = True

        forbidden[
            positive_items
        ] = True

        candidate_negatives = np.flatnonzero(
            ~forbidden
        )

        if len(candidate_negatives) == 0:
            raise RuntimeError(
                f"User {user_index} has no negative candidates."
            )

        sample_size = min(
            negatives_per_user,
            len(candidate_negatives),
        )

        user_seed = np.random.SeedSequence(
            [
                int(seed),
                int(user_index),
            ]
        )

        rng = np.random.default_rng(
            user_seed
        )

        sampled_negative_items = rng.choice(
            candidate_negatives,
            size=sample_size,
            replace=False,
        )

        positive_scores = score_matrix[
            user_index,
            positive_items,
        ]

        negative_scores = score_matrix[
            user_index,
            sampled_negative_items,
        ]

        comparisons = (
            positive_scores[:, None]
            - negative_scores[None, :]
        )

        strict_wins = (
            comparisons > 0
        ).mean()

        ties = (
            comparisons == 0
        ).mean()

        user_auc_values.append(
            float(
                strict_wins
                + 0.5 * ties
            )
        )

    return float(
        np.mean(
            user_auc_values
        )
    )


def evaluate_score_matrix(
    scores: np.ndarray,
    seen_matrix: sparse.csr_matrix,
    target_matrix: sparse.csr_matrix,
    normalized_item_features: sparse.csr_matrix,
    k: int,
    auc_negatives: int,
    seed: int,
) -> tuple[
    dict[str, float],
    np.ndarray,
]:
    """Rank and evaluate one complete user-item score matrix."""

    recommendations = rank_top_k(
        scores=scores,
        seen_matrix=seen_matrix,
        k=k,
    )

    metrics = evaluate_recommendations(
        recommendations=recommendations,
        target_matrix=target_matrix,
        normalized_item_features=normalized_item_features,
        catalog_size=scores.shape[1],
    )

    metrics["sampled_auc"] = sampled_auc(
        scores=scores,
        seen_matrix=seen_matrix,
        target_matrix=target_matrix,
        negatives_per_user=auc_negatives,
        seed=seed,
    )

    return metrics, recommendations


def validate_metric_ranges(
    metrics: dict[str, Any],
) -> None:
    """Ensure every probability-like metric lies in [0, 1]."""

    for metric_name, metric_value in metrics.items():
        value = float(
            metric_value
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{metric_name} is outside [0, 1]: {value}"
            )
