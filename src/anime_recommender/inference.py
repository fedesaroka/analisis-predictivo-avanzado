"""NumPy-only inference for exported recommendation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class DeploymentArtifacts:
    """Complete set of deployment-ready recommendation artifacts."""

    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    user_biases: np.ndarray
    item_biases: np.ndarray
    global_bias: float

    seen_matrix: sparse.csr_matrix
    catalog: pd.DataFrame

    user_index_to_id: np.ndarray
    item_index_to_id: np.ndarray

    manifest: dict


def load_deployment_artifacts(
    artifact_directory: str | Path,
) -> DeploymentArtifacts:
    """Load and validate the deployment artifact set."""

    artifact_directory = Path(
        artifact_directory
    )

    required_paths = {
        "components": (
            artifact_directory
            / "model_components.npz"
        ),
        "seen": (
            artifact_directory
            / "seen_final.npz"
        ),
        "catalog": (
            artifact_directory
            / "anime_catalog.parquet"
        ),
        "user_mapping": (
            artifact_directory
            / "user_mapping.json"
        ),
        "item_mapping": (
            artifact_directory
            / "item_mapping.json"
        ),
        "manifest": (
            artifact_directory
            / "artifact_manifest.json"
        ),
    }

    missing_paths = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing deployment artifacts: "
            + ", ".join(missing_paths)
        )

    with np.load(
        required_paths["components"],
        allow_pickle=False,
    ) as components:
        user_embeddings = components[
            "user_embeddings"
        ].astype(
            np.float32,
            copy=True,
        )

        item_embeddings = components[
            "effective_item_embeddings"
        ].astype(
            np.float32,
            copy=True,
        )

        user_biases = components[
            "user_biases"
        ].reshape(-1).astype(
            np.float32,
            copy=True,
        )

        item_biases = components[
            "item_biases"
        ].reshape(-1).astype(
            np.float32,
            copy=True,
        )

        global_bias = float(
            components[
                "global_bias"
            ].reshape(-1)[0]
        )

    seen_matrix = sparse.load_npz(
        required_paths["seen"]
    ).tocsr()

    catalog = pd.read_parquet(
        required_paths["catalog"]
    )

    with required_paths[
        "user_mapping"
    ].open(
        "r",
        encoding="utf-8",
    ) as file:
        user_mapping = json.load(file)

    with required_paths[
        "item_mapping"
    ].open(
        "r",
        encoding="utf-8",
    ) as file:
        item_mapping = json.load(file)

    with required_paths[
        "manifest"
    ].open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    user_index_to_id = np.asarray(
        user_mapping[
            "user_index_to_id"
        ],
        dtype=np.int64,
    )

    item_index_to_id = np.asarray(
        item_mapping[
            "item_index_to_id"
        ],
        dtype=np.int64,
    )

    n_users, latent_dim = (
        user_embeddings.shape
    )

    n_items, item_latent_dim = (
        item_embeddings.shape
    )

    if latent_dim != item_latent_dim:
        raise ValueError(
            "User and item latent dimensions differ."
        )

    if len(user_biases) != n_users:
        raise ValueError(
            "User bias count does not match user embeddings."
        )

    if len(item_biases) != n_items:
        raise ValueError(
            "Item bias count does not match item embeddings."
        )

    if seen_matrix.shape != (
        n_users,
        n_items,
    ):
        raise ValueError(
            "Seen matrix shape does not match model components."
        )

    if len(user_index_to_id) != n_users:
        raise ValueError(
            "User mapping size does not match model components."
        )

    if len(item_index_to_id) != n_items:
        raise ValueError(
            "Item mapping size does not match model components."
        )

    if len(catalog) != n_items:
        raise ValueError(
            "Catalog size does not match model components."
        )

    catalog = (
        catalog
        .sort_values("item_index")
        .reset_index(drop=True)
    )

    expected_item_indices = np.arange(
        n_items,
        dtype=np.int64,
    )

    actual_item_indices = catalog[
        "item_index"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        expected_item_indices,
        actual_item_indices,
    ):
        raise ValueError(
            "Catalog item indices are not contiguous."
        )

    arrays_to_check = [
        user_embeddings,
        item_embeddings,
        user_biases,
        item_biases,
        np.asarray([global_bias]),
    ]

    if not all(
        np.isfinite(array).all()
        for array in arrays_to_check
    ):
        raise ValueError(
            "Deployment components contain non-finite values."
        )

    return DeploymentArtifacts(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_biases=user_biases,
        item_biases=item_biases,
        global_bias=global_bias,
        seen_matrix=seen_matrix,
        catalog=catalog,
        user_index_to_id=user_index_to_id,
        item_index_to_id=item_index_to_id,
        manifest=manifest,
    )


def score_known_user(
    artifacts: DeploymentArtifacts,
    user_index: int,
) -> np.ndarray:
    """Calculate every catalog score for one known user."""

    if not 0 <= user_index < (
        len(
            artifacts.user_embeddings
        )
    ):
        raise IndexError(
            f"Invalid user_index: {user_index}"
        )

    scores = (
        artifacts.user_embeddings[
            user_index
        ]
        @ artifacts.item_embeddings.T
    )

    scores = (
        scores
        + artifacts.user_biases[
            user_index
        ]
        + artifacts.item_biases
        + artifacts.global_bias
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Inference generated non-finite scores."
        )

    return scores


def recommend_known_user(
    artifacts: DeploymentArtifacts,
    user_index: int,
    k: int = 10,
) -> pd.DataFrame:
    """Return the highest-scoring unseen items for one known user."""

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    scores = score_known_user(
        artifacts=artifacts,
        user_index=user_index,
    ).copy()

    row_start = artifacts.seen_matrix.indptr[
        user_index
    ]

    row_end = artifacts.seen_matrix.indptr[
        user_index + 1
    ]

    seen_items = artifacts.seen_matrix.indices[
        row_start:row_end
    ]

    scores[
        seen_items
    ] = -np.inf

    available_items = int(
        np.isfinite(scores).sum()
    )

    if available_items < k:
        raise RuntimeError(
            f"Only {available_items} unseen items are available."
        )

    item_indices = np.arange(
        len(scores),
        dtype=np.int64,
    )

    ranking = np.lexsort(
        (
            item_indices,
            -scores,
        )
    )[:k]

    recommendations = artifacts.catalog.loc[
        ranking
    ].copy()

    recommendations.insert(
        0,
        "rank",
        np.arange(
            1,
            k + 1,
            dtype=np.int64,
        ),
    )

    recommendations.insert(
        1,
        "score",
        scores[
            ranking
        ],
    )

    return recommendations.reset_index(
        drop=True
    )


def recommend_all_known_users(
    artifacts: DeploymentArtifacts,
    k: int = 10,
) -> np.ndarray:
    """Create deterministic top-k recommendations for all users."""

    recommendations = np.empty(
        (
            len(
                artifacts.user_embeddings
            ),
            k,
        ),
        dtype=np.int64,
    )

    for user_index in range(
        len(
            artifacts.user_embeddings
        )
    ):
        frame = recommend_known_user(
            artifacts=artifacts,
            user_index=user_index,
            k=k,
        )

        recommendations[
            user_index
        ] = frame[
            "item_index"
        ].to_numpy(
            dtype=np.int64
        )

    return recommendations


def _validated_favorite_indices(
    favorite_item_indices: (
        np.ndarray
        | list[int]
        | tuple[int, ...]
    ),
    n_items: int,
) -> np.ndarray:
    """Normalize and validate selected favorite item indices."""

    indices = np.asarray(
        favorite_item_indices,
        dtype=np.int64,
    ).reshape(-1)

    if len(indices) == 0:
        raise ValueError(
            "At least one favorite item is required."
        )

    indices = np.unique(
        indices
    )

    if np.any(
        indices < 0
    ) or np.any(
        indices >= n_items
    ):
        raise IndexError(
            "At least one favorite item index "
            "is outside the catalog."
        )

    return indices


def build_content_profile(
    normalized_item_features: sparse.csr_matrix,
    favorite_item_indices: (
        np.ndarray
        | list[int]
        | tuple[int, ...]
    ),
) -> np.ndarray:
    """Build one normalized content profile from favorite anime."""

    feature_matrix = (
        normalized_item_features
        .astype(np.float32)
        .tocsr()
    )

    favorite_indices = (
        _validated_favorite_indices(
            favorite_item_indices=(
                favorite_item_indices
            ),
            n_items=feature_matrix.shape[0],
        )
    )

    profile = np.asarray(
        feature_matrix[
            favorite_indices
        ].mean(axis=0)
    ).reshape(-1).astype(
        np.float32,
        copy=False,
    )

    profile_norm = float(
        np.linalg.norm(
            profile
        )
    )

    if not np.isfinite(
        profile_norm
    ) or profile_norm <= 0:
        raise RuntimeError(
            "The selected favorites generated "
            "an invalid content profile."
        )

    profile = (
        profile
        / profile_norm
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        profile
    ).all():
        raise RuntimeError(
            "The content profile contains "
            "non-finite values."
        )

    return profile


def score_new_user_content(
    normalized_item_features: sparse.csr_matrix,
    favorite_item_indices: (
        np.ndarray
        | list[int]
        | tuple[int, ...]
    ),
) -> np.ndarray:
    """Score the catalog for a new user using cosine similarity."""

    feature_matrix = (
        normalized_item_features
        .astype(np.float32)
        .tocsr()
    )

    profile = build_content_profile(
        normalized_item_features=(
            feature_matrix
        ),
        favorite_item_indices=(
            favorite_item_indices
        ),
    )

    scores = np.asarray(
        feature_matrix
        @ profile
    ).reshape(-1).astype(
        np.float32,
        copy=False,
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    if not np.isfinite(
        scores
    ).all():
        raise RuntimeError(
            "Cold-start scoring generated "
            "non-finite values."
        )

    return scores


def recommend_new_user(
    artifacts: DeploymentArtifacts,
    normalized_item_features: sparse.csr_matrix,
    favorite_item_indices: (
        np.ndarray
        | list[int]
        | tuple[int, ...]
    ),
    k: int = 10,
) -> pd.DataFrame:
    """Recommend content-similar unseen titles to a new user."""

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    feature_matrix = (
        normalized_item_features
        .astype(np.float32)
        .tocsr()
    )

    if feature_matrix.shape[0] != len(
        artifacts.catalog
    ):
        raise ValueError(
            "Feature rows do not match the "
            "deployment catalog."
        )

    favorite_indices = (
        _validated_favorite_indices(
            favorite_item_indices=(
                favorite_item_indices
            ),
            n_items=feature_matrix.shape[0],
        )
    )

    scores = score_new_user_content(
        normalized_item_features=(
            feature_matrix
        ),
        favorite_item_indices=(
            favorite_indices
        ),
    ).copy()

    scores[
        favorite_indices
    ] = -np.inf

    available_items = int(
        np.isfinite(
            scores
        ).sum()
    )

    if available_items < k:
        raise RuntimeError(
            f"Only {available_items} candidate items "
            "are available."
        )

    item_indices = np.arange(
        len(scores),
        dtype=np.int64,
    )

    ranking = np.lexsort(
        (
            item_indices,
            -scores,
        )
    )[:k]

    recommendations = (
        artifacts.catalog
        .loc[
            ranking
        ]
        .copy()
    )

    recommendations.insert(
        0,
        "rank",
        np.arange(
            1,
            k + 1,
            dtype=np.int64,
        ),
    )

    recommendations.insert(
        1,
        "similarity",
        scores[
            ranking
        ],
    )

    return recommendations.reset_index(
        drop=True
    )

