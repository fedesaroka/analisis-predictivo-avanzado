"""Deterministic item metadata construction."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED_CATALOG_COLUMNS = {
    "item_index",
    "anime_id",
    "genre",
    "type",
}


@dataclass(frozen=True)
class ItemFeatureData:
    """Raw and normalized item metadata matrices."""

    matrix: sparse.csr_matrix
    normalized_matrix: sparse.csr_matrix

    feature_names: tuple[str, ...]
    genre_names: tuple[str, ...]
    type_names: tuple[str, ...]

    summary: dict[str, Any]


def _normalize_token(
    value: object,
) -> str:
    """Normalize one metadata token deterministically."""

    text = unescape(
        str(value)
    )

    text = " ".join(
        text.strip().split()
    )

    text = text.casefold()

    if not text:
        return "unknown"

    return text


def _parse_genres(
    value: object,
) -> tuple[str, ...]:
    """Convert a comma-delimited genre field to sorted tokens."""

    normalized_value = _normalize_token(
        value
    )

    tokens = {
        _normalize_token(token)
        for token in normalized_value.split(",")
        if _normalize_token(token)
    }

    if not tokens:
        tokens = {"unknown"}

    return tuple(
        sorted(tokens)
    )


def _parse_type(
    value: object,
) -> str:
    """Normalize the anime type."""

    return _normalize_token(
        value
    )


def build_item_features(
    catalog: pd.DataFrame,
) -> ItemFeatureData:
    """Construct genre and type features aligned with item_index."""

    missing_columns = sorted(
        REQUIRED_CATALOG_COLUMNS.difference(
            catalog.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Catalog is missing required metadata columns: "
            + ", ".join(missing_columns)
        )

    ordered_catalog = (
        catalog
        .sort_values("item_index")
        .reset_index(drop=True)
        .copy()
    )

    expected_indices = np.arange(
        len(ordered_catalog),
        dtype=np.int64,
    )

    actual_indices = ordered_catalog[
        "item_index"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise ValueError(
            "Catalog item_index values must be contiguous "
            "and begin at zero."
        )

    genres_per_item = [
        _parse_genres(value)
        for value in ordered_catalog["genre"]
    ]

    types_per_item = [
        _parse_type(value)
        for value in ordered_catalog["type"]
    ]

    genre_names = tuple(
        sorted(
            {
                genre
                for item_genres in genres_per_item
                for genre in item_genres
            }
        )
    )

    type_names = tuple(
        sorted(
            set(types_per_item)
        )
    )

    feature_names = tuple(
        [f"genre:{name}" for name in genre_names]
        + [f"type:{name}" for name in type_names]
    )

    feature_to_index = {
        feature_name: index
        for index, feature_name in enumerate(
            feature_names
        )
    }

    row_indices: list[int] = []
    column_indices: list[int] = []

    for item_index, (
        item_genres,
        item_type,
    ) in enumerate(
        zip(
            genres_per_item,
            types_per_item,
            strict=True,
        )
    ):
        item_feature_names = {
            f"genre:{genre}"
            for genre in item_genres
        }

        item_feature_names.add(
            f"type:{item_type}"
        )

        for feature_name in sorted(
            item_feature_names
        ):
            row_indices.append(
                item_index
            )

            column_indices.append(
                feature_to_index[
                    feature_name
                ]
            )

    values = np.ones(
        len(row_indices),
        dtype=np.float32,
    )

    matrix = sparse.coo_matrix(
        (
            values,
            (
                np.asarray(
                    row_indices,
                    dtype=np.int64,
                ),
                np.asarray(
                    column_indices,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(ordered_catalog),
            len(feature_names),
        ),
        dtype=np.float32,
    ).tocsr()

    row_nonzero_counts = np.diff(
        matrix.indptr
    )

    if np.any(
        row_nonzero_counts < 2
    ):
        raise RuntimeError(
            "Every anime must contain at least one genre "
            "feature and one type feature."
        )

    squared_norms = np.asarray(
        matrix.multiply(
            matrix
        ).sum(axis=1)
    ).ravel()

    norms = np.sqrt(
        squared_norms
    )

    if np.any(
        norms <= 0
    ):
        raise RuntimeError(
            "At least one metadata row has zero magnitude."
        )

    inverse_norms = (
        1.0 / norms
    ).astype(np.float32)

    normalized_matrix = (
        sparse.diags(
            inverse_norms,
            format="csr",
        )
        @ matrix
    ).tocsr()

    normalized_norms = np.sqrt(
        np.asarray(
            normalized_matrix.multiply(
                normalized_matrix
            ).sum(axis=1)
        ).ravel()
    )

    if not np.allclose(
        normalized_norms,
        1.0,
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Metadata rows were not normalized correctly."
        )

    feature_frequencies = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    summary: dict[str, Any] = {
        "items": int(
            matrix.shape[0]
        ),
        "features": int(
            matrix.shape[1]
        ),
        "genre_features": int(
            len(genre_names)
        ),
        "type_features": int(
            len(type_names)
        ),
        "nonzero_values": int(
            matrix.nnz
        ),
        "minimum_features_per_item": int(
            row_nonzero_counts.min()
        ),
        "median_features_per_item": float(
            np.median(
                row_nonzero_counts
            )
        ),
        "maximum_features_per_item": int(
            row_nonzero_counts.max()
        ),
        "minimum_feature_frequency": int(
            feature_frequencies.min()
        ),
        "maximum_feature_frequency": int(
            feature_frequencies.max()
        ),
    }

    return ItemFeatureData(
        matrix=matrix,
        normalized_matrix=normalized_matrix,
        feature_names=feature_names,
        genre_names=genre_names,
        type_names=type_names,
        summary=summary,
    )
