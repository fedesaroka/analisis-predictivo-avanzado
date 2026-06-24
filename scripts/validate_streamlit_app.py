"""Validate the final Streamlit application and both inference modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from streamlit.testing.v1 import (
    AppTest,
)

from anime_recommender import (
    build_content_profile,
    load_deployment_artifacts,
    recommend_known_user,
    recommend_new_user,
    score_new_user_content,
)


ROOT = Path(__file__).resolve().parents[1]

APP_PATH = (
    ROOT
    / "app.py"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
)


def validate_application_source() -> None:
    """Ensure the deployment script avoids training dependencies."""

    source = APP_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden_fragments = {
        "import torch": (
            "The application imports PyTorch."
        ),
        "from torch": (
            "The application imports PyTorch."
        ),
        "sklearn": (
            "The application still depends on scikit-learn."
        ),
        "train_recommender": (
            "The application contains training logic."
        ),
        "data/anime.csv": (
            "The application bypasses deployment artifacts."
        ),
        "rating.parquet": (
            "The application reads the training dataset."
        ),
    }

    for fragment, message in (
        forbidden_fragments.items()
    ):
        if fragment in source:
            raise AssertionError(
                message
            )


def validate_known_user_mode(
    artifacts,
    demo_users: list[dict],
) -> None:
    """Compare app inference with exported final recommendations."""

    with np.load(
        ARTIFACT_DIRECTORY
        / "final_recommendations.npz",
        allow_pickle=False,
    ) as archive:
        stored_recommendations = (
            archive[
                "recommendations"
            ]
        )

    for demo_user in demo_users:
        user_index = int(
            demo_user[
                "user_index"
            ]
        )

        first = recommend_known_user(
            artifacts=artifacts,
            user_index=user_index,
            k=10,
        )

        second = recommend_known_user(
            artifacts=artifacts,
            user_index=user_index,
            k=10,
        )

        first_indices = first[
            "item_index"
        ].to_numpy(
            dtype=np.int64
        )

        second_indices = second[
            "item_index"
        ].to_numpy(
            dtype=np.int64
        )

        np.testing.assert_array_equal(
            first_indices,
            second_indices,
        )

        np.testing.assert_array_equal(
            first_indices,
            stored_recommendations[
                user_index
            ],
        )

        if artifacts.seen_matrix[
            user_index,
            first_indices,
        ].nnz:
            raise AssertionError(
                f"Known user {user_index} received "
                "a previously seen item."
            )


def validate_new_user_mode(
    artifacts,
    item_features,
) -> None:
    """Validate deterministic cold-start recommendations."""

    favorite_sets = [
        [0, 1, 2],
        [10, 25, 100, 250],
        [100, 500, 1000, 1500, 2000],
    ]

    for favorites in favorite_sets:
        profile = build_content_profile(
            normalized_item_features=(
                item_features
            ),
            favorite_item_indices=(
                favorites
            ),
        )

        if profile.shape != (
            item_features.shape[1],
        ):
            raise AssertionError(
                "Cold-start profile has an invalid shape."
            )

        if not np.isclose(
            np.linalg.norm(profile),
            1.0,
            rtol=0,
            atol=1e-6,
        ):
            raise AssertionError(
                "Cold-start profile is not normalized."
            )

        scores = score_new_user_content(
            normalized_item_features=(
                item_features
            ),
            favorite_item_indices=(
                favorites
            ),
        )

        if scores.shape != (
            len(
                artifacts.catalog
            ),
        ):
            raise AssertionError(
                "Cold-start score vector has an invalid shape."
            )

        if not np.isfinite(
            scores
        ).all():
            raise AssertionError(
                "Cold-start scores contain non-finite values."
            )

        if not np.all(
            (
                scores >= -1e-6
            )
            & (
                scores <= 1.0 + 1e-6
            )
        ):
            raise AssertionError(
                "Cold-start similarities are outside [0, 1]."
            )

        first = recommend_new_user(
            artifacts=artifacts,
            normalized_item_features=(
                item_features
            ),
            favorite_item_indices=(
                favorites
            ),
            k=10,
        )

        second = recommend_new_user(
            artifacts=artifacts,
            normalized_item_features=(
                item_features
            ),
            favorite_item_indices=(
                list(reversed(favorites))
            ),
            k=10,
        )

        first_indices = first[
            "item_index"
        ].to_numpy(
            dtype=np.int64
        )

        second_indices = second[
            "item_index"
        ].to_numpy(
            dtype=np.int64
        )

        np.testing.assert_array_equal(
            first_indices,
            second_indices,
        )

        if set(first_indices).intersection(
            favorites
        ):
            raise AssertionError(
                "A selected favorite was returned "
                "as a cold-start recommendation."
            )


def validate_streamlit_execution() -> None:
    """Execute the Streamlit script through its testing runtime."""

    application = AppTest.from_file(
        str(APP_PATH),
        default_timeout=60,
    )

    application.run()

    if len(
        application.exception
    ):
        messages = [
            str(
                exception.value
            )
            for exception in (
                application.exception
            )
        ]

        raise AssertionError(
            "The Streamlit application raised exceptions:\n"
            + "\n".join(messages)
        )

    if len(
        application.tabs
    ) != 2:
        raise AssertionError(
            "The application must display exactly two modes."
        )


def main() -> None:
    """Run the complete application validation suite."""

    validate_application_source()

    artifacts = load_deployment_artifacts(
        ARTIFACT_DIRECTORY
    )

    item_features = sparse.load_npz(
        ARTIFACT_DIRECTORY
        / "item_features.npz"
    ).tocsr()

    with (
        ARTIFACT_DIRECTORY
        / "demo_users.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        demo_users = json.load(
            file
        )

    if "torch" in sys.modules:
        raise AssertionError(
            "PyTorch was loaded before Streamlit execution."
        )

    validate_known_user_mode(
        artifacts=artifacts,
        demo_users=demo_users,
    )

    validate_new_user_mode(
        artifacts=artifacts,
        item_features=item_features,
    )

    validate_streamlit_execution()

    print(
        "Known-user recommendations validated."
    )

    print(
        "New-user cold-start recommendations validated."
    )

    print(
        "Streamlit script executed without exceptions."
    )

    print(
        "Final Streamlit application validation passed."
    )


if __name__ == "__main__":
    main()
