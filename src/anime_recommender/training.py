"""Deterministic training for neural recommendation models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

from anime_recommender.models import (
    HybridLatentFactorModel,
)


SUPPORTED_OBJECTIVES = {
    "logistic",
    "bpr",
    "warp_style_hard_negative",
}


@dataclass(frozen=True)
class TrainingSpec:
    """Complete specification for one model run."""

    name: str
    objective: str
    use_metadata: bool

    latent_dim: int
    epochs: int
    batch_size: int

    learning_rate: float
    weight_decay: float
    metadata_weight: float

    negatives_per_positive: int
    max_sampled: int
    margin: float
    gradient_clip_norm: float

    seed: int

    def validate(
        self,
    ) -> None:
        """Validate the training specification."""

        if not self.name:
            raise ValueError(
                "TrainingSpec.name cannot be empty."
            )

        if self.objective not in (
            SUPPORTED_OBJECTIVES
        ):
            raise ValueError(
                f"Unsupported objective: {self.objective}"
            )

        integer_values = {
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "negatives_per_positive": (
                self.negatives_per_positive
            ),
            "max_sampled": self.max_sampled,
        }

        for key, value in integer_values.items():
            if int(value) <= 0:
                raise ValueError(
                    f"{key} must be positive."
                )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative."
            )

        if self.metadata_weight <= 0:
            raise ValueError(
                "metadata_weight must be positive."
            )

        if self.margin <= 0:
            raise ValueError(
                "margin must be positive."
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )


@dataclass
class TrainingRun:
    """Result of one completed neural training run."""

    model: HybridLatentFactorModel
    history: list[dict[str, Any]]
    parameter_count: int
    specification: TrainingSpec


def set_reproducible_seed(
    seed: int,
) -> None:
    """Set random seeds and deterministic CPU behavior."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.set_num_threads(
        1
    )

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        # PyTorch only permits setting this once after startup.
        pass

    torch.use_deterministic_algorithms(
        True
    )


def _dense_seen_matrix(
    train_matrix: sparse.csr_matrix,
) -> np.ndarray:
    """Create a compact Boolean matrix for negative sampling."""

    seen = (
        train_matrix
        .astype(bool)
        .toarray()
    )

    if seen.ndim != 2:
        raise RuntimeError(
            "The dense seen matrix is not two-dimensional."
        )

    return seen


def _sample_negative_items(
    users: np.ndarray,
    seen: np.ndarray,
    n_items: int,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Sample items absent from each user's training history."""

    users = np.asarray(
        users,
        dtype=np.int64,
    )

    if users.ndim != 1:
        raise ValueError(
            "users must be one-dimensional."
        )

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    if np.any(
        seen[
            users
        ].all(axis=1)
    ):
        raise RuntimeError(
            "At least one user has no valid negative items."
        )

    negative_items = rng.integers(
        low=0,
        high=n_items,
        size=(
            len(users),
            n_samples,
        ),
        dtype=np.int64,
    )

    invalid = seen[
        users[:, None],
        negative_items,
    ]

    resampling_round = 0

    while invalid.any():
        resampling_round += 1

        if resampling_round > 1000:
            raise RuntimeError(
                "Negative sampling failed to converge."
            )

        negative_items[
            invalid
        ] = rng.integers(
            low=0,
            high=n_items,
            size=int(
                invalid.sum()
            ),
            dtype=np.int64,
        )

        invalid = seen[
            users[:, None],
            negative_items,
        ]

    return negative_items


def _logistic_loss(
    model: HybridLatentFactorModel,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    users_numpy: np.ndarray,
    seen: np.ndarray,
    item_features: torch.Tensor,
    n_items: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Pointwise logistic loss with sampled negatives."""

    negative_items_numpy = (
        _sample_negative_items(
            users=users_numpy,
            seen=seen,
            n_items=n_items,
            rng=rng,
            n_samples=negatives_per_positive,
        )
    )

    negative_items = torch.as_tensor(
        negative_items_numpy.reshape(-1),
        dtype=torch.long,
    )

    repeated_users = (
        users.repeat_interleave(
            negatives_per_positive
        )
    )

    positive_scores = model.score_pairs(
        users=users,
        items=positive_items,
        item_features=item_features,
    )

    negative_scores = model.score_pairs(
        users=repeated_users,
        items=negative_items,
        item_features=item_features,
    )

    positive_loss = F.softplus(
        -positive_scores
    ).mean()

    negative_loss = F.softplus(
        negative_scores
    ).mean()

    return 0.5 * (
        positive_loss
        + negative_loss
    )


def _bpr_loss(
    model: HybridLatentFactorModel,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    users_numpy: np.ndarray,
    seen: np.ndarray,
    item_features: torch.Tensor,
    n_items: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Exact Bayesian Personalized Ranking pairwise loss."""

    negative_items_numpy = (
        _sample_negative_items(
            users=users_numpy,
            seen=seen,
            n_items=n_items,
            rng=rng,
            n_samples=negatives_per_positive,
        )
    )

    negative_items = torch.as_tensor(
        negative_items_numpy.reshape(-1),
        dtype=torch.long,
    )

    repeated_users = (
        users.repeat_interleave(
            negatives_per_positive
        )
    )

    positive_scores = model.score_pairs(
        users=users,
        items=positive_items,
        item_features=item_features,
    )

    repeated_positive_scores = (
        positive_scores.repeat_interleave(
            negatives_per_positive
        )
    )

    negative_scores = model.score_pairs(
        users=repeated_users,
        items=negative_items,
        item_features=item_features,
    )

    return -F.logsigmoid(
        repeated_positive_scores
        - negative_scores
    ).mean()


def _warp_style_loss(
    model: HybridLatentFactorModel,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    users_numpy: np.ndarray,
    seen: np.ndarray,
    item_features: torch.Tensor,
    n_items: int,
    max_sampled: int,
    margin: float,
    rng: np.random.Generator,
    harmonic_weights: np.ndarray,
) -> tuple[
    torch.Tensor | None,
    int,
]:
    """Hard-negative pairwise ranking inspired by WARP.

    For every positive pair, negatives are sampled sequentially.
    The first negative violating the margin is retained.

    The number of trials estimates the positive item's rank:

        estimated_rank = floor((n_items - 1) / trials)

    A harmonic rank weight then emphasizes mistakes estimated to
    occur near the top of the recommendation list.

    This is deliberately named WARP-style rather than exact WARP.
    """

    batch_size = len(
        users_numpy
    )

    with torch.no_grad():
        detached_positive_scores = (
            model.score_pairs(
                users=users,
                items=positive_items,
                item_features=item_features,
            )
            .detach()
        )

    found = np.zeros(
        batch_size,
        dtype=bool,
    )

    selected_negative_items = np.full(
        batch_size,
        fill_value=-1,
        dtype=np.int64,
    )

    selected_trials = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    for trial in range(
        1,
        max_sampled + 1,
    ):
        active_indices = np.flatnonzero(
            ~found
        )

        if len(active_indices) == 0:
            break

        active_users_numpy = users_numpy[
            active_indices
        ]

        sampled_negative_items = (
            _sample_negative_items(
                users=active_users_numpy,
                seen=seen,
                n_items=n_items,
                rng=rng,
                n_samples=1,
            )
            .reshape(-1)
        )

        active_indices_tensor = torch.as_tensor(
            active_indices,
            dtype=torch.long,
        )

        sampled_negative_tensor = torch.as_tensor(
            sampled_negative_items,
            dtype=torch.long,
        )

        with torch.no_grad():
            negative_scores = (
                model.score_pairs(
                    users=users[
                        active_indices_tensor
                    ],
                    items=sampled_negative_tensor,
                    item_features=item_features,
                )
            )

            violations = (
                negative_scores
                + margin
                > detached_positive_scores[
                    active_indices_tensor
                ]
            ).cpu().numpy()

        violating_active_indices = (
            active_indices[
                violations
            ]
        )

        selected_negative_items[
            violating_active_indices
        ] = sampled_negative_items[
            violations
        ]

        selected_trials[
            violating_active_indices
        ] = trial

        found[
            violating_active_indices
        ] = True

    found_indices = np.flatnonzero(
        found
    )

    if len(found_indices) == 0:
        return None, 0

    found_indices_tensor = torch.as_tensor(
        found_indices,
        dtype=torch.long,
    )

    selected_negative_tensor = torch.as_tensor(
        selected_negative_items[
            found_indices
        ],
        dtype=torch.long,
    )

    selected_positive_scores = (
        model.score_pairs(
            users=users[
                found_indices_tensor
            ],
            items=positive_items[
                found_indices_tensor
            ],
            item_features=item_features,
        )
    )

    selected_negative_scores = (
        model.score_pairs(
            users=users[
                found_indices_tensor
            ],
            items=selected_negative_tensor,
            item_features=item_features,
        )
    )

    rank_estimates = np.floor(
        (
            n_items - 1
        )
        / selected_trials[
            found_indices
        ]
    ).astype(
        np.int64
    )

    rank_estimates = np.clip(
        rank_estimates,
        1,
        n_items - 1,
    )

    example_weights = torch.as_tensor(
        harmonic_weights[
            rank_estimates - 1
        ],
        dtype=torch.float32,
    )

    margin_losses = F.relu(
        margin
        - selected_positive_scores
        + selected_negative_scores
    )

    loss = (
        example_weights
        * margin_losses
    ).mean()

    return loss, len(
        found_indices
    )


def train_recommender(
    train_matrix: sparse.csr_matrix,
    item_feature_matrix: sparse.csr_matrix,
    specification: TrainingSpec,
    verbose: bool = False,
) -> TrainingRun:
    """Train one collaborative or hybrid recommender."""

    specification.validate()

    set_reproducible_seed(
        specification.seed
    )

    train_matrix = (
        train_matrix
        .astype(np.float32)
        .tocsr()
    )

    item_feature_matrix = (
        item_feature_matrix
        .astype(np.float32)
        .tocsr()
    )

    n_users, n_items = (
        train_matrix.shape
    )

    if item_feature_matrix.shape[0] != n_items:
        raise ValueError(
            "Metadata row count must equal item count."
        )

    n_features = (
        item_feature_matrix.shape[1]
    )

    dense_item_features = (
        item_feature_matrix
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    item_features_tensor = torch.from_numpy(
        dense_item_features
    )

    seen = _dense_seen_matrix(
        train_matrix
    )

    train_coo = train_matrix.tocoo()

    canonical_order = np.lexsort(
        (
            train_coo.col,
            train_coo.row,
        )
    )

    all_users = train_coo.row[
        canonical_order
    ].astype(
        np.int64,
        copy=False,
    )

    all_positive_items = train_coo.col[
        canonical_order
    ].astype(
        np.int64,
        copy=False,
    )

    if len(all_users) == 0:
        raise ValueError(
            "The training matrix contains no interactions."
        )

    model = HybridLatentFactorModel(
        n_users=n_users,
        n_items=n_items,
        n_features=n_features,
        latent_dim=specification.latent_dim,
        use_metadata=specification.use_metadata,
        metadata_weight=specification.metadata_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=specification.learning_rate,
        weight_decay=specification.weight_decay,
    )

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    rng = np.random.default_rng(
        specification.seed
    )

    harmonic_weights = np.cumsum(
        1.0
        / np.arange(
            1,
            n_items,
            dtype=np.float64,
        )
    ).astype(
        np.float32
    )

    history: list[
        dict[str, Any]
    ] = []

    n_interactions = len(
        all_users
    )

    for epoch in range(
        1,
        specification.epochs + 1,
    ):
        model.train()

        permutation = rng.permutation(
            n_interactions
        )

        total_loss = 0.0
        total_loss_examples = 0
        total_violations = 0
        total_positive_examples = 0
        completed_batches = 0

        for batch_start in range(
            0,
            n_interactions,
            specification.batch_size,
        ):
            batch_indices = permutation[
                batch_start:
                batch_start
                + specification.batch_size
            ]

            users_numpy = all_users[
                batch_indices
            ]

            positive_items_numpy = (
                all_positive_items[
                    batch_indices
                ]
            )

            users = torch.as_tensor(
                users_numpy,
                dtype=torch.long,
            )

            positive_items = torch.as_tensor(
                positive_items_numpy,
                dtype=torch.long,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if specification.objective == (
                "logistic"
            ):
                loss = _logistic_loss(
                    model=model,
                    users=users,
                    positive_items=positive_items,
                    users_numpy=users_numpy,
                    seen=seen,
                    item_features=item_features_tensor,
                    n_items=n_items,
                    negatives_per_positive=(
                        specification
                        .negatives_per_positive
                    ),
                    rng=rng,
                )

                loss_example_count = len(
                    users_numpy
                )

            elif specification.objective == (
                "bpr"
            ):
                loss = _bpr_loss(
                    model=model,
                    users=users,
                    positive_items=positive_items,
                    users_numpy=users_numpy,
                    seen=seen,
                    item_features=item_features_tensor,
                    n_items=n_items,
                    negatives_per_positive=(
                        specification
                        .negatives_per_positive
                    ),
                    rng=rng,
                )

                loss_example_count = len(
                    users_numpy
                )

            else:
                (
                    loss,
                    violation_count,
                ) = _warp_style_loss(
                    model=model,
                    users=users,
                    positive_items=positive_items,
                    users_numpy=users_numpy,
                    seen=seen,
                    item_features=item_features_tensor,
                    n_items=n_items,
                    max_sampled=(
                        specification.max_sampled
                    ),
                    margin=specification.margin,
                    rng=rng,
                    harmonic_weights=(
                        harmonic_weights
                    ),
                )

                total_positive_examples += len(
                    users_numpy
                )

                total_violations += (
                    violation_count
                )

                loss_example_count = (
                    violation_count
                )

                if loss is None:
                    continue

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"{specification.name} generated "
                    "a non-finite loss."
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    specification
                    .gradient_clip_norm
                ),
            )

            optimizer.step()

            total_loss += (
                float(
                    loss.item()
                )
                * loss_example_count
            )

            total_loss_examples += (
                loss_example_count
            )

            completed_batches += 1

        mean_loss = (
            total_loss
            / max(
                total_loss_examples,
                1,
            )
        )

        if specification.objective == (
            "warp_style_hard_negative"
        ):
            violation_rate: float | None = (
                total_violations
                / max(
                    total_positive_examples,
                    1,
                )
            )
        else:
            violation_rate = None

        epoch_record = {
            "epoch": int(epoch),
            "loss": float(
                mean_loss
            ),
            "completed_batches": int(
                completed_batches
            ),
            "loss_examples": int(
                total_loss_examples
            ),
            "violation_rate": (
                None
                if violation_rate is None
                else float(
                    violation_rate
                )
            ),
        }

        history.append(
            epoch_record
        )

        if verbose:
            message = (
                f"{specification.name} "
                f"epoch {epoch:02d}/"
                f"{specification.epochs:02d} "
                f"loss={mean_loss:.6f}"
            )

            if violation_rate is not None:
                message += (
                    " violation_rate="
                    f"{violation_rate:.4f}"
                )

            print(
                message,
                flush=True,
            )

    return TrainingRun(
        model=model,
        history=history,
        parameter_count=parameter_count,
        specification=specification,
    )


def score_all_items(
    model: HybridLatentFactorModel,
    item_feature_matrix: sparse.csr_matrix,
    n_users: int,
    user_batch_size: int,
) -> np.ndarray:
    """Generate a finite score for every user-item pair."""

    if user_batch_size <= 0:
        raise ValueError(
            "user_batch_size must be positive."
        )

    item_feature_matrix = (
        item_feature_matrix
        .astype(np.float32)
        .tocsr()
    )

    item_features_tensor = torch.from_numpy(
        item_feature_matrix
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    model.eval()

    batches: list[
        np.ndarray
    ] = []

    with torch.no_grad():
        for user_start in range(
            0,
            n_users,
            user_batch_size,
        ):
            users = torch.arange(
                user_start,
                min(
                    user_start
                    + user_batch_size,
                    n_users,
                ),
                dtype=torch.long,
            )

            batch_scores = (
                model.score_all_items(
                    users=users,
                    item_features=(
                        item_features_tensor
                    ),
                )
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            batches.append(
                batch_scores
            )

    score_matrix = np.vstack(
        batches
    )

    expected_shape = (
        n_users,
        item_feature_matrix.shape[0],
    )

    if score_matrix.shape != expected_shape:
        raise RuntimeError(
            "Unexpected score matrix shape: "
            f"{score_matrix.shape}, expected {expected_shape}."
        )

    if not np.isfinite(
        score_matrix
    ).all():
        raise RuntimeError(
            "The model generated non-finite scores."
        )

    return score_matrix
