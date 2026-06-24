"""Smoke test for the final PyTorch recommendation environment."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED = 42


def set_seed(seed: int) -> None:
    """Set every random seed used by this smoke test."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class TinyHybridRecommender(nn.Module):
    """Small hybrid latent-factor model for environment verification."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_features: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(
            n_users,
            latent_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            latent_dim,
        )

        self.metadata_projection = nn.Linear(
            n_features,
            latent_dim,
            bias=False,
        )

        self.item_bias = nn.Embedding(
            n_items,
            1,
        )

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.05,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            std=0.05,
        )

        nn.init.normal_(
            self.metadata_projection.weight,
            std=0.05,
        )

        nn.init.zeros_(
            self.item_bias.weight,
        )

    def score(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate one score for every user-item pair."""

        user_vectors = self.user_embedding(users)

        item_vectors = (
            self.item_embedding(items)
            + self.metadata_projection(
                item_features[items]
            )
        )

        biases = (
            self.item_bias(items)
            .squeeze(-1)
        )

        return (
            user_vectors * item_vectors
        ).sum(dim=1) + biases


def run_once() -> tuple[float, float, np.ndarray]:
    """Train a tiny model using exact BPR loss."""

    set_seed(SEED)

    device = torch.device("cpu")

    item_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    users = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3],
        dtype=torch.long,
        device=device,
    )

    positive_items = torch.tensor(
        [0, 1, 2, 3, 0, 4, 3, 5],
        dtype=torch.long,
        device=device,
    )

    negative_items = torch.tensor(
        [3, 4, 0, 5, 2, 1, 0, 2],
        dtype=torch.long,
        device=device,
    )

    model = TinyHybridRecommender(
        n_users=4,
        n_items=6,
        n_features=3,
        latent_dim=8,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
        weight_decay=1e-6,
    )

    def calculate_loss() -> torch.Tensor:
        positive_scores = model.score(
            users,
            positive_items,
            item_features,
        )

        negative_scores = model.score(
            users,
            negative_items,
            item_features,
        )

        return -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

    with torch.no_grad():
        initial_loss = float(
            calculate_loss().item()
        )

    for _ in range(100):
        optimizer.zero_grad()

        loss = calculate_loss()

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training produced a non-finite loss."
            )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(
            calculate_loss().item()
        )

        all_items = torch.arange(
            6,
            dtype=torch.long,
            device=device,
        )

        repeated_user = torch.zeros(
            6,
            dtype=torch.long,
            device=device,
        )

        predictions = (
            model.score(
                repeated_user,
                all_items,
                item_features,
            )
            .cpu()
            .numpy()
        )

    if final_loss >= initial_loss:
        raise AssertionError(
            "The BPR training loss did not decrease."
        )

    if not np.isfinite(predictions).all():
        raise AssertionError(
            "The model produced invalid predictions."
        )

    return (
        initial_loss,
        final_loss,
        predictions,
    )


def main() -> None:
    first_run = run_once()
    second_run = run_once()

    np.testing.assert_allclose(
        first_run[2],
        second_run[2],
        rtol=0,
        atol=1e-7,
    )

    print("PyTorch hybrid recommender smoke test passed.")
    print(
        f"Initial BPR loss: {first_run[0]:.6f}"
    )
    print(
        f"Final BPR loss:   {first_run[1]:.6f}"
    )
    print(
        "Predictions:",
        first_run[2],
    )
    print(
        "Repeated execution is deterministic."
    )


if __name__ == "__main__":
    main()
