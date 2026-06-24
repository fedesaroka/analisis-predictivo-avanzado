"""PyTorch latent-factor architectures for the recommender."""

from __future__ import annotations

import torch
from torch import nn


class HybridLatentFactorModel(nn.Module):
    """Collaborative or hybrid latent-factor recommender.

    Collaborative item representation:
        q_i

    Hybrid item representation:
        q_i + metadata_weight * W x_i

    Final score:
        global_bias
        + user_bias
        + item_bias
        + user_vector dot item_vector
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_features: int,
        latent_dim: int,
        use_metadata: bool,
        metadata_weight: float,
    ) -> None:
        super().__init__()

        if n_users <= 0:
            raise ValueError(
                "n_users must be positive."
            )

        if n_items <= 0:
            raise ValueError(
                "n_items must be positive."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        if use_metadata and n_features <= 0:
            raise ValueError(
                "Hybrid models require metadata features."
            )

        if metadata_weight <= 0:
            raise ValueError(
                "metadata_weight must be positive."
            )

        self.n_users = int(
            n_users
        )

        self.n_items = int(
            n_items
        )

        self.n_features = int(
            n_features
        )

        self.latent_dim = int(
            latent_dim
        )

        self.use_metadata = bool(
            use_metadata
        )

        self.metadata_weight = float(
            metadata_weight
        )

        self.user_embedding = nn.Embedding(
            self.n_users,
            self.latent_dim,
        )

        self.item_embedding = nn.Embedding(
            self.n_items,
            self.latent_dim,
        )

        self.user_bias = nn.Embedding(
            self.n_users,
            1,
        )

        self.item_bias = nn.Embedding(
            self.n_items,
            1,
        )

        self.global_bias = nn.Parameter(
            torch.zeros(
                1,
                dtype=torch.float32,
            )
        )

        if self.use_metadata:
            self.metadata_projection = nn.Linear(
                self.n_features,
                self.latent_dim,
                bias=False,
            )
        else:
            self.metadata_projection = None

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        """Initialize all trainable parameters reproducibly."""

        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.05,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.05,
        )

        nn.init.zeros_(
            self.user_bias.weight
        )

        nn.init.zeros_(
            self.item_bias.weight
        )

        nn.init.zeros_(
            self.global_bias
        )

        if self.metadata_projection is not None:
            nn.init.normal_(
                self.metadata_projection.weight,
                mean=0.0,
                std=0.02,
            )

    def _pair_item_vectors(
        self,
        items: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Create item representations for selected item indices."""

        item_vectors = self.item_embedding(
            items
        )

        if self.use_metadata:
            if item_features.ndim != 2:
                raise ValueError(
                    "item_features must be two-dimensional."
                )

            metadata_vectors = (
                self.metadata_projection(
                    item_features[
                        items
                    ]
                )
            )

            item_vectors = (
                item_vectors
                + self.metadata_weight
                * metadata_vectors
            )

        return item_vectors

    def effective_item_embeddings(
        self,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Create representations for the complete item catalog."""

        item_vectors = (
            self.item_embedding.weight
        )

        if self.use_metadata:
            metadata_vectors = (
                self.metadata_projection(
                    item_features
                )
            )

            item_vectors = (
                item_vectors
                + self.metadata_weight
                * metadata_vectors
            )

        return item_vectors

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score corresponding user-item pairs."""

        if users.shape != items.shape:
            raise ValueError(
                "users and items must have the same shape."
            )

        user_vectors = self.user_embedding(
            users
        )

        item_vectors = self._pair_item_vectors(
            items,
            item_features,
        )

        interaction_scores = (
            user_vectors
            * item_vectors
        ).sum(dim=1)

        user_biases = (
            self.user_bias(
                users
            )
            .squeeze(-1)
        )

        item_biases = (
            self.item_bias(
                items
            )
            .squeeze(-1)
        )

        return (
            interaction_scores
            + user_biases
            + item_biases
            + self.global_bias
        )

    def score_all_items(
        self,
        users: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every catalog item for a batch of users."""

        user_vectors = self.user_embedding(
            users
        )

        item_vectors = (
            self.effective_item_embeddings(
                item_features
            )
        )

        scores = (
            user_vectors
            @ item_vectors.T
        )

        scores = (
            scores
            + self.user_bias(
                users
            )
            + self.item_bias.weight.T
            + self.global_bias
        )

        return scores
