"""Core implementation for the anime recommendation project.

The package root intentionally excludes PyTorch-dependent modules.
Training processes import anime_recommender.training explicitly after
PyTorch has been initialized.
"""

from anime_recommender.baselines import (
    content_score_matrix,
    evaluate_baselines,
    popularity_score_matrix,
    recommendation_examples,
)
from anime_recommender.config import (
    load_experiment_config,
)
from anime_recommender.data import (
    PreparedData,
    prepare_data,
)
from anime_recommender.evaluation import (
    evaluate_recommendations,
    evaluate_score_matrix,
    rank_top_k,
    sampled_auc,
)
from anime_recommender.features import (
    ItemFeatureData,
    build_item_features,
)
from anime_recommender.inference import (
    DeploymentArtifacts,
    load_deployment_artifacts,
    recommend_all_known_users,
    recommend_known_user,
    score_known_user,
)


__all__ = [
    "DeploymentArtifacts",
    "ItemFeatureData",
    "PreparedData",
    "build_item_features",
    "content_score_matrix",
    "evaluate_baselines",
    "evaluate_recommendations",
    "evaluate_score_matrix",
    "load_deployment_artifacts",
    "load_experiment_config",
    "popularity_score_matrix",
    "prepare_data",
    "rank_top_k",
    "recommend_all_known_users",
    "recommend_known_user",
    "recommendation_examples",
    "sampled_auc",
    "score_known_user",
]
