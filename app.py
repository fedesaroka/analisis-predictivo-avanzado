"""Streamlit application for the final anime recommender."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

from anime_recommender import (
    build_content_profile,
    load_deployment_artifacts,
    recommend_known_user,
    recommend_new_user,
)


ROOT = Path(__file__).resolve().parent

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
)


st.set_page_config(
    page_title="Recomendador de Anime",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #172033;
    }

    .app-subtitle {
        color: #536075;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.2rem;
    }

    .mode-note {
        background: #f4f6fa;
        border-left: 4px solid #3157d5;
        border-radius: 0.35rem;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }

    .technical-note {
        color: #667085;
        font-size: 0.88rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(
    show_spinner=(
        "Cargando el modelo y los artefactos..."
    )
)
def load_application_resources():
    """Load every immutable application resource once."""

    artifacts = load_deployment_artifacts(
        ARTIFACT_DIRECTORY
    )

    item_features = sparse.load_npz(
        ARTIFACT_DIRECTORY
        / "item_features.npz"
    ).tocsr()

    with (
        ARTIFACT_DIRECTORY
        / "feature_names.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_information = json.load(
            file
        )

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

    if item_features.shape[0] != len(
        artifacts.catalog
    ):
        raise RuntimeError(
            "La matriz de contenido no coincide "
            "con el catálogo desplegado."
        )

    return (
        artifacts,
        item_features,
        feature_information,
        demo_users,
    )


def format_anime_option(
    item_index: int,
    item_labels: dict[int, str],
) -> str:
    """Format one catalog option for Streamlit widgets."""

    return item_labels[
        int(item_index)
    ]


def recommendation_table(
    recommendations: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Create a compact user-facing recommendation table."""

    preferred_columns = [
        "rank",
        score_column,
        "name",
        "genre",
        "type",
        "rating",
        "members",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in recommendations.columns
    ]

    table = recommendations[
        available_columns
    ].copy()

    table = table.rename(
        columns={
            "rank": "Puesto",
            score_column: "Puntaje",
            "name": "Anime",
            "genre": "Géneros",
            "type": "Tipo",
            "rating": "Rating global",
            "members": "Miembros",
        }
    )

    if "Puntaje" in table:
        table["Puntaje"] = pd.to_numeric(
            table["Puntaje"],
            errors="coerce",
        ).round(4)

    if "Rating global" in table:
        table["Rating global"] = pd.to_numeric(
            table["Rating global"],
            errors="coerce",
        ).round(2)

    if "Miembros" in table:
        table["Miembros"] = (
            pd.to_numeric(
                table["Miembros"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

    return table


def display_recommendations(
    recommendations: pd.DataFrame,
    score_column: str,
    heading: str,
) -> None:
    """Display recommendations as a table and score chart."""

    st.subheader(heading)

    table = recommendation_table(
        recommendations=(
            recommendations
        ),
        score_column=score_column,
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    chart_data = (
        table[
            [
                "Anime",
                "Puntaje",
            ]
        ]
        .set_index("Anime")
        .sort_values("Puntaje")
    )

    st.bar_chart(
        chart_data,
        use_container_width=True,
    )


def known_user_history(
    artifacts,
    user_index: int,
    maximum_rows: int = 15,
) -> pd.DataFrame:
    """Return a readable sample of one known user's history."""

    row_start = artifacts.seen_matrix.indptr[
        user_index
    ]

    row_end = artifacts.seen_matrix.indptr[
        user_index + 1
    ]

    seen_item_indices = (
        artifacts.seen_matrix.indices[
            row_start:row_end
        ]
    )

    history = (
        artifacts.catalog
        .loc[
            seen_item_indices
        ]
        .copy()
    )

    sorting_columns = [
        column
        for column in [
            "rating",
            "members",
            "name",
        ]
        if column in history.columns
    ]

    ascending_values = [
        False
        if column in {
            "rating",
            "members",
        }
        else True
        for column in sorting_columns
    ]

    if sorting_columns:
        history = history.sort_values(
            sorting_columns,
            ascending=ascending_values,
            na_position="last",
        )

    visible_columns = [
        column
        for column in [
            "name",
            "genre",
            "type",
            "rating",
            "members",
        ]
        if column in history.columns
    ]

    history = (
        history[
            visible_columns
        ]
        .head(maximum_rows)
        .rename(
            columns={
                "name": "Anime",
                "genre": "Géneros",
                "type": "Tipo",
                "rating": "Rating global",
                "members": "Miembros",
            }
        )
    )

    return history.reset_index(
        drop=True
    )


def profile_preference_labels(
    item_features: sparse.csr_matrix,
    favorite_item_indices: list[int],
    feature_names: list[str],
    maximum_labels: int = 8,
) -> list[str]:
    """Return the strongest content dimensions in a profile."""

    profile = build_content_profile(
        normalized_item_features=(
            item_features
        ),
        favorite_item_indices=(
            favorite_item_indices
        ),
    )

    ranking = np.lexsort(
        (
            np.arange(
                len(profile),
                dtype=np.int64,
            ),
            -profile,
        )
    )

    labels: list[str] = []

    for feature_index in ranking:
        feature_weight = float(
            profile[
                feature_index
            ]
        )

        if feature_weight <= 0:
            continue

        raw_name = str(
            feature_names[
                feature_index
            ]
        )

        if raw_name.startswith(
            "genre:"
        ):
            label = (
                "Género · "
                + raw_name.split(
                    ":",
                    1,
                )[1].title()
            )
        elif raw_name.startswith(
            "type:"
        ):
            label = (
                "Tipo · "
                + raw_name.split(
                    ":",
                    1,
                )[1].upper()
            )
        else:
            label = raw_name

        labels.append(
            label
        )

        if len(labels) >= maximum_labels:
            break

    return labels


(
    artifacts,
    item_features,
    feature_information,
    demo_users,
) = load_application_resources()


catalog = artifacts.catalog.copy()

catalog_lookup = (
    catalog
    .set_index("item_index")
)

item_labels = {
    int(row.item_index): (
        f"{row.name} · {row.type} · "
        f"anime_id {int(row.anime_id)}"
    )
    for row in catalog[
        [
            "item_index",
            "anime_id",
            "name",
            "type",
        ]
    ].itertuples(
        index=False
    )
}

user_id_to_index = {
    int(user_id): int(user_index)
    for user_index, user_id
    in enumerate(
        artifacts.user_index_to_id
    )
}

demo_labels_by_user_id = {
    int(user["user_id"]): (
        f"Usuario {int(user['user_id'])} · "
        f"{int(user['positive_history_count'])} "
        "títulos positivos · demo"
    )
    for user in demo_users
}

all_user_ids = [
    int(user_id)
    for user_id in (
        artifacts.user_index_to_id
    )
]

default_user_id = int(
    demo_users[0][
        "user_id"
    ]
)

default_user_position = (
    all_user_ids.index(
        default_user_id
    )
)


st.title(
    "🎌 Sistema de Recomendación de Anime"
)

st.markdown(
    """
<div class="app-subtitle">
Modelo híbrido de factores latentes con metadatos de género y tipo.
</div>
    """,
    unsafe_allow_html=True,
)


selected_test_metrics = (
    artifacts.manifest[
        "selected_test_metrics"
    ]
)

metric_columns = st.columns(
    4
)

metric_columns[0].metric(
    "Precision@10",
    f"{float(selected_test_metrics['precision_at_10']):.1%}",
)

metric_columns[1].metric(
    "Recall@10",
    f"{float(selected_test_metrics['recall_at_10']):.1%}",
)

metric_columns[2].metric(
    "NDCG@10",
    f"{float(selected_test_metrics['ndcg_at_10']):.3f}",
)

metric_columns[3].metric(
    "Hit Rate@10",
    f"{float(selected_test_metrics['hit_rate_at_10']):.1%}",
)


with st.sidebar:
    st.header(
        "Acerca del modelo"
    )

    st.write(
        "**Modelo:** híbrido WARP-style optimizado"
    )

    st.write(
        f"**Usuarios entrenados:** "
        f"{int(artifacts.manifest['n_users']):,}"
    )

    st.write(
        f"**Anime en catálogo:** "
        f"{int(artifacts.manifest['n_items']):,}"
    )

    st.write(
        f"**Dimensión latente:** "
        f"{int(artifacts.manifest['latent_dim'])}"
    )

    st.write(
        f"**Interacciones de entrenamiento:** "
        f"{int(artifacts.manifest['seen_interactions']):,}"
    )

    st.divider()

    st.caption(
        "El prototipo utiliza interacciones positivas "
        "definidas por ratings mayores o iguales a 8."
    )

    st.caption(
        "La aplicación realiza inferencia con NumPy y SciPy. "
        "No carga PyTorch ni reentrena el modelo."
    )


existing_user_tab, new_user_tab = st.tabs(
    [
        "👤 Usuario existente",
        "✨ Usuario nuevo",
    ]
)


with existing_user_tab:
    st.markdown(
        """
<div class="mode-note">
Este modo utiliza el embedding aprendido para un usuario presente
en el conjunto de entrenamiento. Los títulos ya observados se
excluyen automáticamente.
</div>
        """,
        unsafe_allow_html=True,
    )

    selected_user_id = st.selectbox(
        "Seleccioná un usuario",
        options=all_user_ids,
        index=default_user_position,
        format_func=lambda user_id: (
            demo_labels_by_user_id.get(
                int(user_id),
                f"Usuario {int(user_id)}",
            )
        ),
        key="known_user_selector",
    )

    known_user_k = st.slider(
        "Cantidad de recomendaciones",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="known_user_k",
    )

    selected_user_index = (
        user_id_to_index[
            int(selected_user_id)
        ]
    )

    seen_count = int(
        artifacts.seen_matrix.indptr[
            selected_user_index + 1
        ]
        - artifacts.seen_matrix.indptr[
            selected_user_index
        ]
    )

    information_columns = st.columns(
        2
    )

    information_columns[0].metric(
        "Usuario",
        int(selected_user_id),
    )

    information_columns[1].metric(
        "Títulos positivos usados",
        seen_count,
    )

    known_recommendations = (
        recommend_known_user(
            artifacts=artifacts,
            user_index=(
                selected_user_index
            ),
            k=known_user_k,
        )
    )

    display_recommendations(
        recommendations=(
            known_recommendations
        ),
        score_column="score",
        heading=(
            f"Top {known_user_k} "
            "recomendaciones personalizadas"
        ),
    )

    with st.expander(
        "Ver una muestra del historial positivo"
    ):
        st.dataframe(
            known_user_history(
                artifacts=artifacts,
                user_index=(
                    selected_user_index
                ),
            ),
            use_container_width=True,
            hide_index=True,
        )


with new_user_tab:
    st.markdown(
        """
<div class="mode-note">
Este modo resuelve el problema de usuario nuevo. Como todavía no
existe un embedding colaborativo, se construye un perfil de contenido
a partir de los anime favoritos seleccionados.
</div>
        """,
        unsafe_allow_html=True,
    )

    selected_favorite_indices = (
        st.multiselect(
            "Seleccioná entre 3 y 10 anime favoritos",
            options=[
                int(value)
                for value in catalog[
                    "item_index"
                ]
            ],
            format_func=lambda item_index: (
                format_anime_option(
                    int(item_index),
                    item_labels,
                )
            ),
            key="new_user_favorites",
        )
    )

    new_user_k = st.slider(
        "Cantidad de recomendaciones",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="new_user_k",
    )

    if len(
        selected_favorite_indices
    ) < 3:
        st.info(
            "Seleccioná al menos tres títulos para "
            "construir un perfil suficientemente informativo."
        )

    elif len(
        selected_favorite_indices
    ) > 10:
        st.warning(
            "Seleccioná como máximo diez títulos."
        )

    else:
        selected_favorite_indices = [
            int(value)
            for value in (
                selected_favorite_indices
            )
        ]

        favorite_rows = (
            catalog_lookup
            .loc[
                selected_favorite_indices
            ][
                [
                    "name",
                    "genre",
                    "type",
                ]
            ]
            .reset_index(drop=True)
            .rename(
                columns={
                    "name": "Anime favorito",
                    "genre": "Géneros",
                    "type": "Tipo",
                }
            )
        )

        with st.expander(
            "Ver títulos usados para construir el perfil"
        ):
            st.dataframe(
                favorite_rows,
                use_container_width=True,
                hide_index=True,
            )

        preference_labels = (
            profile_preference_labels(
                item_features=(
                    item_features
                ),
                favorite_item_indices=(
                    selected_favorite_indices
                ),
                feature_names=(
                    feature_information[
                        "feature_names"
                    ]
                ),
            )
        )

        st.markdown(
            "**Preferencias inferidas:** "
            + " · ".join(
                preference_labels
            )
        )

        cold_start_recommendations = (
            recommend_new_user(
                artifacts=artifacts,
                normalized_item_features=(
                    item_features
                ),
                favorite_item_indices=(
                    selected_favorite_indices
                ),
                k=new_user_k,
            )
        )

        display_recommendations(
            recommendations=(
                cold_start_recommendations
            ),
            score_column="similarity",
            heading=(
                f"Top {new_user_k} "
                "recomendaciones por contenido"
            ),
        )


st.divider()

st.markdown(
    """
<div class="technical-note">
Prototipo académico offline. Las métricas mostradas corresponden al
conjunto de prueba reservado. Las recomendaciones para usuarios nuevos
se basan únicamente en similitud de género y tipo.
</div>
    """,
    unsafe_allow_html=True,
)
