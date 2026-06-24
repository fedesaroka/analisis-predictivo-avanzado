from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = np.load(ART / "hybrid_model.npz")
    metadata = pd.read_parquet(ART / "anime_deploy.parquet")
    item_features = np.load(ART / "item_features.npy")
    seen = sparse.load_npz(ART / "seen_interactions.npz").tocsr()
    demo_users = json.loads((ART / "demo_users.json").read_text(encoding="utf-8"))

    title_to_idx = dict(zip(metadata["name"], metadata["item_idx"]))
    return {
        "user_repr": model["user_repr"],
        "item_repr": model["item_repr"],
        "user_bias": model["user_bias"],
        "item_bias": model["item_bias"],
        "user_ids": model["user_ids"],
        "item_ids": model["item_ids"],
        "metadata": metadata,
        "item_features": item_features,
        "seen": seen,
        "demo_users": demo_users,
        "title_to_idx": title_to_idx,
    }


def rank_existing_user(data, user_idx: int, n: int) -> pd.DataFrame:
    scores = (
        data["user_repr"][user_idx] @ data["item_repr"].T
        + data["user_bias"][user_idx]
        + data["item_bias"]
    ).astype(float)

    start, end = data["seen"].indptr[user_idx : user_idx + 2]
    scores[data["seen"].indices[start:end]] = -np.inf

    top = np.argpartition(scores, -n)[-n:]
    top = top[np.argsort(scores[top])[::-1]]

    result = data["metadata"].iloc[top][
        ["name", "genre", "type", "rating", "members"]
    ].copy()
    result.insert(0, "Score", np.round(scores[top], 3))
    result = result.rename(
        columns={
            "name": "Anime",
            "genre": "Géneros",
            "type": "Tipo",
            "rating": "Rating medio",
            "members": "Miembros",
        }
    )
    result.index = np.arange(1, len(result) + 1)
    return result


def shared_genres(selected_genres: set[str], candidate: str) -> str:
    candidate_genres = set(str(candidate).split(", "))
    overlap = sorted(selected_genres & candidate_genres)
    return ", ".join(overlap) if overlap else "Afinidad general de perfil"


def rank_new_user(data, favorites: list[str], n: int) -> pd.DataFrame:
    idx = np.array([data["title_to_idx"][title] for title in favorites], dtype=int)
    profile = data["item_features"][idx].mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    similarity = data["item_features"] @ profile

    # Small popularity tie-breaker. Similarity remains the dominant signal.
    members = np.log1p(data["metadata"]["members"].fillna(0).to_numpy(dtype=float))
    members = (members - members.min()) / max(1e-12, members.max() - members.min())
    scores = similarity + 0.03 * members
    scores[idx] = -np.inf

    top = np.argpartition(scores, -n)[-n:]
    top = top[np.argsort(scores[top])[::-1]]

    selected_genres: set[str] = set()
    for genre_text in data["metadata"].iloc[idx]["genre"]:
        selected_genres.update(str(genre_text).split(", "))

    result = data["metadata"].iloc[top][
        ["name", "genre", "type", "rating", "members"]
    ].copy()
    result.insert(0, "Afinidad", np.round(similarity[top], 3))
    result.insert(
        2,
        "Por qué",
        [shared_genres(selected_genres, g) for g in result["genre"]],
    )
    result = result.rename(
        columns={
            "name": "Anime",
            "genre": "Géneros",
            "type": "Tipo",
            "rating": "Rating medio",
            "members": "Miembros",
        }
    )
    result.index = np.arange(1, len(result) + 1)
    return result


data = load_artifacts()

st.title("🎌 Sistema híbrido de recomendación de anime")
st.markdown(
    "**TP2 — Análisis Predictivo Avanzado** · "
    "Factorización híbrida tipo LightFM + fallback basado en contenido"
)
st.caption(
    "El modelo fue entrenado offline. La aplicación carga representaciones compactas "
    "y genera recomendaciones sin reentrenar en la nube."
)

col1, col2, col3 = st.columns(3)
col1.metric("Usuarios modelados", f"{len(data['user_ids']):,}")
col2.metric("Animes modelados", f"{len(data['item_ids']):,}")
col3.metric("Features de contenido", f"{data['item_features'].shape[1]:,}")

st.divider()
known_tab, new_tab, method_tab = st.tabs(
    ["Usuario existente", "Usuario nuevo", "Cómo funciona"]
)

with known_tab:
    st.subheader("Recomendaciones personalizadas con historial")
    labels = {
        f"Usuario {entry['user_id']} · {entry['n_likes']} gustos positivos": entry
        for entry in data["demo_users"]
    }
    selected_label = st.selectbox("Elegí un usuario de demostración", list(labels))
    n_known = st.slider(
        "Cantidad de recomendaciones", 5, 20, 10, key="known_count"
    )
    entry = labels[selected_label]

    st.markdown("**Algunos favoritos de su historial**")
    st.write(" · ".join(entry["favorites"][:6]))

    known_result = rank_existing_user(data, int(entry["user_idx"]), n_known)
    st.markdown(f"**Top {n_known} animes no vistos**")
    st.dataframe(known_result, width="stretch")

with new_tab:
    st.subheader("Onboarding para un usuario sin historial")
    st.write(
        "Seleccioná entre 3 y 5 animes favoritos. El sistema construye un perfil "
        "de géneros y tipo de contenido para resolver el arranque en frío."
    )

    titles = data["metadata"].sort_values("members", ascending=False)["name"].tolist()
    favorites = st.multiselect(
        "Animes favoritos",
        titles,
        default=titles[:3],
        max_selections=5,
    )
    n_new = st.slider("Cantidad de recomendaciones", 5, 20, 10, key="new_count")

    if len(favorites) < 3:
        st.info("Seleccioná al menos tres títulos para construir un perfil más estable.")
    else:
        new_result = rank_new_user(data, favorites, n_new)
        st.markdown(f"**Top {n_new} recomendaciones de onboarding**")
        st.dataframe(new_result, width="stretch")

with method_tab:
    st.subheader("Arquitectura del entregable")
    st.markdown(
        """
        **Usuarios existentes.** El score combina un embedding del usuario, una
        representación del anime formada por identidad más metadata, y sesgos de
        usuario e ítem. Los títulos ya observados se excluyen antes de ordenar.

        **Usuarios nuevos.** Como todavía no existe embedding personal, se promedian
        las features de los títulos favoritos y se busca afinidad de contenido. Esta
        rama funciona como estrategia de cold start.

        **Validación.** El modelo se seleccionó con Precision@10 sobre un conjunto de
        validación por usuario y se evaluó una sola vez sobre un test reservado.
        """
    )
    st.info(
        "El deployment demuestra el uso del modelo. En producción debería integrarse "
        "con eventos de consumo, catálogo actualizado y un experimento A/B."
    )
