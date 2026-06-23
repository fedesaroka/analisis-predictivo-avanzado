import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide"
)

# ── Carga y preparación de datos ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    anime = pd.read_csv("data/anime.csv")
    anime_cb = anime.dropna(subset=["genre"]).copy().reset_index(drop=True)
    anime_cb["genre_clean"] = anime_cb["genre"].str.replace(", ", " ").str.lower()
    return anime, anime_cb

@st.cache_data
def calcular_similitudes(anime_cb):
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(anime_cb["genre_clean"])
    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
    return similarity_matrix

def recomendar(nombre, anime_cb, similarity_matrix, n=10):
    matches = anime_cb[anime_cb["name"].str.lower() == nombre.lower()]
    if matches.empty:
        matches = anime_cb[anime_cb["name"].str.lower().str.contains(nombre.lower(), na=False)]
    if matches.empty:
        return None, None

    idx = matches.index[0]
    anime_info = anime_cb.loc[idx]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]

    indices = [i for i, _ in sim_scores]
    scores  = [s for _, s in sim_scores]

    resultado = anime_cb.iloc[indices][["name", "genre", "type", "rating", "members"]].copy()
    resultado.insert(0, "Similitud", [round(s, 3) for s in scores])
    resultado = resultado.rename(columns={
        "name": "Anime", "genre": "Géneros",
        "type": "Tipo", "rating": "Rating", "members": "Miembros"
    })
    resultado = resultado.reset_index(drop=True)
    resultado.index += 1
    return anime_info, resultado

# ── Carga ────────────────────────────────────────────────────────────────────
anime, anime_cb = cargar_datos()
similarity_matrix = calcular_similitudes(anime_cb)
nombres = sorted(anime_cb["name"].dropna().unique().tolist())

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🎌 Sistema de Recomendación de Anime")
st.markdown("**TP2 — Análisis Predictivo Avanzado** | Content-Based Filtering (TF-IDF + Cosine Similarity)")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    seleccion = st.selectbox(
        "Buscá un anime:",
        options=[""] + nombres,
        index=0,
        placeholder="Escribí para filtrar..."
    )
    n_recs = st.slider("Cantidad de recomendaciones", min_value=5, max_value=20, value=10)

with col2:
    st.markdown("### Estadísticas del dataset")
    st.metric("Animes totales", f"{len(anime):,}")
    st.metric("Con género definido", f"{len(anime_cb):,}")
    st.metric("Géneros únicos", anime_cb["genre"].str.split(", ").explode().nunique())

st.divider()

if seleccion:
    anime_info, resultado = recomendar(seleccion, anime_cb, similarity_matrix, n=n_recs)

    if resultado is None:
        st.error(f'No se encontró "{seleccion}" en el dataset.')
    else:
        # Info del anime seleccionado
        st.subheader(f"📺 {anime_info['name']}")
        cols = st.columns(4)
        cols[0].metric("Tipo", anime_info["type"] if pd.notna(anime_info["type"]) else "—")
        cols[1].metric("Rating", f"{anime_info['rating']:.2f}" if pd.notna(anime_info["rating"]) else "—")
        cols[2].metric("Episodios", int(anime_info["episodes"]) if pd.notna(anime_info["episodes"]) and str(anime_info["episodes"]).isdigit() else "—")
        cols[3].metric("Miembros", f"{int(anime_info['members']):,}" if pd.notna(anime_info["members"]) else "—")
        st.caption(f"**Géneros:** {anime_info['genre']}")

        st.divider()
        st.subheader(f"Top {n_recs} recomendaciones")
        st.dataframe(resultado, use_container_width=True)

        # Gráfico de similitudes
        st.subheader("Similitud coseno de las recomendaciones")
        chart_data = resultado[["Anime", "Similitud"]].set_index("Anime")
        st.bar_chart(chart_data)

else:
    st.info("Seleccioná un anime arriba para ver recomendaciones.")

    # Mostramos los más populares como pantalla de inicio
    st.subheader("🏆 Animes más populares del dataset")
    top_popular = (
        anime.dropna(subset=["members", "rating"])
        .nlargest(10, "members")[["name", "genre", "type", "rating", "members"]]
        .rename(columns={"name": "Anime", "genre": "Géneros", "type": "Tipo",
                         "rating": "Rating", "members": "Miembros"})
        .reset_index(drop=True)
    )
    top_popular.index += 1
    st.dataframe(top_popular, use_container_width=True)
