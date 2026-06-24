"""Create the notebook through metadata, metrics, and baselines."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from create_phase2_notebook import (
    main as create_phase2_notebook,
)


ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATH = (
    ROOT
    / "notebooks"
    / "TP2_sistema_recomendacion_anime.ipynb"
)


def markdown(
    source: str,
):
    return nbf.v4.new_markdown_cell(
        source.strip()
    )


def code(
    source: str,
):
    return nbf.v4.new_code_cell(
        source.strip()
    )


def main() -> None:
    create_phase2_notebook()

    notebook = nbf.read(
        NOTEBOOK_PATH,
        as_version=4,
    )

    notebook.cells.extend(
        [
            markdown(
                """
## 6. Representación de metadatos

Cada anime se representa mediante variables binarias de género y tipo. Los nombres de las características y los índices de los ítems se ordenan de forma determinística.

Las filas se normalizan con norma L2. Esta normalización permite interpretar el producto punto entre dos vectores como una similitud coseno y evita favorecer automáticamente a los títulos asociados con más géneros.
                """
            ),
            code(
                """
from anime_recommender import (
    build_item_features,
)

item_features = build_item_features(
    prepared.catalog
)

display(
    pd.DataFrame(
        {
            "Measure": list(
                item_features.summary.keys()
            ),
            "Value": list(
                item_features.summary.values()
            ),
        }
    )
)
                """
            ),
            code(
                """
feature_frequencies = np.asarray(
    item_features.matrix.sum(axis=0)
).ravel()

feature_frequency_table = (
    pd.DataFrame(
        {
            "feature": (
                item_features.feature_names
            ),
            "anime_count": (
                feature_frequencies.astype(
                    int
                )
            ),
        }
    )
    .sort_values(
        [
            "anime_count",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

display(
    feature_frequency_table.head(20)
)
                """
            ),
            code(
                """
top_features = (
    feature_frequency_table
    .head(20)
    .sort_values(
        "anime_count"
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    top_features["feature"],
    top_features["anime_count"],
)

ax.set_title(
    "Most frequent genre and type features"
)
ax.set_xlabel(
    "Anime titles"
)
ax.set_ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()
                """
            ),
            markdown(
                """
## 7. Protocolo de evaluación

Todos los modelos generan un puntaje para cada combinación usuario-anime. Antes del ranking se excluyen las interacciones positivas observadas en entrenamiento.

La métrica principal es Precision@10. También se reportan Recall@10, NDCG@10, Hit Rate@10, AUC con negativos muestreados, cobertura del catálogo y diversidad interna.

Los empates de puntaje se resuelven por índice de ítem. Esto evita que el orden de resultados dependa de implementaciones internas o del sistema operativo.
                """
            ),
            markdown(
                """
## 8. Baselines

Se utilizan dos referencias mínimas.

**Popularidad**

Ordena los títulos según la cantidad de interacciones positivas observadas en entrenamiento. Permite determinar si los modelos personalizados agregan valor frente a una estrategia global.

**Contenido**

Construye un perfil por usuario sumando los vectores normalizados de sus títulos positivos de entrenamiento. Luego ordena el catálogo por similitud coseno con dicho perfil.
                """
            ),
            code(
                """
from anime_recommender import (
    evaluate_baselines,
    recommendation_examples,
)

(
    baseline_validation_results,
    baseline_recommendations,
) = evaluate_baselines(
    prepared=prepared,
    item_features=item_features,
    config=config,
)

display(
    baseline_validation_results
)
                """
            ),
            code(
                """
metric_columns = [
    column
    for column in (
        baseline_validation_results.columns
    )
    if column != "model"
]

baseline_plot_data = (
    baseline_validation_results
    .set_index("model")[
        [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "hit_rate_at_10",
        ]
    ]
    .T
)

ax = baseline_plot_data.plot(
    kind="bar",
    figsize=(10, 5),
)

ax.set_title(
    "Validation ranking metrics for baseline models"
)
ax.set_xlabel(
    "Metric"
)
ax.set_ylabel(
    "Value"
)
ax.set_ylim(
    0,
    max(
        0.05,
        baseline_plot_data.to_numpy().max()
        * 1.15,
    ),
)
ax.tick_params(
    axis="x",
    rotation=0,
)

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
example_user_indices = [
    0,
    len(prepared.user_ids) // 2,
    len(prepared.user_ids) - 1,
]

baseline_examples = recommendation_examples(
    prepared=prepared,
    recommendations_by_model=(
        baseline_recommendations
    ),
    user_indices=example_user_indices,
)

display(
    baseline_examples[
        [
            "model",
            "user_id",
            "rank",
            "title",
            "genre",
            "type",
            "validation_hit",
        ]
    ]
)
                """
            ),
            code(
                """
for model_name, recommendations in (
    baseline_recommendations.items()
):
    for user_index, item_indices in enumerate(
        recommendations
    ):
        assert (
            prepared.train_matrix[
                user_index,
                item_indices,
            ].nnz
            == 0
        )

assert (
    baseline_validation_results
    .drop(columns="model")
    .apply(
        lambda column: column.between(
            0,
            1,
        ).all()
    )
    .all()
)

print(
    "Baseline ranking and metric checks passed."
)
                """
            ),
            markdown(
                """
## 9. Conclusión de los baselines

La popularidad establece el rendimiento de una recomendación global sin personalización. El modelo de contenido incorpora preferencias individuales, aunque solo puede utilizar similitud explícita de género y tipo.

Los modelos colaborativos e híbridos posteriores deberán superar estas referencias en Precision@10 para justificar su mayor complejidad.
                """
            ),
        ]
    )

    nbf.write(
        notebook,
        NOTEBOOK_PATH,
    )

    print(
        f"Phase 3 notebook created: "
        f"{NOTEBOOK_PATH}"
    )


if __name__ == "__main__":
    main()
