"""Create the notebook through final test evaluation."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from create_phase5_notebook import (
    main as create_phase5_notebook,
)


ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATH = (
    ROOT
    / "notebooks"
    / "TP2_sistema_recomendacion_anime.ipynb"
)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(
        source.strip()
    )


def code(source: str):
    return nbf.v4.new_code_cell(
        source.strip()
    )


def main() -> None:
    create_phase5_notebook()

    notebook = nbf.read(
        NOTEBOOK_PATH,
        as_version=4,
    )

    notebook.cells.extend(
        [
            markdown(
                """
## 16. Entrenamiento final y evaluación de prueba

La configuración seleccionada se congeló antes de acceder al conjunto de prueba. Luego se combinaron entrenamiento y validación, produciendo el conjunto final de aprendizaje.

La evaluación de prueba se realiza en una única fase final. Se reportan cuatro referencias previamente definidas:

- Popularidad
- Contenido
- Modelo híbrido WARP-style por defecto
- Modelo híbrido WARP-style optimizado

El modelo optimizado permanece seleccionado independientemente del resultado de prueba. Cambiar la decisión después de observar este resultado convertiría prueba en una nueva validación.
                """
            ),
            code(
                """
FINAL_RESULTS_DIR = (
    ROOT
    / "results"
    / "final_model"
)

ARTIFACTS_DIR = (
    ROOT
    / "artifacts"
)

required_final_files = [
    FINAL_RESULTS_DIR
    / "test_metrics.csv",
    FINAL_RESULTS_DIR
    / "training_history.csv",
    FINAL_RESULTS_DIR
    / "validation_test_comparison.csv",
    FINAL_RESULTS_DIR
    / "run_summary.json",
    ARTIFACTS_DIR
    / "artifact_manifest.json",
    ARTIFACTS_DIR
    / "model_components.npz",
    ARTIFACTS_DIR
    / "seen_final.npz",
]

assert all(
    path.exists()
    for path in required_final_files
), (
    "Final outputs are missing. Run "
    "scripts/train_final_model.py with the explicit "
    "--confirm-test-access flag before executing the notebook."
)

print(
    "Using the frozen final evaluation outputs."
)
                """
            ),
            code(
                """
final_test_metrics = pd.read_csv(
    FINAL_RESULTS_DIR
    / "test_metrics.csv"
)

final_training_history = pd.read_csv(
    FINAL_RESULTS_DIR
    / "training_history.csv"
)

validation_test_comparison = pd.read_csv(
    FINAL_RESULTS_DIR
    / "validation_test_comparison.csv"
)

final_examples = pd.read_csv(
    FINAL_RESULTS_DIR
    / "recommendation_examples.csv"
)

with (
    FINAL_RESULTS_DIR
    / "run_summary.json"
).open(
    "r",
    encoding="utf-8",
) as file:
    final_run_summary = json.load(file)

with (
    ARTIFACTS_DIR
    / "artifact_manifest.json"
).open(
    "r",
    encoding="utf-8",
) as file:
    artifact_manifest = json.load(file)

display(
    final_test_metrics
)
                """
            ),
            code(
                """
test_metric_plot = (
    final_test_metrics
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

ax = test_metric_plot.plot(
    kind="bar",
    figsize=(11, 5),
)

ax.set_title(
    "Final held-out test performance"
)
ax.set_xlabel(
    "Metric"
)
ax.set_ylabel(
    "Value"
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
display(
    validation_test_comparison[
        [
            "model",
            "validation_precision_at_10",
            "precision_at_10",
            "precision_generalization_gap",
            "validation_ndcg_at_10",
            "ndcg_at_10",
        ]
    ]
)
                """
            ),
            code(
                """
fig, ax = plt.subplots(
    figsize=(10, 5)
)

for model_name, model_history in (
    final_training_history.groupby(
        "model",
        sort=True,
    )
):
    ax.plot(
        model_history["epoch"],
        model_history["loss"],
        marker="o",
        label=model_name,
    )

ax.set_title(
    "Final train-plus-validation learning curves"
)
ax.set_xlabel(
    "Epoch"
)
ax.set_ylabel(
    "Training objective"
)
ax.legend()

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
print(
    "Frozen selected model:",
    final_run_summary[
        "selected_model_before_test"
    ],
)

print(
    "Selection changed after test:",
    final_run_summary[
        "selection_changed_after_test"
    ],
)

print(
    "Test set accessed:",
    final_run_summary[
        "test_set_accessed"
    ],
)

print(
    "NumPy/PyTorch maximum score difference:",
    final_run_summary[
        "score_parity_max_abs_error"
    ],
)

print(
    "Deployment requires PyTorch:",
    artifact_manifest[
        "deployment_requires_pytorch"
    ],
)
                """
            ),
            code(
                """
display(
    final_examples[
        [
            "user_id",
            "positive_history_count",
            "rank",
            "title",
            "genre",
            "type",
            "test_hit",
        ]
    ]
)
                """
            ),
            code(
                """
assert (
    final_run_summary[
        "selected_model_before_test"
    ]
    == "tuned_hybrid_warp_style"
)

assert (
    final_run_summary[
        "selection_changed_after_test"
    ]
    is False
)

assert (
    final_run_summary[
        "test_set_accessed"
    ]
    is True
)

assert (
    artifact_manifest[
        "deployment_requires_pytorch"
    ]
    is False
)

assert (
    artifact_manifest[
        "cohort_split_sha256"
    ]
    == prepared.summary[
        "cohort_split_sha256"
    ]
)

assert (
    artifact_manifest[
        "score_parity_max_abs_error"
    ]
    <= 1e-4
)

print(
    "Final evaluation and artifact checks passed."
)
                """
            ),
            markdown(
                """
## 17. Conclusiones

El sistema final utiliza una arquitectura híbrida de factores latentes y una pérdida de negativos difíciles inspirada en WARP. La selección se realizó mediante Precision@10 de validación y optimización bayesiana.

El conjunto de prueba se reservó hasta congelar arquitectura, objetivo e hiperparámetros. Después de la evaluación final, la selección permaneció inalterada.

Para el despliegue se exportaron las representaciones efectivas, sesgos, catálogo, mappings e interacciones vistas. La aplicación utiliza NumPy y SciPy, sin cargar PyTorch.

### Limitaciones

- El dataset carece de marcas temporales confiables.
- Las interacciones positivas se definen mediante un umbral fijo de calificación.
- Los negativos muestreados pueden incluir títulos que un usuario disfrutaría pero todavía no calificó.
- La variante WARP-style es una implementación propia inspirada en WARP, no una reproducción exacta de LightFM.
- La evaluación es offline y no mide clics, tiempo de reproducción ni retención real.

### Mejoras futuras

- Incorporar timestamps y validación temporal.
- Utilizar feedback implícito real.
- Añadir descripciones, estudios, autores y embeddings de texto.
- Evaluar usuarios completamente nuevos.
- Realizar experimentos A/B en una plataforma real.
                """
            ),
        ]
    )

    nbf.write(
        notebook,
        NOTEBOOK_PATH,
    )

    print(
        f"Phase 6 notebook created: "
        f"{NOTEBOOK_PATH}"
    )


if __name__ == "__main__":
    main()
