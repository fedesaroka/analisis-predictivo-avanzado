"""Create the notebook through default neural-model comparison."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from create_phase3_notebook import (
    main as create_phase3_notebook,
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
    create_phase3_notebook()

    notebook = nbf.read(
        NOTEBOOK_PATH,
        as_version=4,
    )

    notebook.cells.extend(
        [
            markdown(
                """
## 10. Modelos de factores latentes

Se implementa una arquitectura común con embeddings de usuario e ítem, sesgos y producto interno en un espacio latente.

La variante colaborativa representa cada anime mediante un embedding libre. La variante híbrida suma a ese embedding una proyección aprendida de sus géneros y tipo.

\[
s(u,i)=b+b_u+b_i+p_u^\top(q_i+\alpha Wx_i)
\]

Esta estructura permite medir directamente el aporte de los metadatos manteniendo constante el resto de la arquitectura.
                """
            ),
            markdown(
                """
## 11. Objetivos de entrenamiento

Se comparan tres funciones objetivo.

**Logistic**

Clasifica interacciones observadas como positivas y negativos muestreados como negativos.

**BPR**

Maximiza la diferencia de puntaje entre un ítem positivo y un negativo muestreado.

\[
-\log \sigma(s(u,i^+)-s(u,i^-))
\]

**WARP-style hard negative**

Muestrea negativos secuencialmente hasta encontrar uno que viola un margen. El error recibe un peso armónico basado en el rango aproximado. Es una implementación propia inspirada en WARP, por lo que se evita presentarla como el algoritmo WARP exacto de LightFM.
                """
            ),
            markdown(
                """
## 12. Ejecución externa de PyTorch

En este equipo, PyTorch funciona correctamente en el proceso Conda activado, pero la inicialización de sus DLL falla dentro del subproceso del kernel Jupyter de Windows.

Por ese motivo, el entrenamiento se ejecuta mediante un script independiente dentro del mismo entorno reproducible. El notebook orquesta la ejecución y luego carga todos los resultados. La implementación, los parámetros y las salidas permanecen dentro del repositorio.

La variable `FORCE_RETRAIN_DEFAULT_MODELS` permite forzar una nueva ejecución. Si los resultados validados ya existen, se reutilizan para evitar repetir innecesariamente el entrenamiento durante cada apertura.
                """
            ),
            code(
                """
import os
import shutil
import subprocess

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
    / "default_models"
)

REQUIRED_DEFAULT_RESULT_FILES = [
    DEFAULT_RESULTS_DIR
    / "validation_metrics.csv",
    DEFAULT_RESULTS_DIR
    / "training_history.csv",
    DEFAULT_RESULTS_DIR
    / "recommendation_examples.csv",
    DEFAULT_RESULTS_DIR
    / "recommendations.npz",
    DEFAULT_RESULTS_DIR
    / "run_summary.json",
]

FORCE_RETRAIN_DEFAULT_MODELS = False

results_are_complete = all(
    path.exists()
    for path in REQUIRED_DEFAULT_RESULT_FILES
)

if (
    FORCE_RETRAIN_DEFAULT_MODELS
    or not results_are_complete
):
    conda_candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(
            Path.home()
            / "miniconda3"
            / "Scripts"
            / "conda.exe"
        ),
        str(
            Path.home()
            / "anaconda3"
            / "Scripts"
            / "conda.exe"
        ),
    ]

    conda_executable = next(
        (
            candidate
            for candidate in conda_candidates
            if candidate
            and Path(candidate).exists()
        ),
        None,
    )

    if conda_executable is None:
        raise FileNotFoundError(
            "A Conda executable could not be located."
        )

    command = [
        conda_executable,
        "run",
        "-n",
        "apa-tp2-final",
        "--no-capture-output",
        "python",
        "scripts/train_default_models.py",
        "--overwrite",
    ]

    print(
        "Running:",
        " ".join(command),
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for output_line in process.stdout:
        print(
            output_line,
            end="",
            flush=True,
        )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "Default-model training failed with "
            f"exit code {return_code}."
        )
else:
    print(
        "Using existing validated default-model results."
    )
                """
            ),
            code(
                """
default_validation_results = pd.read_csv(
    DEFAULT_RESULTS_DIR
    / "validation_metrics.csv"
)

default_training_history = pd.read_csv(
    DEFAULT_RESULTS_DIR
    / "training_history.csv"
)

default_examples = pd.read_csv(
    DEFAULT_RESULTS_DIR
    / "recommendation_examples.csv"
)

with (
    DEFAULT_RESULTS_DIR
    / "run_summary.json"
).open(
    "r",
    encoding="utf-8",
) as file:
    default_run_summary = json.load(
        file
    )

display(
    default_validation_results
)
                """
            ),
            code(
                """
comparison_columns = [
    "model",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "hit_rate_at_10",
    "sampled_auc",
    "catalog_coverage_at_10",
    "intra_list_diversity_at_10",
]

display(
    default_validation_results[
        comparison_columns
    ]
    .sort_values(
        [
            "precision_at_10",
            "ndcg_at_10",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)
                """
            ),
            code(
                """
precision_plot = (
    default_validation_results[
        [
            "model",
            "precision_at_10",
        ]
    ]
    .sort_values(
        "precision_at_10"
    )
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    precision_plot["model"],
    precision_plot["precision_at_10"],
)

ax.set_title(
    "Validation Precision@10"
)
ax.set_xlabel(
    "Precision@10"
)
ax.set_ylabel(
    "Model"
)

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
fig, ax = plt.subplots(
    figsize=(10, 6)
)

for model_name, model_history in (
    default_training_history.groupby(
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
    "Training loss by epoch"
)
ax.set_xlabel(
    "Epoch"
)
ax.set_ylabel(
    "Training objective"
)
ax.legend(
    bbox_to_anchor=(
        1.02,
        1,
    ),
    loc="upper left",
)

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
neural_results = (
    default_validation_results[
        default_validation_results[
            "model_family"
        ].eq("pytorch")
    ]
    .copy()
)

architecture_comparison = (
    neural_results
    .pivot(
        index="objective",
        columns="uses_metadata",
        values="precision_at_10",
    )
    .rename(
        columns={
            False: "collaborative",
            True: "hybrid",
        }
    )
)

architecture_comparison[
    "hybrid_lift"
] = (
    architecture_comparison[
        "hybrid"
    ]
    - architecture_comparison[
        "collaborative"
    ]
)

display(
    architecture_comparison
)
                """
            ),
            code(
                """
selected_default_model = (
    default_run_summary[
        "selected_default_model"
    ]
)

print(
    "Selected default neural model:",
    selected_default_model,
)

print(
    "Test set accessed:",
    default_run_summary[
        "test_set_accessed"
    ],
)

display(
    default_examples[
        default_examples[
            "model"
        ].eq(
            selected_default_model
        )
    ][
        [
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
expected_default_models = {
    "popularity",
    "content",
    "collaborative_logistic",
    "collaborative_bpr",
    "collaborative_warp_style",
    "hybrid_logistic",
    "hybrid_bpr",
    "hybrid_warp_style",
}

assert set(
    default_validation_results[
        "model"
    ]
) == expected_default_models

assert (
    default_validation_results[
        [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "hit_rate_at_10",
            "sampled_auc",
            "catalog_coverage_at_10",
            "intra_list_diversity_at_10",
        ]
    ]
    .apply(
        lambda column: column.between(
            0,
            1,
        ).all()
    )
    .all()
)

assert (
    default_run_summary[
        "cohort_split_sha256"
    ]
    == prepared.summary[
        "cohort_split_sha256"
    ]
)

assert (
    default_run_summary[
        "test_set_accessed"
    ]
    is False
)

print(
    "Default-model result checks passed."
)
                """
            ),
            markdown(
                """
## 13. Selección del candidato por defecto

El candidato se selecciona únicamente entre los seis modelos neuronales utilizando Precision@10 de validación. NDCG@10 y Hit Rate@10 funcionan como criterios de desempate.

Esta selección todavía no representa el modelo final. El siguiente paso aplicará optimización bayesiana sobre la mejor familia de arquitectura y pérdida. El conjunto de prueba continúa completamente reservado.
                """
            ),
        ]
    )

    nbf.write(
        notebook,
        NOTEBOOK_PATH,
    )

    print(
        f"Phase 4 notebook created: "
        f"{NOTEBOOK_PATH}"
    )


if __name__ == "__main__":
    main()
