"""Create the notebook through Bayesian optimization."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from create_phase4_notebook import (
    main as create_phase4_notebook,
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
    create_phase4_notebook()

    notebook = nbf.read(
        NOTEBOOK_PATH,
        as_version=4,
    )

    notebook.cells.extend(
        [
            markdown(
                """
## 14. Optimización bayesiana

La comparación por defecto seleccionó el modelo híbrido con pérdida WARP-style. La optimización mantiene fija esta arquitectura y explora sus hiperparámetros mediante Optuna y un muestreador TPE con semilla fija.

La función objetivo es Precision@10 sobre validación. NDCG@10 y Hit Rate@10 se utilizan únicamente como desempates determinísticos.

El trial 0 reproduce exactamente la configuración por defecto. Por lo tanto, el estudio contiene siempre el benchmark existente y la configuración seleccionada no puede ser inferior a este por la métrica principal.
                """
            ),
            markdown(
                """
### Espacio de búsqueda

Se optimizan la dimensión latente, cantidad de épocas, tamaño de lote, tasa de aprendizaje, regularización, peso de metadatos, cantidad máxima de negativos inspeccionados y margen de violación.

El conjunto de prueba permanece reservado. Todos los trials utilizan la misma cohorte, el mismo split y la misma semilla de entrenamiento.
                """
            ),
            code(
                """
OPTUNA_RESULTS_DIR = (
    ROOT
    / "results"
    / "optuna"
)

REQUIRED_OPTUNA_RESULT_FILES = [
    OPTUNA_RESULTS_DIR
    / "best_params.json",
    OPTUNA_RESULTS_DIR
    / "best_recommendations.npz",
    OPTUNA_RESULTS_DIR
    / "best_training_history.csv",
    OPTUNA_RESULTS_DIR
    / "best_validation_metrics.csv",
    OPTUNA_RESULTS_DIR
    / "parameter_importances.csv",
    OPTUNA_RESULTS_DIR
    / "recommendation_examples.csv",
    OPTUNA_RESULTS_DIR
    / "run_summary.json",
    OPTUNA_RESULTS_DIR
    / "trials.csv",
    OPTUNA_RESULTS_DIR
    / "validation_comparison.csv",
]

FORCE_RETUNE_SELECTED_MODEL = False

optuna_results_are_complete = all(
    path.exists()
    for path in REQUIRED_OPTUNA_RESULT_FILES
)

if (
    FORCE_RETUNE_SELECTED_MODEL
    or not optuna_results_are_complete
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
        "scripts/tune_selected_model.py",
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
            "Optuna tuning failed with "
            f"exit code {return_code}."
        )
else:
    print(
        "Using existing validated Optuna results."
    )
                """
            ),
            code(
                """
optuna_trials = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "trials.csv"
)

best_validation_metrics = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "best_validation_metrics.csv"
)

validation_comparison = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "validation_comparison.csv"
)

parameter_importances = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "parameter_importances.csv"
)

best_training_history = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "best_training_history.csv"
)

optuna_examples = pd.read_csv(
    OPTUNA_RESULTS_DIR
    / "recommendation_examples.csv"
)

with (
    OPTUNA_RESULTS_DIR
    / "best_params.json"
).open(
    "r",
    encoding="utf-8",
) as file:
    best_optuna_params = json.load(file)

with (
    OPTUNA_RESULTS_DIR
    / "run_summary.json"
).open(
    "r",
    encoding="utf-8",
) as file:
    optuna_run_summary = json.load(file)

display(
    validation_comparison
)

display(
    pd.DataFrame(
        {
            "parameter": list(
                best_optuna_params.keys()
            ),
            "value": list(
                best_optuna_params.values()
            ),
        }
    )
)
                """
            ),
            code(
                """
fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    optuna_trials["trial_number"],
    optuna_trials["precision_at_10"],
    marker="o",
)

ax.axhline(
    validation_comparison.loc[
        validation_comparison[
            "variant"
        ].eq("default"),
        "precision_at_10",
    ].iloc[0],
    linestyle="--",
    label="Default model",
)

ax.set_title(
    "Optuna validation Precision@10 by trial"
)
ax.set_xlabel(
    "Trial"
)
ax.set_ylabel(
    "Precision@10"
)
ax.legend()

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
importance_plot = (
    parameter_importances
    .sort_values("importance")
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    importance_plot["parameter"],
    importance_plot["importance"],
)

ax.set_title(
    "Optuna parameter importance"
)
ax.set_xlabel(
    "Relative importance"
)
ax.set_ylabel(
    "Hyperparameter"
)

plt.tight_layout()
plt.show()
                """
            ),
            code(
                """
comparison_plot = (
    validation_comparison
    .set_index("variant")[
        [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "hit_rate_at_10",
        ]
    ]
    .T
)

ax = comparison_plot.plot(
    kind="bar",
    figsize=(10, 5),
)

ax.set_title(
    "Default versus tuned validation performance"
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
print(
    "Selected Optuna trial:",
    optuna_run_summary[
        "selected_trial_number"
    ],
)

print(
    "Precision lift versus default:",
    optuna_run_summary[
        "precision_lift_vs_default"
    ],
)

print(
    "Test set accessed:",
    optuna_run_summary[
        "test_set_accessed"
    ],
)

display(
    optuna_examples[
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
assert len(
    optuna_trials
) == config[
    "optuna_trials"
]

assert (
    optuna_trials[
        "state"
    ].eq("COMPLETE")
    .all()
)

assert (
    optuna_run_summary[
        "cohort_split_sha256"
    ]
    == prepared.summary[
        "cohort_split_sha256"
    ]
)

assert (
    optuna_run_summary[
        "test_set_accessed"
    ]
    is False
)

assert (
    validation_comparison.loc[
        validation_comparison[
            "variant"
        ].eq("tuned"),
        "precision_at_10",
    ].iloc[0]
    >=
    validation_comparison.loc[
        validation_comparison[
            "variant"
        ].eq("default"),
        "precision_at_10",
    ].iloc[0]
)

print(
    "Optuna result checks passed."
)
                """
            ),
            markdown(
                """
## 15. Selección del modelo optimizado

El modelo ajustado se selecciona por Precision@10 de validación, con NDCG@10 y Hit Rate@10 como desempates. El trial seleccionado se vuelve a entrenar desde cero para confirmar que sus resultados son reproducibles.

A partir de este punto se congelan la arquitectura, la pérdida y los hiperparámetros. La siguiente fase combinará entrenamiento y validación, entrenará una única vez el modelo final y utilizará el conjunto de prueba por primera y única vez.
                """
            ),
        ]
    )

    nbf.write(
        notebook,
        NOTEBOOK_PATH,
    )

    print(
        f"Phase 5 notebook created: "
        f"{NOTEBOOK_PATH}"
    )


if __name__ == "__main__":
    main()
