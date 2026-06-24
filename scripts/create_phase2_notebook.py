"""Create the clean TP2 notebook through the data-pipeline phase."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


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
    notebook = nbf.v4.new_notebook()

    notebook.metadata["kernelspec"] = {
        "display_name": (
            "Python 3.11 - APA TP2 Final"
        ),
        "language": "python",
        "name": "apa-tp2-final",
    }

    notebook.metadata["language_info"] = {
        "name": "python",
        "version": "3.11",
    }

    notebook.cells = [
        markdown(
            """
# Sistema híbrido de recomendación de anime

**Trabajo Práctico 2 — Análisis Predictivo Avanzado**

Este notebook construye, optimiza y evalúa un sistema de recomendación top-k para una plataforma hipotética de streaming de anime.

El objetivo operativo consiste en seleccionar los diez títulos no vistos que deberían mostrarse primero a cada usuario. La implementación compara baselines de popularidad y contenido con modelos colaborativos e híbridos entrenados mediante objetivos logísticos, BPR y muestreo de negativos difíciles inspirado en WARP.
            """
        ),
        markdown(
            """
## 1. Diseño del experimento

La unidad de análisis es una interacción positiva entre un usuario y un anime. Se considera positiva una calificación explícita de al menos 8 puntos.

El conjunto de datos no contiene marcas temporales confiables. Por este motivo, la separación se realiza aleatoriamente dentro de cada usuario, utilizando semillas determinísticas. Cada usuario conserva observaciones en entrenamiento, validación y prueba.

La validación se utilizará para comparar arquitecturas, pérdidas e hiperparámetros. El conjunto de prueba permanecerá reservado hasta que todas las decisiones de modelado hayan sido fijadas.
            """
        ),
        code(
            """
%matplotlib inline

from pathlib import Path
import json
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from anime_recommender import (
    load_experiment_config,
    prepare_data,
)

ROOT = Path.cwd().resolve()

if ROOT.name == "notebooks":
    ROOT = ROOT.parent

CONFIG_PATH = ROOT / "config" / "experiment.json"
ANIME_PATH = ROOT / "data" / "anime.csv"
RATINGS_PATH = ROOT / "data" / "rating.parquet"

config = load_experiment_config(CONFIG_PATH)

print("Project root:", ROOT)
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("Modeling framework:", config["framework"])
print("Device:", config["device"])
            """
        ),
        code(
            """
display(
    pd.DataFrame(
        {
            "Parameter": list(config.keys()),
            "Value": [
                json.dumps(value)
                if isinstance(value, (list, dict))
                else value
                for value in config.values()
            ],
        }
    )
)
            """
        ),
        markdown(
            """
## 2. Carga, validación y definición de interacciones positivas

El catálogo y las calificaciones se validan antes de cualquier modelado. Los identificadores se convierten a enteros, los títulos se normalizan y las referencias a anime inexistentes se eliminan.

Las calificaciones iguales a `-1` representan consumo sin una evaluación explícita. Se excluyen del conjunto positivo principal. Una interacción positiva requiere una calificación mayor o igual a 8.
            """
        ),
        code(
            """
prepared = prepare_data(
    anime_path=ANIME_PATH,
    ratings_path=RATINGS_PATH,
    config=config,
)

summary_table = pd.DataFrame(
    {
        "Measure": list(prepared.summary.keys()),
        "Value": list(prepared.summary.values()),
    }
)

display(summary_table)
            """
        ),
        markdown(
            """
## 3. Cohorte determinística

Los usuarios candidatos deben tener suficiente historial positivo para sostener una evaluación individual. Los identificadores elegibles se ordenan antes del muestreo y se utiliza una semilla fija.

Luego se aplican iterativamente dos restricciones:

- Cada usuario debe conservar como mínimo la cantidad configurada de interacciones positivas.
- Cada anime debe haber sido valorado positivamente por al menos la cantidad configurada de usuarios.

El proceso se repite hasta que ambas condiciones se cumplen simultáneamente.
            """
        ),
        code(
            """
user_positive_counts = (
    prepared.interactions
    .groupby("user_id")
    .size()
)

item_positive_counts = (
    prepared.interactions
    .groupby("anime_id")
    .size()
)

count_diagnostics = pd.DataFrame(
    {
        "Statistic": [
            "Users",
            "Items",
            "Positive interactions",
            "Minimum positives per user",
            "Median positives per user",
            "Maximum positives per user",
            "Minimum users per item",
            "Median users per item",
            "Maximum users per item",
        ],
        "Value": [
            prepared.interactions["user_id"].nunique(),
            prepared.interactions["anime_id"].nunique(),
            len(prepared.interactions),
            user_positive_counts.min(),
            user_positive_counts.median(),
            user_positive_counts.max(),
            item_positive_counts.min(),
            item_positive_counts.median(),
            item_positive_counts.max(),
        ],
    }
)

display(count_diagnostics)
            """
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.hist(
    user_positive_counts,
    bins=30,
)

ax.set_title(
    "Positive interactions per retained user"
)
ax.set_xlabel(
    "Number of positive interactions"
)
ax.set_ylabel(
    "Users"
)

plt.tight_layout()
plt.show()
            """
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.hist(
    item_positive_counts,
    bins=40,
)

ax.set_title(
    "Positive users per retained anime"
)
ax.set_xlabel(
    "Number of positive users"
)
ax.set_ylabel(
    "Anime titles"
)

plt.tight_layout()
plt.show()
            """
        ),
        markdown(
            """
## 4. Train, validation and test split

Every user has at least one validation interaction and one test interaction. The remainder is assigned to training.

A deterministic repair step guarantees that every retained anime appears at least once in training. When necessary, it exchanges a held-out interaction with a training interaction from the same user without leaving the replacement anime unsupported.

This avoids evaluating collaborative models on item identifiers that were never observed during training.
            """
        ),
        code(
            """
split_counts = (
    prepared.split_interactions
    .groupby("split")
    .agg(
        interactions=("anime_id", "size"),
        users=("user_id", "nunique"),
        items=("anime_id", "nunique"),
    )
    .reindex(
        [
            "train",
            "validation",
            "test",
        ]
    )
)

display(split_counts)
            """
        ),
        code(
            """
per_user_split_counts = (
    prepared.split_interactions
    .groupby(
        [
            "user_id",
            "split",
        ]
    )
    .size()
    .unstack(fill_value=0)
)

display(
    per_user_split_counts.describe().T
)
            """
        ),
        code(
            """
assert (
    prepared.train_matrix.multiply(
        prepared.validation_matrix
    ).nnz
    == 0
)

assert (
    prepared.train_matrix.multiply(
        prepared.test_matrix
    ).nnz
    == 0
)

assert (
    prepared.validation_matrix.multiply(
        prepared.test_matrix
    ).nnz
    == 0
)

assert prepared.summary[
    "items_without_training_support"
] == 0

assert (
    prepared.train_matrix.nnz
    + prepared.validation_matrix.nnz
    + prepared.test_matrix.nnz
    == prepared.all_positive_matrix.nnz
)

print(
    "Split integrity checks passed."
)
print(
    "Cohort hash:",
    prepared.summary[
        "cohort_split_sha256"
    ],
)
            """
        ),
        markdown(
            """
## 5. Phase 2 conclusion

The final cohort, mappings, and split assignments are deterministic and free of overlap. Every retained user is represented in all three subsets, and every retained item has training support.

The following sections will construct metadata features, evaluation functions, baselines, and the collaborative and hybrid recommendation models.
            """
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nbf.write(
        notebook,
        NOTEBOOK_PATH,
    )

    print(
        f"Notebook created: {NOTEBOOK_PATH}"
    )


if __name__ == "__main__":
    main()
