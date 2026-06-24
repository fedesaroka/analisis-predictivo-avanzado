# Sistema híbrido de recomendación de anime

Trabajo Práctico 2 de **Análisis Predictivo Avanzado**.

**Demo:** https://fedesaroka-analisis-predictivo-avanzado-app-xmct2f.streamlit.app/

## Objetivo

El proyecto construye un sistema de ranking top-k que recomienda anime no observados a partir del historial positivo de cada usuario.

Se implementaron y compararon:

- Baseline de popularidad
- Baseline basado en contenido
- Factores latentes colaborativos
- Factores latentes híbridos con género y tipo
- Objetivos logistic, BPR y WARP-style hard negative
- Optimización bayesiana con Optuna

El modelo final combina embeddings colaborativos con una proyección aprendida de los metadatos de género y tipo.

## Resultado final

El modelo fue seleccionado únicamente con validación. Después de congelar arquitectura e hiperparámetros, se combinaron train y validación y se utilizó test una única vez.

| Métrica de test | Resultado |
|---|---:|
| Precision@10 | 0.1153 |
| Recall@10 | 0.1961 |
| NDCG@10 | 0.1904 |
| Hit Rate@10 | 0.6662 |
| Sampled AUC | 0.9245 |
| Cobertura@10 | 0.3663 |
| Diversidad interna@10 | 0.5936 |

## Aplicación

La aplicación Streamlit ofrece dos modos.

### Usuario existente

Utiliza el embedding aprendido para uno de los 1.498 usuarios retenidos. Los títulos observados durante entrenamiento se excluyen automáticamente.

### Usuario nuevo

Construye un perfil de contenido a partir de entre 3 y 10 anime favoritos. Las recomendaciones se calculan mediante similitud de género y tipo.

La aplicación no reentrena el modelo y no carga PyTorch. La inferencia utiliza componentes exportados en NumPy y SciPy.

## Datos

Fuente original:

- Kaggle — Anime Recommendations Database

Archivos incluidos:

- `data/anime.csv`
- `data/rating.parquet`

La cohorte final contiene:

- 1.498 usuarios
- 2.438 anime
- 110.025 interacciones positivas
- 89.313 interacciones de entrenamiento
- 10.356 interacciones de validación
- 10.356 interacciones de prueba

Una interacción se considera positiva cuando el rating explícito es mayor o igual a 8.

## Metodología

1. Limpieza y deduplicación determinística.
2. Selección reproducible de usuarios.
3. Filtrado iterativo de usuarios e ítems.
4. Split por usuario en train, validación y test.
5. Construcción de features binarias de género y tipo.
6. Comparación de baselines y seis modelos neuronales.
7. Selección por Precision@10 de validación.
8. Optimización TPE con 15 trials de Optuna.
9. Entrenamiento final con train más validación.
10. Evaluación única sobre test.
11. Exportación de artefactos NumPy para Streamlit.

## Modelo final

- Arquitectura: factores latentes híbridos
- Objetivo: WARP-style hard negative
- Dimensión latente: 64
- Épocas: 10
- Batch size: 4096
- Learning rate: 0.0080503845
- Weight decay: 2.3423849847e-06
- Peso de metadatos: 0.25
- Máximo de negativos inspeccionados: 5
- Margen: 2.0

La pérdida se denomina WARP-style porque es una implementación propia inspirada en WARP y no una reproducción exacta de LightFM.

## Estructura principal

```text
.
├── app.py
├── artifacts/
├── config/
│   └── experiment.json
├── data/
│   ├── anime.csv
│   └── rating.parquet
├── notebooks/
│   └── TP2_sistema_recomendacion_anime.ipynb
├── results/
│   ├── default_models/
│   ├── optuna/
│   └── final_model/
├── scripts/
├── src/
│   └── anime_recommender/
├── requirements.txt
├── requirements-notebook.txt
└── README.md
```

## Ejecutar la aplicación

Se recomienda Python 3.11.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar el notebook

```bash
pip install -r requirements-notebook.txt
pip install -e .
python scripts/execute_notebook_with_progress.py
```

El notebook utiliza resultados ya validados para las fases costosas. PyTorch se ejecuta mediante scripts externos debido a una limitación de inicialización de DLL dentro del kernel Jupyter en Windows.

## Validaciones

```bash
python scripts/smoke_test_pytorch.py
python scripts/validate_data_pipeline.py
python scripts/validate_phase3.py
python scripts/validate_phase4.py
python scripts/validate_phase5.py
python scripts/validate_final_artifacts.py
python scripts/validate_streamlit_app.py
```

## Artefactos de despliegue

La carpeta `artifacts/` contiene:

* Embeddings de usuarios
* Embeddings efectivos de ítems
* Sesgos de usuario e ítem
* Matriz de interacciones observadas
* Catálogo procesado
* Matriz de metadatos
* Mappings de usuarios e ítems
* Usuarios de demostración
* Manifest con hashes SHA-256

`artifact_manifest.json` permite validar procedencia, dimensiones, métricas, hashes e integridad del despliegue.

## Limitaciones

* El dataset no ofrece timestamps confiables.
* Los negativos muestreados pueden contener preferencias aún no observadas.
* El cold-start utiliza solamente género y tipo.
* Las métricas son offline.
* El prototipo no mide clics, reproducción, retención ni satisfacción real.

## Reproducibilidad

* Semilla global: 42
* Python de entrenamiento: 3.11.15
* PyTorch: 2.12.1 CPU
* Optuna: 3.6.1
* Entrenamiento y evaluación determinísticos en CPU
* Cohorte y splits identificados mediante SHA-256
