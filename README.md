# TP2 — Sistema híbrido de recomendación de anime

Trabajo Práctico 2 de **Análisis Predictivo Avanzado**. El proyecto construye y despliega un sistema de recomendación top-k para una plataforma hipotética de streaming de anime.

## Problema de negocio

El catálogo contiene miles de títulos y una política única de popularidad ofrece la misma experiencia a todos. El objetivo es ordenar los animes todavía no consumidos según la afinidad estimada de cada usuario, para facilitar el descubrimiento y apoyar métricas como reproducciones iniciadas, incorporaciones a la lista y retención.

La aplicación desplegada contempla dos situaciones:

- **Usuario existente:** recomendaciones personalizadas a partir de su historial.
- **Usuario nuevo:** fallback basado en géneros y tipo de sus títulos favoritos para resolver el cold start inicial.

## Datos

Fuente: Anime Recommendations Database de Kaggle.

- `data/anime.csv`: metadata de 12.294 animes.
- `data/rating.parquet`: 7.813.737 interacciones de usuarios.
- `rating = -1`: visto sin rating explícito.
- Señal positiva del modelo: rating explícito mayor o igual a 8.

Para conservar historias coherentes se muestrean usuarios y se mantienen todas sus interacciones positivas. La cohorte final contiene:

- 1.499 usuarios activos.
- 2.451 animes.
- 110.204 interacciones positivas.

## Metodología

El notebook compara:

1. Baseline de popularidad.
2. Baseline basado en contenido.
3. Factorización colaborativa con pérdida logística, BPR y WARP-style.
4. Factorización híbrida con embeddings de usuario, anime, género y tipo.
5. Modelo híbrido WARP-style optimizado con Optuna mediante TPE.

La partición se hace dentro de cada usuario en entrenamiento, validación y test. El test permanece reservado hasta seleccionar arquitectura e hiperparámetros.

La implementación reproduce la idea central de LightFM —sumar representaciones latentes de identidad y metadata— mediante PyTorch. Esta decisión evita los problemas de compilación de LightFM en versiones nuevas de Python y mantiene explícita la lógica del modelo.

## Resultado final

Resultados sobre el test reservado:

| Modelo | Precision@10 | Recall@10 | AUC | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|---:|
| Popularidad | 0,059 | 0,093 | 0,814 | 0,018 | 0,818 |
| Contenido | 0,014 | 0,024 | 0,689 | 0,299 | 0,397 |
| Híbrido WARP-style sin tuning | 0,128 | 0,201 | 0,905 | 0,345 | 0,785 |
| **Híbrido WARP-style optimizado** | **0,133** | **0,207** | **0,901** | **0,220** | **0,781** |

El modelo final mejora Precision@10 aproximadamente **126 %** frente al baseline de popularidad.

## Estructura principal

```text
.
├── app.py
├── requirements.txt
├── requirements-notebook.txt
├── data/
│   ├── anime.csv
│   └── rating.parquet
├── artifacts/
│   ├── hybrid_model.npz
│   ├── item_features.npy
│   ├── seen_interactions.npz
│   ├── anime_deploy.parquet
│   ├── demo_users.json
│   ├── model_results.csv
│   └── optuna_trials.csv
└── notebooks/
    └── TP2_sistema_recomendacion_anime.ipynb
```

## Ejecutar el notebook

Se recomienda Python 3.11 o 3.12.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements-notebook.txt
jupyter notebook notebooks/TP2_sistema_recomendacion_anime.ipynb
```

El notebook detecta si se ejecuta desde la raíz o desde `notebooks/` y guarda los artefactos en `artifacts/`.

## Ejecutar la aplicación

```bash
pip install -r requirements.txt
streamlit run app.py
```

El deployment no entrena el modelo. Carga representaciones y metadata ya exportadas, por lo que su ejecución es liviana.

## Deployment

Demo en Streamlit Community Cloud:

`https://fedesaroka-analisis-predictivo-avanzado-app-a76w3e.streamlit.app/`

El deployment público se actualizará al hacer push de esta versión a la rama configurada en Streamlit.

## Limitaciones

El dataset no contiene timestamps, por lo que la separación es aleatoria dentro de cada usuario y no temporal. Los resultados son offline y deben validarse con un experimento A/B antes de atribuir impacto comercial. La definición de positivo también debería revisarse con datos reales de consumo y objetivos del producto.
