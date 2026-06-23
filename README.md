# TP2 - Análisis Predictivo Avanzado
## Sistema de Recomendación de Anime

### Descripción del proyecto

Trabajo Práctico 2 de la materia **Análisis Predictivo Avanzado**. El objetivo es construir un **sistema de recomendación** usando el dataset de Anime Recommendations Database de Kaggle.

El problema de negocio es: dado el historial de ratings de un usuario, recomendar animes que probablemente le gusten.

**Fecha de presentación:** miércoles 24/6 a las 19hs.

---

### Dataset

Fuente: [Kaggle - Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

#### `data/anime.csv`
Información de cada anime. Columnas:
- `anime_id`: ID único del anime
- `name`: nombre del anime
- `genre`: géneros separados por coma (ej: "Action, Comedy, Drama")
- `type`: formato (TV, Movie, OVA, etc.)
- `episodes`: cantidad de episodios
- `rating`: rating promedio del anime (escala 1-10)
- `members`: cantidad de usuarios que tienen el anime en su lista

#### `data/rating.parquet`
Ratings individuales de usuarios. Originalmente `rating.csv` (106 MB), convertido a Parquet (17.5 MB) para poder subir al repo. Columnas:
- `user_id`: ID del usuario
- `anime_id`: ID del anime
- `rating`: rating dado por el usuario (-1 si lo vio pero no lo rateó, 1-10 si lo rateó)

> **Nota:** el archivo original `rating.csv` está excluido del repo por superar el límite de GitHub (100 MB). Para leerlo usar `pd.read_parquet('data/rating.parquet')`.

---

### Estructura del repo

```
TP2 APA/
├── data/
│   ├── anime.csv           # metadata de animes
│   └── rating.parquet      # ratings de usuarios (7.8M filas)
├── C09 - SR1/              # Clase 9: Sistemas de Recomendación (parte 1)
│   └── C09.ipynb
├── C10 - SR2/              # Clase 10: Sistemas de Recomendación (parte 2)
│   ├── C10.ipynb
│   ├── C10_solución.ipynb
│   └── Data/               # datasets usados en clase (movies y books)
├── notebooks/              # notebooks del TP
│   └── 01_analisis_descriptivo.ipynb
├── Consigna TP2.docx       # consigna oficial
└── README.md
```

---

### Consigna (resumen)

Entregar:
1. **PPT** con:
   - Introducción y problema de negocio
   - Desarrollo: técnicas de ML y técnicas específicas de la materia (sistema de recomendación)
   - **Deploy obligatorio** del modelo en alguna nube
   - Conclusiones y acciones de negocio hipotéticas
2. **Notebook** funcional, bien comentada, que corra de principio a fin y esté alineada a la PPT

---

### Cómo correr el proyecto

```bash
# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn jupyter pyarrow

# Abrir Jupyter
jupyter notebook
```

Los notebooks están en la carpeta `notebooks/`, correrlos en orden numérico.

---

### Contexto para Claude

- El dataset tiene **7.8 millones de ratings** de usuarios sobre animes, más metadata de los animes (género, tipo, rating promedio, miembros).
- El enfoque es **sistema de recomendación** (Collaborative Filtering y/o Content-Based).
- Los ratings `-1` en `rating.parquet` indican que el usuario vio el anime pero no lo rateó explícitamente.
- Las clases de referencia están en `C09 - SR1/` y `C10 - SR2/` con ejemplos de sistemas de recomendación sobre datasets de películas y libros.
- La presentación es el **24/6** así que los tiempos son ajustados.
