# Task 4: Netflix Content Segmentation (Clustering / Unsupervised Learning)

## Overview
An unsupervised machine learning pipeline designed to discover organic groupings and content segments across the Netflix catalog using **K-Means Clustering** and **Silhouette Analysis** based on metadata attributes (content type, country, rating, genre tags, duration, and release era).

---

## Project Structure

```text
task4/
├── models/
│   └── netflix_kmeans_model.joblib   # Serialized preprocessor, K-Means model & metadata
├── outputs/
│   ├── silhouette_scores.csv         # Silhouette score evaluation across K=2 to 8
│   ├── clustered_netflix_titles.csv  # Full dataset with assigned cluster IDs
│   ├── cluster_summary.csv           # Detailed aggregate profile for each segment
│   ├── cluster_examples.csv          # Sample titles representative of each cluster
│   ├── cluster_visualization.png     # 2D SVD projection scatter plot of clusters
│   ├── cluster_sizes.png             # Bar chart showing title counts per cluster
│   └── cluster_content_types.png     # Stacked bar chart of Movie vs TV Show mix
├── src/
│   ├── data_loader.py                # Dataset loading and schema validation
│   ├── preprocessing.py              # Duration parsing, imputation, encoding & scaling
│   ├── clustering.py                 # Silhouette score optimization & K-Means fitting
│   ├── analysis.py                   # Cluster summarization & example extraction
│   └── visualization.py              # SVD projection & cluster distribution charts
├── main.py                           # End-to-end execution pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## Methodology & Workflow

### 1. Data Cleaning & Feature Preparation
- **Deduplication:** Dropped exact duplicate records while preserving indices.
- **Attributes Used:** `type`, `country`, `rating`, `listed_in`, `release_year`, `duration_value` (numeric amount), and `duration_unit` (`min` vs `Seasons`).
- **Excluded Attributes:** `title` and `show_id` (identifiers without generalizable cluster signals).
- **ColumnTransformer Preprocessing:**
  - Categoricals (`type`, `country`, `rating`, `listed_in`, `duration_unit`): Imputed with `"Unknown"` and encoded via `OneHotEncoder(handle_unknown="ignore", min_frequency=2)`.
  - Numericals (`release_year`, `duration_value`): Imputed with median and normalized via `StandardScaler`.

### 2. Optimal K Selection (Silhouette Analysis)
Evaluated candidate cluster counts from $K = 2$ to $K = 8$ using the **Silhouette Coefficient**:
$$\text{Silhouette Score} = \frac{b - a}{\max(a, b)}$$

| Number of Clusters ($K$) | Silhouette Score |
|---|---|
| $K = 2$ | 0.2984 |
| **$K = 3$** | **0.3015 (Optimal)** |
| $K = 4$ | 0.1570 |
| $K = 5$ | 0.1707 |
| $K = 6$ | 0.1303 |
| $K = 7$ | 0.1194 |
| $K = 8$ | 0.1199 |

**Selected Number of Clusters:** **$K = 3$** (achieves maximum cluster cohesion and separation).

---

## Content Segments & Cluster Interpretation

### Cluster 0: Modern Feature Films (5,519 titles)
- **Format:** 100.0% Movies
- **Average Release Year:** 2015.7
- **Top Genres:** International Movies, Dramas, Comedies, Documentaries, Independent Movies
- **Top Producing Countries:** United States, India, United Kingdom
- **Common Ratings:** TV-MA, TV-14, R
- **Representative Titles:** *Dick Johnson Is Dead*, *Confessions of an Invisible Girl*, *The Starling*, *Je Suis Karl*

### Cluster 1: Binge-Worthy TV & Episodic Content (2,641 titles)
- **Format:** 99.96% TV Shows
- **Average Release Year:** 2017.0
- **Top Genres:** International TV Shows, TV Dramas, TV Comedies, Crime TV Shows, Kids' TV
- **Top Producing Countries:** United States, Pakistan, United Kingdom
- **Common Ratings:** TV-MA, TV-14, TV-PG
- **Representative Titles:** *Ganglands*, *Midnight Mass*, *The Great British Baking Show*, *Jailbirds New Orleans*

### Cluster 2: Vintage Cinema & Classic Vault (627 titles)
- **Format:** 96.33% Movies / 3.67% TV Shows
- **Average Release Year:** **1988.7** (spanning 1940s to early 2000s)
- **Top Genres:** Dramas, Comedies, International Movies, Action & Adventure, Classic Movies
- **Top Producing Countries:** United States, India, United Kingdom
- **Common Ratings:** R, TV-14, PG-13
- **Representative Titles:** *Sankofa*, *Jeans*, *Minsara Kanavu*, *Avvai Shanmughi*, *Jaws*

---

## Artifacts Generated

1. **Serialized Model:** `models/netflix_kmeans_model.joblib` (preprocessor + fitted KMeans model).
2. **Cluster Assignment Data:** `outputs/clustered_netflix_titles.csv`.
3. **Cluster Summary Report:** `outputs/cluster_summary.csv`.
4. **Visualizations:**
   - `outputs/cluster_visualization.png`: 2D Truncated SVD projection showing cluster boundary separations.
   - `outputs/cluster_sizes.png`: Distribution of title volume across clusters.
   - `outputs/cluster_content_types.png`: Format breakdown (Movie vs TV Show) per cluster.

---

## How to Run

Execute the clustering pipeline:
```bash
python task4/main.py
```
*(Runs silhouette analysis, cluster assignment, visual chart exports, and model serialization.)*
