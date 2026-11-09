# Task 1: Netflix Content Recommendation Engine

## Overview
A content-based recommendation system built on Netflix movies and TV shows dataset. It analyzes title attributes (genre, director, country, cast, rating, and duration) to calculate content similarity using **TF-IDF Vectorization** and **Cosine Similarity**, recommending the top-N most relevant titles for any selected movie or TV show.

---

## Project Structure

```text
task1/
├── data/
│   └── raw/
│       └── Dataset.csv               # Netflix titles dataset
├── src/
│   ├── data_loader.py                # Dataset loading utilities
│   ├── data_quality.py               # Missing values & duplicate detection/handling
│   ├── eda.py                        # Exploratory data analysis & visualizations
│   ├── preprocessing.py              # Text cleaning, normalization & column pruning
│   ├── feature_engineering.py        # Standard & feature-weighted TF-IDF vectorization
│   ├── recommendation.py             # Cosine similarity recommendation engine
│   └── evaluation.py                 # Precision@K, genre overlap & model comparison
├── evaluation.py                     # Standalone evaluation benchmark script
├── main.py                           # Full end-to-end pipeline execution
└── README.MD                         # Documentation
```

---

## Recommendation Methodology

### 1. Data Cleaning & Preprocessing
- Removal of duplicate content records across key metadata fields with index realignment (`reset_index(drop=True)`).
- Normalization of text columns (stripping whitespace, lowercasing, handling missing values).
- Dropping redundant/identifier columns (`show_id`, `date_added`).

### 2. Feature Engineering
Two vectorization approaches are implemented and compared:
- **Standard TF-IDF:** Concatenates all textual features into a single combined string per title and extracts TF-IDF n-grams (up to 10,000 features).
- **Weighted Multi-Feature TF-IDF:** Applies dedicated TF-IDF vectorizers to each feature column independently with customized weights reflecting their importance in recommendation relevance:
  - `listed_in` (Genres): **50%**
  - `director`: **20%**
  - `country`: **10%**
  - `type` (Movie vs TV Show): **10%**
  - `rating`: **5%**
  - `duration`: **5%**

### 3. Similarity Calculation
Cosine similarity is calculated between the query item's feature vector and all catalog vectors:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

Titles are ranked in descending order, excluding the query title itself, and the top-N results are returned.

---

## Evaluation Metrics

To quantitatively evaluate content-based recommendations without user interaction logs, the system measures:
1. **Genre Match Rate (Precision@K):** Percentage of top-K recommendations sharing at least one genre with the query title.
2. **Content Type Consistency:** Percentage of recommendations that match the format (Movie vs TV Show) of the input.
3. **Average Genre Overlap (Jaccard Index):** Intersection over Union (IoU) of genres between the query and recommendations:
   $$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
4. **Mean Similarity Score:** Average cosine similarity across top-K recommendations.

### Benchmark Results (Top-10 Recommendations across 10 diverse titles)

| Model | Titles Evaluated | Top-K | Avg Genre Match Rate (%) | Content Type Match Rate (%) | Avg Genre Overlap (%) | Avg Similarity Score |
|---|---|---|---|---|---|---|
| **Standard TF-IDF** | 10 | 10 | 93.0% | 100.0% | 70.45% | 0.8269 |
| **Weighted TF-IDF** | 10 | 10 | **100.0%** | **100.0%** | **93.50%** | **0.9328** |

*Weighted TF-IDF significantly outperforms unweighted TF-IDF in genre preservation (+23% overlap).*

---

## Sample Recommendations

### Query: *"Breaking Bad"* (Crime TV Shows, TV Dramas, TV Thrillers)
1. **Ozark** (Crime TV Shows, TV Dramas, TV Mysteries) – Similarity: 0.994
2. **Designated Survivor** (Crime TV Shows, TV Dramas, TV Mysteries) – Similarity: 0.994
3. **The Assassination of Gianni Versace** (Crime TV Shows, TV Dramas) – Similarity: 0.992
4. **Dare Me** (Crime TV Shows, TV Dramas, TV Mysteries) – Similarity: 0.992
5. **The Blacklist** (Crime TV Shows, TV Action & Adventure, TV Dramas) – Similarity: 0.986
6. **The Lizzie Borden Chronicles** (Crime TV Shows, TV Dramas) – Similarity: 0.984
7. **House of Cards** (TV Dramas, TV Mysteries, TV Thrillers) – Similarity: 0.891
8. **Messiah** (TV Dramas, TV Mysteries, TV Thrillers) – Similarity: 0.889
9. **What / If** (TV Dramas, TV Mysteries, TV Thrillers) – Similarity: 0.889
10. **Containment** (TV Dramas, TV Mysteries, TV Thrillers) – Similarity: 0.889

---

## How to Run

### Run Full Pipeline
```bash
python task1/main.py
```

### Run Standalone Evaluation Benchmark
```bash
python task1/evaluation.py
```
