# Task 2: Content Type Prediction Model (Binary Classification)

## Overview
A machine learning classification pipeline developed to predict whether a given Netflix title is a **Movie** or a **TV Show** based on its available metadata features (excluding direct label leaks such as duration).

---

## Project Structure

```text
task2/
├── models/
│   ├── best_model.joblib             # Serialized top-performing pipeline
│   ├── confusion_matrix.png          # Confusion matrix visualization
│   └── model_comparison.csv         # Side-by-side metric comparison table
├── src/
│   ├── preprocessing.py              # Feature selection, one-hot encoding & genre vectorizer
│   ├── model_training.py             # 80/20 stratified train/test split & model training
│   └── evaluation.py                 # Accuracy, precision, recall, F1, report & matrix plot
├── main.py                           # End-to-end execution script
└── README.md                         # Documentation
```

---

## Workflow & Methodology

### Step 1: Feature Selection & Leak Prevention
- Selected features:
  - `director` (categorical)
  - `country` (categorical)
  - `rating` (categorical)
  - `listed_in` (multi-category genres)
  - `release_year` (numerical)
- **Target:** `type` (`Movie` vs `TV Show`).
- **Feature Leak Avoidance:** The `duration` column is intentionally excluded from the feature set, as duration values directly reveal the format (e.g., minutes indicate Movies, whereas season counts indicate TV Shows).

### Step 2: Categorical & Numerical Encoding
Implemented via `sklearn.compose.ColumnTransformer`:
- **Categorical Features (`director`, `country`, `rating`):** Imputed with `"unknown"` and transformed using `OneHotEncoder(handle_unknown="ignore")`.
- **Genre Labels (`listed_in`):** Multi-label tokenization using a binary `CountVectorizer` to properly represent individual genre tags rather than monolithic strings.
- **Numerical Features (`release_year`):** Median imputation via `SimpleImputer` followed by standard scaling (`StandardScaler`).

### Step 3: Model Training
Data is split 80% training (7,029 records) and 20% test (1,758 records) using stratified sampling to preserve the target class balance. Four classification algorithms are trained within scikit-learn `Pipeline` objects:
1. **Linear SVM (`LinearSVC`)**
2. **Logistic Regression (`LogisticRegression`)**
3. **Decision Tree (`DecisionTreeClassifier`)**
4. **Random Forest (`RandomForestClassifier`)**

### Step 4: Model Evaluation
Models are evaluated on unseen test data using:
- Accuracy
- Macro-averaged Precision
- Macro-averaged Recall
- Macro-averaged F1 Score
- Detailed Classification Report & Confusion Matrix

### Step 5: Model Comparison & Selection

| Model | Accuracy | Precision | Recall | Macro F1 Score |
|---|---|---|---|---|
| **Linear SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **0.9994** | 0.9996 | 0.9991 | 0.9993 |
| **Decision Tree** | **0.9989** | 0.9992 | 0.9981 | 0.9987 |
| **Random Forest** | **0.9915** | 0.9887 | 0.9912 | 0.9899 |

**Selected Best Model:** `Linear SVM` (achieving near-perfect classification based on genre signals and metadata).

---

## Artifacts Generated

1. **Serialized Model:** `models/best_model.joblib` (contains the end-to-end trained pipeline including preprocessors and classifier).
2. **Metrics Table:** `models/model_comparison.csv`.
3. **Confusion Matrix Plot:** `models/confusion_matrix.png`.

---

## Sample Inference Output (New Titles)

```text
Title 1 (Genre: TV Dramas, Crime TV Shows | Rating: TV-MA) -> Prediction: TV Show
Title 2 (Director: Christopher Nolan | Genre: Action & Adventure, Sci-Fi | Rating: PG-13) -> Prediction: Movie
```

---

## How to Run

Execute the complete training, evaluation, and inference pipeline:
```bash
python task2/main.py
```
