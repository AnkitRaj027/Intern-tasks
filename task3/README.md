# Task 3: Netflix Audience Rating Classification (Multi-Class Classification)

## Overview
A machine learning multi-class classification pipeline designed to predict the **audience maturity rating category** (e.g., `TV-MA`, `TV-14`, `TV-PG`, `R`, `PG-13`, `TV-Y7`, `TV-Y`, `PG`, `TV-G`, `NR`, `G`, `Other`) of Netflix titles based on their content attributes.

---

## Project Structure

```text
task3/
├── models/
│   ├── best_rating_model.joblib      # Serialized top-performing pipeline
│   └── model_metadata.joblib         # Feature column names & preprocessing metadata
├── outputs/
│   ├── rating_distribution.png       # Audience rating frequency visualization
│   ├── confusion_matrix.png          # Normalized confusion matrix heatmap
│   └── model_comparison.csv         # Side-by-side model comparison metrics
├── src/
│   ├── data_analysis.py              # Rating distribution analysis & visualization
│   ├── preprocessing.py              # Duration parsing, imputation, encoding & scaling
│   ├── model_training.py             # Decision Tree, Random Forest & Hyperparameter Tuning
│   ├── evaluation.py                 # Multi-class evaluation metrics & report generation
│   └── predict.py                    # Interactive inference CLI for existing or new titles
├── main.py                           # End-to-end execution pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## Task Workflow & Methodology

### Step 1: Analyze Rating Categories
- Computed frequency counts and percentage distributions across all rating categories.
- Handled class imbalance: Extremely rare categories (<10 occurrences such as `TV-Y7-FV`, `NC-17`, `UR`) were grouped into an **`Other`** class to allow meaningful stratified cross-validation and evaluation.
- Generated and saved a visual distribution chart to `outputs/rating_distribution.png`.

### Step 2: Prepare Training Datasets
- **Features Used:** `type`, `director`, `country`, `listed_in` (genres), `release_year`, `duration_value` (numeric amount), and `duration_unit` (e.g., `min`, `Seasons`).
- **Feature Exclusions:** `title`, `show_id`, and `rating` (target) were excluded from feature inputs.
- **Duration Parsing:** Extracted both the numeric value and the unit from textual durations (e.g., `"90 min"` -> `90.0`, `"min"`).
- **ColumnTransformer Pipeline:**
  - Categoricals (`type`, `director`, `country`, `listed_in`, `duration_unit`): Imputed with `"Unknown"` and encoded via `OneHotEncoder(handle_unknown="ignore", min_frequency=2)`.
  - Numericals (`release_year`, `duration_value`): Imputed with median and scaled using `StandardScaler`.

### Step 3: Train Classification Models
The dataset was split using an **80/20 stratified train/test split** (7,029 training samples, 1,758 testing samples). Three tree-based classifiers were trained with class weighting to address imbalance:
1. **Decision Tree Classifier** (`DecisionTreeClassifier`)
2. **Random Forest Classifier** (`RandomForestClassifier`)
3. **Extra Trees Classifier** (`ExtraTreesClassifier`)

### Step 4: Hyperparameter Tuning (Optimization)
- Applied `RandomizedSearchCV` on the **Random Forest** pipeline using Stratified K-Fold cross-validation (`cv=3`) optimizing for `f1_macro`.
- Explored parameter space:
  - `n_estimators`: `[200, 300, 400]`
  - `max_depth`: `[None, 15, 25, 35]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `max_features`: `["sqrt", 0.5]`
- **Best Identified Parameters:**
  ```python
  {
      'n_estimators': 200,
      'min_samples_split': 5,
      'min_samples_leaf': 1,
      'max_features': 0.5,
      'max_depth': 25
  }
  ```

### Step 5: Evaluate Prediction Accuracy
Models were evaluated on the held-out test set across **Accuracy**, **Macro Precision**, **Macro Recall**, and **Macro F1**:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Tuned Random Forest** | **0.5063** | **0.3924** | **0.4172** | **0.3942** |
| **Random Forest (Baseline)** | 0.4272 | 0.3038 | 0.4064 | 0.3260 |
| **Extra Trees** | 0.4152 | 0.3017 | 0.3808 | 0.3149 |
| **Decision Tree** | 0.3788 | 0.2885 | 0.3531 | 0.3085 |

> **Key Finding:** Hyperparameter optimization improved overall classification accuracy from 42.7% to **50.6%** and Macro F1 by **+20.9%** (0.326 -> 0.394) on an unbalanced 12-class prediction task.

---

## Artifacts Generated

1. **Serialized Pipeline:** `models/best_rating_model.joblib`
2. **Metadata File:** `models/model_metadata.joblib`
3. **Rating Distribution Chart:** `outputs/rating_distribution.png`
4. **Confusion Matrix Heatmap:** `outputs/confusion_matrix.png`
5. **Model Comparison CSV:** `outputs/model_comparison.csv`

---

## Interactive Rating Prediction CLI

The pipeline includes an interactive prediction menu in `src/predict.py`:
1. **Predict for an existing title:** Looks up any title in the dataset, feeds its attributes into the model, and compares the prediction against the actual label.
2. **Predict for a new custom title:** Prompts for type, director, country, release year, genres, and duration, returning the predicted rating and probability confidence score.

---

## How to Run

### Run Complete Pipeline & Interactive Menu
```bash
python task3/main.py
```
*(Option 1 to search titles, Option 2 for manual entry, Option 3 to exit)*
