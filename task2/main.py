
import pandas as pd
import joblib

from pathlib import Path

from src.model_training import train_models
from src.evaluation import evaluate_models


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "Dataset.csv"

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)


# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Normalize target labels
df["type"] = df["type"].astype(str).str.strip()

# Remove rows with invalid target values
df = df[df["type"].isin(["Movie", "TV Show"])].copy()

# Remove duplicate content records
duplicate_columns = [
    "type",
    "title",
    "director",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in"
]

df = df.drop_duplicates(
    subset=duplicate_columns
).reset_index(drop=True)

# Normalize categorical text columns
for col in ["director", "country", "rating", "listed_in"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

print("Records after cleaning:", len(df))


# ==========================================
# STEP 3: FEATURE SELECTION
# ==========================================

feature_columns = [
    "director",
    "country",
    "release_year",
    "rating",
    "listed_in"
]

X = df[feature_columns].copy()

y = df["type"].copy()

print("\nTarget Distribution:")
print(y.value_counts())


# ==========================================
# STEP 4: TRAIN MODELS
# ==========================================

trained_models, X_test, y_test = train_models(X, y)


# ==========================================
# STEP 5: EVALUATE MODELS
# ==========================================

best_model, best_model_name, results = evaluate_models(
    trained_models,
    X_test,
    y_test
)


# ==========================================
# STEP 6: SAVE BEST MODEL
# ==========================================

model_path = MODEL_DIR / "best_model.joblib"

joblib.dump(best_model, model_path)

print("\nBest model saved successfully!")
print("Model location:", model_path)


# Save model comparison
results.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


# ==========================================
# STEP 7: PREDICT NEW NETFLIX TITLES
# ==========================================

new_titles = pd.DataFrame([

    {
        "director": "unknown",
        "country": "united states",
        "release_year": 2023,
        "rating": "tv-ma",
        "listed_in": "tv dramas, crime tv shows"
    },

    {
        "director": "christopher nolan",
        "country": "united states",
        "release_year": 2023,
        "rating": "pg-13",
        "listed_in": "action & adventure, sci-fi"
    }

])

predictions = best_model.predict(new_titles)

print("\n========== NEW TITLE PREDICTIONS ==========")

for i, prediction in enumerate(predictions):

    print(f"Title {i + 1}: {prediction}")


print("\nTask 2 completed successfully!")
