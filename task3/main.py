import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup paths and imports
TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from src.data_analysis import analyze_ratings
from src.preprocessing import prepare_features
from src.model_training import train_and_tune_models
from src.evaluation import evaluate_models
from src.predict import prediction_menu

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
MODEL_DIR = TASK_DIR / "models"
OUTPUT_DIR = TASK_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ==========================================
    # STEP 1: LOAD & ANALYZE AUDIENCE RATINGS
    # ==========================================
    print("\n========== STEP 1: LOADING & ANALYZING DATASET ==========")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please ensure Dataset.csv is placed inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully! Shape:", df.shape)

    # Clean whitespace and handle missing target labels
    df["rating"] = df["rating"].astype("string").str.strip()
    df["rating"] = df["rating"].replace("", pd.NA)
    df = df.dropna(subset=["rating"]).copy()

    # Normalize text columns
    for col in ["title", "type", "director", "country", "listed_in", "duration"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Combine rare rating categories (<10 records) into 'Other' for robust stratification
    min_class_count = 10
    rating_counts = df["rating"].value_counts()
    rare_classes = rating_counts[rating_counts < min_class_count].index
    if len(rare_classes) > 0:
        print("\nCombining rare rating categories (<10 occurrences) into 'Other':", list(rare_classes))
        df["rating"] = df["rating"].where(~df["rating"].isin(rare_classes), "Other")

    # Run exploratory data analysis and export distribution chart
    analyze_ratings(df, OUTPUT_DIR)

    # ==========================================
    # STEP 2: PREPARE TRAINING DATASETS
    # ==========================================
    print("\n========== STEP 2: PREPARING TRAINING DATASETS ==========")
    feature_df = prepare_features(df)
    target = df["rating"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

    print("Training samples:", len(X_train))
    print("Testing samples: ", len(X_test))
    print("Features extracted:", list(feature_df.columns))

    # ===================================================
    # STEP 3 & 4: TRAIN CLASSIFIERS & HYPERPARAMETER TUNING
    # ===================================================
    print("\n========== STEP 3 & 4: MODEL TRAINING & HYPERPARAMETER TUNING ==========")
    models = train_and_tune_models(X_train, y_train)

    # ==========================================
    # STEP 5: EVALUATE PREDICTION ACCURACY
    # ==========================================
    print("\n========== STEP 5: MODEL EVALUATION ==========")
    best_name, best_model, results = evaluate_models(
        models, X_test, y_test, OUTPUT_DIR
    )

    # ==========================================
    # STEP 6: SAVE BEST MODEL & ARTIFACTS
    # ==========================================
    print("\n========== STEP 6: SAVING BEST MODEL & ARTIFACTS ==========")
    model_path = MODEL_DIR / "best_rating_model.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(
        {
            "model_name": best_name,
            "feature_columns": list(feature_df.columns),
            "rare_class_threshold": min_class_count
        },
        MODEL_DIR / "model_metadata.joblib"
    )

    print(f"Best model ({best_name}) saved to: {model_path}")
    print(f"Visual outputs and CSV comparison saved to: {OUTPUT_DIR}")

    # ==========================================
    # STEP 7: SAMPLE AUDIENCE RATING PREDICTIONS
    # ==========================================
    print("\n========== STEP 7: SAMPLE AUDIENCE RATING PREDICTIONS ==========")

    sample_titles = ["stranger things", "the irishman", "cocomelon", "breaking bad"]
    for title in sample_titles:
        matches = df[df["title"].astype(str).str.strip().str.casefold() == title.casefold()]
        if not matches.empty:
            row = matches.iloc[0]
            feat = prepare_features(pd.DataFrame([row.to_dict()]))
            pred = best_model.predict(feat)[0]
            actual = row.get("rating", "N/A")
            print(f"Title: {row['title']}")
            print(f"  • Predicted Rating: {pred}")
            print(f"  • Actual Rating:    {actual}")
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(feat)[0]
                confidence = probs.max() * 100
                print(f"  • Confidence:       {confidence:.1f}%\n")

    # Sample new custom title
    new_custom_title = pd.DataFrame([{
        "type": "Movie",
        "director": "Christopher Nolan",
        "country": "United States",
        "release_year": 2023,
        "listed_in": "Action & Adventure, Sci-Fi",
        "duration": "180 min"
    }])
    new_feat = prepare_features(new_custom_title)
    new_pred = best_model.predict(new_feat)[0]
    print("New Custom Title: Christopher Nolan Sci-Fi Movie (180 min)")
    print(f"  • Predicted Rating: {new_pred}")
    if hasattr(best_model, "predict_proba"):
        new_confidence = best_model.predict_proba(new_feat)[0].max() * 100
        print(f"  • Confidence:       {new_confidence:.1f}%\n")

    print("Task 3 completed successfully!")

    # If running interactively, offer interactive lookup menu
    if sys.stdin.isatty():
        try:
            run_menu = input("\nLaunch interactive prediction menu? (y/n, default=n): ").strip().lower()
            if run_menu in ["y", "yes"]:
                prediction_menu(best_model, df)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
