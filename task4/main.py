import sys
from pathlib import Path

import joblib
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
sys.path.insert(0, str(TASK_DIR))

from src.data_loader import load_dataset
from src.clustering import fit_clustering
from src.analysis import build_cluster_summary, get_cluster_examples
from src.visualization import save_visualizations


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
MODEL_DIR = TASK_DIR / "models"
OUTPUT_DIR = TASK_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("          NETFLIX CONTENT SEGMENTATION")
    print("=" * 65)

    df = load_dataset(DATA_PATH)
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove exact duplicate content rows when possible, excluding show_id.
    duplicate_columns = [c for c in df.columns if c != "show_id"]
    before = len(df)
    df = df.drop_duplicates(subset=duplicate_columns).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows.")

    # Fit clustering and select K by silhouette score.
    print("\nEvaluating possible cluster counts...")
    result = fit_clustering(df, n_clusters=None)

    print(f"\nSelected number of clusters: {result['n_clusters']}")

    if not result["score_table"].empty:
        result["score_table"].to_csv(
            OUTPUT_DIR / "silhouette_scores.csv", index=False
        )

    clustered_df, summary = build_cluster_summary(df, result["labels"])
    examples = get_cluster_examples(clustered_df)

    # Save result tables.
    clustered_df.to_csv(OUTPUT_DIR / "clustered_netflix_titles.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "cluster_examples.csv", index=False)

    print("\nCLUSTER SUMMARY")
    print(summary.to_string(index=False))

    print("\nEXAMPLE TITLES")
    print(examples.to_string(index=False))

    # Save visualizations.
    save_visualizations(
        result["transformed_data"], clustered_df, OUTPUT_DIR
    )

    # Save reusable clustering artifacts.
    joblib.dump(
        {
            "preprocessor": result["preprocessor"],
            "kmeans": result["kmeans"],
            "n_clusters": result["n_clusters"]
        },
        MODEL_DIR / "netflix_kmeans_model.joblib"
    )

    print("\nCompleted successfully.")
    print(f"Model saved in: {MODEL_DIR}")
    print(f"Results and charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
