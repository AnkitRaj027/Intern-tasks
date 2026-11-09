import os
import sys
import pandas as pd

# Ensure task1 and task1/src can be imported regardless of execution directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_loader import load_dataset
from src.data_quality import remove_duplicate_records
from src.preprocessing import preprocess_data
from src.feature_engineering import create_combined_features, apply_tfidf, create_weighted_tfidf
from src.recommendation import get_recommendations
from src.evaluation import evaluate_single_title, evaluate_model, compare_models


def run_evaluation():
    # 1. Resolve dataset path
    dataset_path = "data/raw/Dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(current_dir), "data", "raw", "Dataset.csv")

    print(f"Loading dataset from: {dataset_path}")
    raw_df = load_dataset(dataset_path)

    # 2. Clean and preprocess
    clean_df = remove_duplicate_records(raw_df)
    processed_df = preprocess_data(clean_df)
    processed_df = create_combined_features(processed_df)

    # 3. Create feature matrices
    print("\nBuilding Standard TF-IDF Matrix...")
    _, standard_matrix = apply_tfidf(processed_df)

    print("\nBuilding Weighted TF-IDF Matrix...")
    weighted_matrix = create_weighted_tfidf(processed_df)

    # 4. Detailed evaluation for a sample title
    test_title = "breaking bad"
    print(f"\n==================================================")
    print(f" DETAILED EVALUATION FOR: '{test_title.upper()}'")
    print(f"==================================================")

    recs = get_recommendations(test_title, processed_df, weighted_matrix, top_n=10)
    print("\nTop-10 Recommendations (Weighted Model):")
    print(recs[["title", "type", "release_year", "listed_in", "similarity_score"]])

    single_eval = evaluate_single_title(test_title, recs, processed_df)
    print("\nMetrics for this recommendation:")
    print(f"  • Genre Match Rate (Precision@10): {single_eval['genre_match_rate']:.1f}%")
    print(f"  • Type Consistency (TV vs Movie): {single_eval['type_match_rate']:.1f}%")
    print(f"  • Average Genre Overlap:          {single_eval['avg_genre_overlap']:.1f}%")
    print(f"  • Mean Similarity Score:          {single_eval['avg_similarity_score']:.4f}")

    # 5. Evaluate models across a statistically representative sample of 50 titles
    import numpy as np
    np.random.seed(42)
    sample_indices = np.random.choice(len(processed_df), size=50, replace=False)
    test_titles = processed_df.iloc[sample_indices]["title"].tolist()

    print(f"\n==================================================")
    print(f" MODEL COMPARISON ACROSS {len(test_titles)} CATALOG TITLES")
    print(f"==================================================")

    comparison_results = compare_models(
        titles=test_titles,
        df=processed_df,
        standard_matrix=standard_matrix,
        weighted_matrix=weighted_matrix,
        top_n=10
    )

    print("\nEvaluation Comparison Summary:")
    print(comparison_results.to_string(index=False))

    # Save to outputs
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    comparison_results.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to: {csv_path}")

    # Generate visualization chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["Precision@10", "Format Match", "Jaccard Overlap", "NDCG@10"]
    weighted_scores = [
        float(comparison_results.iloc[0]["Precision@10 (%)"]),
        float(comparison_results.iloc[0]["Format Match (%)"]),
        float(comparison_results.iloc[0]["Avg Genre Overlap (%)"]),
        float(comparison_results.iloc[0]["NDCG@10"]) * 100
    ]
    standard_scores = [
        float(comparison_results.iloc[1]["Precision@10 (%)"]),
        float(comparison_results.iloc[1]["Format Match (%)"]),
        float(comparison_results.iloc[1]["Avg Genre Overlap (%)"]),
        float(comparison_results.iloc[1]["NDCG@10"]) * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    r1 = ax.bar(x - width/2, weighted_scores, width, label="Weighted Multi-Feature TF-IDF", color="#0f172a")
    r2 = ax.bar(x + width/2, standard_scores, width, label="Standard Combined TF-IDF", color="#94a3b8")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="600", color="#334155")
    ax.set_title("Task 1: Recommendation Engine Benchmark (50 Diverse Catalog Titles)", fontsize=11, fontweight="700", color="#0f172a", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, fontweight="500", color="#1e293b")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#64748b")

    plt.tight_layout()
    chart_path = os.path.join(outputs_dir, "model_comparison.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison chart to: {chart_path}")

    print("\n==================================================")
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    run_evaluation()
