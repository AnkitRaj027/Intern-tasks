
import pandas as pd

from src.data_loader import load_dataset
from src.data_quality import analyze_data_quality
from src.data_quality import investigate_duplicate_titles
from src.data_quality import remove_duplicate_records
from src.eda import analyze_content_distribution
from src.eda import visualize_content_distribution 
from src.preprocessing import preprocess_data
from src.feature_engineering import create_combined_features
from src.feature_engineering import apply_tfidf
from src.recommendation import get_recommendations
from src.feature_engineering import create_weighted_tfidf
from src.evaluation import evaluate_single_title, compare_models

DATASET_PATH = "data/raw/Dataset.csv"

df = load_dataset(DATASET_PATH)
df = remove_duplicate_records(df)

analyze_data_quality(df)
investigate_duplicate_titles(df)
analyze_content_distribution(df)

processed_df = preprocess_data(df)
processed_df = create_combined_features(processed_df)

weighted_matrix = create_weighted_tfidf(processed_df)

tfidf_vectorizer, tfidf_matrix = apply_tfidf(processed_df)

weighted_recommendations = get_recommendations(
    title="stranger things",
    df=processed_df,
    tfidf_matrix=weighted_matrix,
    top_n=10
)

print("\n========== WEIGHTED RECOMMENDATIONS ==========")
print(weighted_recommendations)


print("\n========== RECOMMENDATION EVALUATION ==========")
eval_metrics = evaluate_single_title("stranger things", weighted_recommendations, processed_df)
if eval_metrics:
    print(f"Metrics for '{eval_metrics['title']}':")
    print(f"  • Genre Match Rate (Precision@{eval_metrics['k']}): {eval_metrics['genre_match_rate']:.1f}%")
    print(f"  • Content Type Match Rate:           {eval_metrics['type_match_rate']:.1f}%")
    print(f"  • Average Genre Overlap:             {eval_metrics['avg_genre_overlap']:.1f}%")
    print(f"  • Average Similarity Score:          {eval_metrics['avg_similarity_score']:.4f}")


test_titles = ["stranger things", "breaking bad", "narcos", "the queen's gambit", "the irishman"]
comparison_df = compare_models(
    titles=test_titles,
    df=processed_df,
    standard_matrix=tfidf_matrix,
    weighted_matrix=weighted_matrix,
    top_n=10
)
print("\n========== MODEL COMPARISON SUMMARY ==========")
print(comparison_df.to_string(index=False))

