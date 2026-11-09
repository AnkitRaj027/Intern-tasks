import numpy as np
import pandas as pd
from src.recommendation import get_recommendations


def parse_genres(genre_str):
    """Split comma-separated genres into a set of cleaned strings."""
    if not isinstance(genre_str, str) or not genre_str.strip():
        return set()
    return {g.strip().lower() for g in genre_str.split(",") if g.strip()}


def evaluate_single_title(title, recommendations_df, df):
    """
    Evaluates recommendations for a single query title.
    Returns a dictionary of metrics:
      - genre_match_rate: % of recommendations sharing >= 1 genre (Precision@K)
      - avg_genre_overlap: average Jaccard similarity of genres
      - type_match_rate: % of recommendations matching Movie/TV Show
      - ndcg_score: Normalized Discounted Cumulative Gain (graded by Jaccard overlap)
      - mrr_score: Mean Reciprocal Rank of first relevant item
      - avg_similarity_score: mean similarity score
    """
    matches = df[df["title"].str.casefold() == title.strip().casefold()]
    if matches.empty or recommendations_df is None or recommendations_df.empty:
        return None

    query_row = matches.iloc[0]
    query_genres = parse_genres(query_row.get("listed_in", ""))
    query_type = str(query_row.get("type", "")).strip().lower()

    k = len(recommendations_df)
    genre_matches = 0
    type_matches = 0
    overlaps = []

    for _, row in recommendations_df.iterrows():
        rec_genres = parse_genres(row.get("listed_in", ""))
        rec_type = str(row.get("type", "")).strip().lower()

        # Genre Match (at least one shared genre)
        intersection = query_genres.intersection(rec_genres)
        if len(intersection) > 0:
            genre_matches += 1

        # Genre Overlap (Jaccard index)
        union = query_genres.union(rec_genres)
        overlap = len(intersection) / len(union) if len(union) > 0 else 0.0
        overlaps.append(overlap)

        # Content Type Match (Movie vs TV Show)
        if rec_type == query_type:
            type_matches += 1

    # NDCG@K calculation
    dcg = sum((score / np.log2(idx + 2)) for idx, score in enumerate(overlaps))
    ideal_scores = sorted(overlaps, reverse=True)
    idcg = sum((score / np.log2(idx + 2)) for idx, score in enumerate(ideal_scores))
    ndcg = (dcg / idcg) if idcg > 0 else 1.0

    # MRR (first item with >= 1 shared genre)
    first_rel_rank = next((idx + 1 for idx, score in enumerate(overlaps) if score > 0), None)
    mrr = (1.0 / first_rel_rank) if first_rel_rank else 0.0

    return {
        "title": title,
        "k": k,
        "genre_match_rate": (genre_matches / k) * 100,
        "type_match_rate": (type_matches / k) * 100,
        "avg_genre_overlap": float(np.mean(overlaps)) * 100 if overlaps else 0.0,
        "ndcg_score": float(ndcg),
        "mrr_score": float(mrr),
        "avg_similarity_score": float(recommendations_df["similarity_score"].mean()),
    }


def compute_intra_list_diversity(recommendations_df):
    """Calculates Intra-List Diversity (ILD) based on genre dissimilarity among recommended items."""
    if recommendations_df is None or len(recommendations_df) < 2:
        return 0.0

    rec_genres = [parse_genres(r.get("listed_in", "")) for _, r in recommendations_df.iterrows()]
    pair_distances = []

    for i in range(len(rec_genres)):
        for j in range(i + 1, len(rec_genres)):
            inter = len(rec_genres[i].intersection(rec_genres[j]))
            uni = len(rec_genres[i].union(rec_genres[j]))
            jaccard_sim = (inter / uni) if uni > 0 else 0.0
            pair_distances.append(1.0 - jaccard_sim)

    return float(np.mean(pair_distances)) if pair_distances else 0.0


def evaluate_model(titles, df, matrix, top_n=10, model_name="Model"):
    """
    Evaluates a recommendation model across a list of test titles.
    Computes Precision@K, Format Consistency, Jaccard Overlap, NDCG@K, MRR, ILD, and Catalog Coverage.
    """
    results = []
    ild_scores = []
    all_recommended_titles = set()

    for title in titles:
        recs = get_recommendations(title=title, df=df, tfidf_matrix=matrix, top_n=top_n)
        if recs is not None and not recs.empty:
            metrics = evaluate_single_title(title, recs, df)
            if metrics:
                results.append(metrics)
                ild_scores.append(compute_intra_list_diversity(recs))
                all_recommended_titles.update(recs["title"].tolist())

    if not results:
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(results)
    coverage_pct = (len(all_recommended_titles) / len(df)) * 100

    summary = {
        "Model": model_name,
        "Titles Evaluated": len(results_df),
        "Top-K": top_n,
        "Precision@10 (%)": round(results_df["genre_match_rate"].mean(), 2),
        "Format Match (%)": round(results_df["type_match_rate"].mean(), 2),
        "Avg Genre Overlap (%)": round(results_df["avg_genre_overlap"].mean(), 2),
        "NDCG@10": round(results_df["ndcg_score"].mean(), 4),
        "MRR": round(results_df["mrr_score"].mean(), 4),
        "Catalog Coverage (%)": round(coverage_pct, 2),
        "Unique Titles Surfaced": len(all_recommended_titles)
    }

    return results_df, summary


def compare_models(titles, df, standard_matrix, weighted_matrix, top_n=10):
    """
    Compares Standard TF-IDF vs Weighted TF-IDF on the given test titles.
    """
    print(f"\nEvaluating models on {len(titles)} test titles (Top-{top_n} recommendations)...")

    _, standard_summary = evaluate_model(
        titles=titles,
        df=df,
        matrix=standard_matrix,
        top_n=top_n,
        model_name="Standard Combined TF-IDF"
    )

    _, weighted_summary = evaluate_model(
        titles=titles,
        df=df,
        matrix=weighted_matrix,
        top_n=top_n,
        model_name="Weighted Multi-Feature TF-IDF"
    )

    comparison_df = pd.DataFrame([weighted_summary, standard_summary])
    return comparison_df

