import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, prepare_feature_frame


def find_best_k(X, k_values=range(2, 9), random_state=42):
    """
    Compare candidate K values using multi-index triangulation:
      - Silhouette Score (higher is better)
      - Davies-Bouldin Index (lower is better)
      - Calinski-Harabasz Variance Ratio (higher is better)
      - Inertia / Sum of Squared Errors
    """
    results = []
    n_samples = X.shape[0]
    X_dense = X.toarray() if hasattr(X, "toarray") else X

    for k in k_values:
        if k >= n_samples:
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

        sil = silhouette_score(
            X, labels,
            sample_size=min(2000, n_samples),
            random_state=random_state
        )
        ch = calinski_harabasz_score(X_dense, labels)
        db = davies_bouldin_score(X_dense, labels)
        inertia = model.inertia_

        results.append({
            "k": k,
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 3),
            "calinski_harabasz_score": round(ch, 1),
            "inertia": round(inertia, 1)
        })
        print(f"K={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.3f}, Calinski-Harabasz={ch:.1f}, Inertia={inertia:.1f}")

    if not results:
        raise ValueError("Not enough records to evaluate clustering.")

    results_df = pd.DataFrame(results).sort_values("silhouette_score", ascending=False)
    
    # Save multi-index metrics to outputs
    try:
        from pathlib import Path
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        results_df.to_csv(out_dir / "cluster_metrics_comparison.csv", index=False)
    except Exception:
        pass

    return int(results_df.iloc[0]["k"]), results_df



def fit_clustering(df, n_clusters=None, random_state=42):
    features = prepare_feature_frame(df)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(features)

    if n_clusters is None:
        n_clusters, score_table = find_best_k(X, random_state=random_state)
    else:
        score_table = pd.DataFrame()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(X)

    return {
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "labels": labels,
        "transformed_data": X,
        "score_table": score_table,
        "n_clusters": n_clusters
    }
