import pandas as pd


def build_cluster_summary(df, labels):
    result = df.copy().reset_index(drop=True)
    result["cluster"] = labels

    summaries = []
    for cluster_id, group in result.groupby("cluster"):
        top_genres = (
            group["listed_in"].dropna().astype(str)
            .str.split(",").explode().str.strip()
            .value_counts().head(5).index.tolist()
        )
        top_countries = group["country"].dropna().astype(str).str.split(",").explode().str.strip()
        top_countries = top_countries.value_counts().head(3).index.tolist()

        top_ratings = group["rating"].dropna().astype(str).value_counts().head(3).index.tolist()
        type_mix = group["type"].fillna("Unknown").value_counts(normalize=True)

        summaries.append({
            "cluster": int(cluster_id),
            "title_count": len(group),
            "movie_percentage": round(type_mix.get("Movie", 0) * 100, 2),
            "tv_show_percentage": round(type_mix.get("TV Show", 0) * 100, 2),
            "average_release_year": round(
                pd.to_numeric(group["release_year"], errors="coerce").mean(), 1
            ),
            "top_genres": ", ".join(top_genres),
            "top_countries": ", ".join(top_countries),
            "common_ratings": ", ".join(top_ratings)
        })

    return result, pd.DataFrame(summaries).sort_values("cluster")


def get_cluster_examples(clustered_df, n=5):
    examples = []
    for cluster_id, group in clustered_df.groupby("cluster"):
        titles = group["title"].dropna().astype(str).head(n).tolist()
        examples.append({
            "cluster": int(cluster_id),
            "example_titles": " | ".join(titles)
        })
    return pd.DataFrame(examples)
