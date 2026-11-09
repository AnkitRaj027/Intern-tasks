
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(title, df, tfidf_matrix, top_n=10):

    # Find the selected movie
    matches = df[
        df["title"].str.casefold() == title.strip().casefold()
    ]

    if matches.empty:
        print(f"Title '{title}' not found!")
        return None

    # Get the row position (integer location) of the selected title
    idx = df.index.get_loc(matches.index[0])

    # Calculate similarity with all other titles
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort titles by similarity score
    similar_indices = similarity_scores.argsort()[::-1]

    # Remove the selected title itself
    similar_indices = [
        i for i in similar_indices if i != idx
    ]

    # Select top N recommendations
    top_indices = similar_indices[:top_n]

    recommendations = df.iloc[top_indices][
        ["title", "type", "release_year", "listed_in"]
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[i] for i in top_indices
    ]

    return recommendations.reset_index(drop=True)
