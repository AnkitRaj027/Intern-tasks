from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy.sparse import hstack
def create_combined_features(df):

    print("\n========== FEATURE ENGINEERING ==========")

    df = df.copy()

    # Features selected for recommendation
    features = [
        "type",
        "director",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]

    # Handle missing values
    for feature in features:
        df[feature] = df[feature].fillna("")

    # Combine features into one text column
    df["combined_features"] = df[features].agg(
        " ".join,
        axis=1
    )

    print("\nCombined Features:")
    print(
        df[["title", "combined_features"]].head()
    )

    return df


def apply_tfidf(df):

    print("\n========== TF-IDF VECTORIZATION ==========")

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    # Convert combined text into numerical vectors
    tfidf_matrix = tfidf.fit_transform(
        df["combined_features"]
    )

    print("\nTF-IDF Matrix Shape:")
    print(tfidf_matrix.shape)

    print("\nTotal Features Extracted:")
    print(len(tfidf.get_feature_names_out()))

    print("\nSample Extracted Features:")
    print(tfidf.get_feature_names_out()[:20])

    return tfidf, tfidf_matrix




def create_weighted_tfidf(df):

    weights = {
        "listed_in": 0.50,
        "director": 0.20,
        "country": 0.10,
        "type": 0.10,
        "rating": 0.05,
        "duration": 0.05
    }

    matrices = []

    for feature, weight in weights.items():

        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(", ") if feature == "listed_in" else x.split(),
            token_pattern=None,
            lowercase=True
        )

        matrix = vectorizer.fit_transform(
            df[feature].fillna("").astype(str)
        )

        matrices.append(matrix * weight)

    final_matrix = hstack(matrices).tocsr()

    print("Weighted TF-IDF shape:", final_matrix.shape)

    return final_matrix


