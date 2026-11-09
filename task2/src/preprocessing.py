import re
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer


def tokenize_genres(text):
    """
    Module-level tokenizer function so pipeline is pickle/joblib serializable.
    Strips explicit target format leakage words ('tv', 'movie', 'show') from genre tags
    so the model learns pure content themes without label leakage.
    """
    if not isinstance(text, str):
        return []
    tokens = []
    for g in text.split(","):
        g = g.strip().lower()
        g = re.sub(r"\b(tv shows?|tv|movies?|show|shows)\b", "", g).strip()
        g = re.sub(r"\s+", " ", g).strip()
        if g:
            tokens.append(g)
    return tokens


def prepare_features(df=None):

    # Features used for prediction
    categorical_features = [
        "director",
        "country",
        "rating"
    ]

    numerical_features = [
        "release_year"
    ]

    # Categorical preprocessing for single-value columns
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    # Multi-label genre preprocessing (splits comma-separated genre tags)
    genre_vectorizer = CountVectorizer(
        tokenizer=tokenize_genres,
        token_pattern=None,
        binary=True
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("scaler", StandardScaler())
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "genres",
            genre_vectorizer,
            "listed_in"
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ])

    return preprocessor
