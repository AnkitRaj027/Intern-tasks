import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "type", "director", "country", "listed_in", "duration_unit"
]
NUMERICAL_FEATURES = ["release_year", "duration_value"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def prepare_features(df):
    """Create model-ready features without using the target or title."""
    data = df.copy()

    for col in ["type", "director", "country", "listed_in"]:
        if col not in data.columns:
            data[col] = np.nan

    if "release_year" not in data.columns:
        data["release_year"] = np.nan

    if "duration" not in data.columns:
        data["duration"] = np.nan

    # Extract the numeric amount and unit from values such as '90 min' or '2 Seasons'.
    duration_text = data["duration"].fillna("").astype(str).str.strip()
    extracted = duration_text.str.extract(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>.*)")
    data["duration_value"] = pd.to_numeric(extracted["value"], errors="coerce")
    data["duration_unit"] = extracted["unit"].str.strip().replace("", np.nan)

    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype("object")

    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
    data["duration_value"] = pd.to_numeric(data["duration_value"], errors="coerce")

    return data[FEATURE_COLUMNS]


def build_preprocessor():
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES)
    ])
