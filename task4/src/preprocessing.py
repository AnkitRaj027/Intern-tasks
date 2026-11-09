import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["type", "country", "rating", "listed_in", "duration_unit"]
NUMERICAL_FEATURES = ["release_year", "duration_value"]


def prepare_feature_frame(df):
    """Prepare content attributes. Title and show_id are intentionally excluded."""
    data = df.copy()

    for col in ["type", "country", "rating", "listed_in", "release_year", "duration"]:
        if col not in data.columns:
            data[col] = np.nan

    # Duration examples: '90 min', '2 Seasons'
    duration = data["duration"].fillna("").astype(str).str.strip()
    extracted = duration.str.extract(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>.*)")
    data["duration_value"] = pd.to_numeric(extracted["value"], errors="coerce")
    data["duration_unit"] = extracted["unit"].str.strip().replace("", np.nan)
    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype("object")

    return data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]


def build_preprocessor():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
    ])
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numerical", numerical, NUMERICAL_FEATURES)
    ])
