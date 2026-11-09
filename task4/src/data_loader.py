from pathlib import Path
import pandas as pd


def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place Dataset.csv inside data/raw/."
        )
    df = pd.read_csv(path)
    required = ["title", "type", "director", "country", "release_year",
                "rating", "duration", "listed_in"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df
