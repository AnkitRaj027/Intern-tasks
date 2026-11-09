
import pandas as pd


def preprocess_data(df):

    print("\n========== DATA PREPROCESSING ==========")

    # Create a copy to protect original data
    df = df.copy()

    # 1. Normalize text columns
    text_columns = [
        "type",
        "title",
        "director",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Convert text to lowercase
    for col in text_columns:
        df[col] = df[col].str.lower()

    # 3. Handle missing-like values
    df.replace(
        ["nan", "none", ""],
        pd.NA,
        inplace=True
    )

    # 4. Remove identifier and unused date column
    df.drop(
        columns=["show_id", "date_added"],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    print("\nPreprocessing completed!")

    print("\nProcessed Dataset:")
    print(df.head())

    print("\nProcessed Dataset Shape:")
    print(df.shape)

    return df
