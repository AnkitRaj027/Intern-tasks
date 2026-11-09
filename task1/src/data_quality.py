
def analyze_data_quality(df):

    print("\n========== DATA QUALITY ANALYSIS ==========")

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Total missing values
    print("\nTotal Missing Values:")
    print(df.isnull().sum().sum())

    # Duplicate records
    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    # Missing percentage
    print("\nMissing Values Percentage:")
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    print(missing_percentage.sort_values(ascending=False))

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Unique titles
    print("\nUnique Titles:")
    print(df["title"].nunique())

    # Duplicate titles
    print("\nDuplicate Titles:")
    print(df["title"].duplicated().sum())



def investigate_duplicate_titles(df):

    print("\n========== DUPLICATE TITLE INVESTIGATION ==========")

    duplicates = df[
        df["title"].duplicated(keep=False)
    ].sort_values("title")

    columns = [
        "show_id",
        "title",
        "type",
        "director",
        "country",
        "release_year",
        "rating",
        "duration",
        "listed_in"
    ]

    print(duplicates[columns].to_string(index=False))

    print("\nTotal duplicate title records:", len(duplicates))


def remove_duplicate_records(df):

    print("\n========== DUPLICATE REMOVAL ==========")

    initial_count = len(df)

    # Ignore show_id while checking duplicate content
    df = df.drop_duplicates(
        subset=[
            "type",
            "title",
            "director",
            "country",
            "date_added",
            "release_year",
            "rating",
            "duration",
            "listed_in"
        ]
    ).reset_index(drop=True)

    removed_count = initial_count - len(df)

    print("Original records:", initial_count)
    print("Removed records:", removed_count)
    print("Remaining records:", len(df))

    return df
