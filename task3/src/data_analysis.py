import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def analyze_ratings(df, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("NETFLIX AUDIENCE RATING ANALYSIS")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nRating counts:")
    print(df["rating"].value_counts(dropna=False))
    print("\nRating percentages:")
    print((df["rating"].value_counts(normalize=True, dropna=False) * 100).round(2))

    counts = df["rating"].dropna().value_counts().sort_values(ascending=False)
    ax = counts.plot(kind="bar", figsize=(12, 6), edgecolor="black")
    ax.set_title("Netflix Audience Rating Distribution")
    ax.set_xlabel("Audience rating")
    ax.set_ylabel("Number of titles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=150)
    plt.close()

    return counts
