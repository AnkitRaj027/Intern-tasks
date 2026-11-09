
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_content_distribution(df):

    print("\n========== CONTENT DISTRIBUTION ==========")

    # Movies vs TV Shows
    print("\nContent Type Distribution:")
    print(df["type"].value_counts())

    # Percentage distribution
    print("\nContent Type Percentage:")
    print(
        df["type"].value_counts(normalize=True) * 100
    )

    # Release year distribution
    print("\nTop 10 Release Years:")
    print(df["release_year"].value_counts().head(10))

    # Rating distribution
    print("\nContent Rating Distribution:")
    print(df["rating"].value_counts())

    # Most common genres
    print("\nTop 10 Genres:")
    genres = df["listed_in"].str.split(", ").explode()
    print(genres.value_counts().head(10))

    # Top countries
    print("\nTop 10 Countries:")
    countries = df["country"].str.split(", ").explode()
    print(countries.value_counts().head(10))



def visualize_content_distribution(df):

    sns.set_theme(style="whitegrid")

    # 1. Movies vs TV Shows
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="type",
        hue="type",
        legend=False
    )

    plt.title("Movies vs TV Shows Distribution")
    plt.xlabel("Content Type")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.show()


    # 2. Content released over the years
    plt.figure(figsize=(12, 5))

    yearly_content = df["release_year"].value_counts().sort_index()

    plt.plot(
        yearly_content.index,
        yearly_content.values
    )

    plt.title("Netflix Content Distribution by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.show()


    # 3. Top 10 genres
    plt.figure(figsize=(10, 6))

    genres = df["listed_in"].str.split(", ").explode()
    top_genres = genres.value_counts().head(10)

    sns.barplot(
        x=top_genres.values,
        y=top_genres.index
    )

    plt.title("Top 10 Netflix Genres")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()


    # 4. Top 10 countries
    plt.figure(figsize=(10, 6))

    countries = df["country"].str.split(", ").explode()
    top_countries = countries.value_counts().head(10)

    sns.barplot(
        x=top_countries.values,
        y=top_countries.index
    )

    plt.title("Top 10 Content Producing Countries")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()


    # 5. Content rating distribution
    plt.figure(figsize=(10, 6))

    rating_counts = df["rating"].value_counts()

    sns.barplot(
        x=rating_counts.values,
        y=rating_counts.index
    )

    plt.title("Netflix Content Rating Distribution")
    plt.xlabel("Number of Titles")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.show()
