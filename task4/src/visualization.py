from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def save_visualizations(X, clustered_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Project the high-dimensional encoded data into 2D for visualization only.
    svd = TruncatedSVD(n_components=2, random_state=42)
    points = svd.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        points[:, 0], points[:, 1],
        c=clustered_df["cluster"],
        cmap="tab10",
        s=16,
        alpha=0.7
    )
    plt.title("Netflix Content Clusters (2D SVD Projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_visualization.png", dpi=160)
    plt.close()

    counts = clustered_df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar", edgecolor="black")
    plt.title("Number of Netflix Titles per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of titles")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=160)
    plt.close()

    # Distribution of content type in each cluster.
    type_table = pd.crosstab(
        clustered_df["cluster"], clustered_df["type"], normalize="index"
    ) * 100
    ax = type_table.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Content Type Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage")
    ax.legend(title="Content type", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_content_types.png", dpi=160)
    plt.close()

    return points
