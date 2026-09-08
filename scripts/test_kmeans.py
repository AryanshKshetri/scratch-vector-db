import numpy as np

from vector_db.kmeans import KMeans


def main():
    print("Loading benchmark vectors...")

    vectors = np.load(
        "data/benchmark_embeddings.npy"
    )

    print(
        f"Dataset shape: {vectors.shape}"
    )

    # Use 100 clusters as planned.

    kmeans = KMeans(
        n_clusters=100,
        max_iterations=10,
        seed=42,
    )

    kmeans.fit(vectors)

    assignments = kmeans.predict(
        vectors
    )

    print()
    print("K-Means completed.")
    print(
        f"Centroids shape: "
        f"{kmeans.centroids.shape}"
    )

    print(
        f"Assignments shape: "
        f"{assignments.shape}"
    )

    # Check cluster distribution.

    counts = np.bincount(
        assignments,
        minlength=100,
    )

    print()
    print("Cluster statistics")
    print("-------------------")
    print(
        f"Minimum cluster size: {counts.min()}"
    )
    print(
        f"Maximum cluster size: {counts.max()}"
    )
    print(
        f"Average cluster size: {counts.mean():.2f}"
    )


if __name__ == "__main__":
    main()