import numpy as np

from vector_db.ivf import IVFIndex


def main():
    print("Loading benchmark vectors...")

    vectors = np.load(
        "data/benchmark_embeddings.npy"
    )

    print(
        f"Dataset shape: {vectors.shape}"
    )

    dimension = vectors.shape[1]

    # -------------------------------------------------
    # Create IVF index
    # -------------------------------------------------

    index = IVFIndex(
        dimension=dimension,
        n_clusters=100,
        seed=42,
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    index.train(vectors)

    # -------------------------------------------------
    # Insert vectors
    # -------------------------------------------------

    print("Inserting vectors...")

    for i, vector in enumerate(vectors):
        index.insert(
            str(i),
            vector,
        )

    print(
        f"Inserted {index.size()} vectors."
    )

    # -------------------------------------------------
    # Cluster statistics
    # -------------------------------------------------

    cluster_sizes = np.array(
        [
            len(cluster)
            for cluster in index.inverted_lists
        ]
    )

    print()
    print("Inverted List Statistics")
    print("-------------------------")
    print(
        f"Number of clusters: "
        f"{len(index.inverted_lists)}"
    )
    print(
        f"Minimum size: "
        f"{cluster_sizes.min()}"
    )
    print(
        f"Maximum size: "
        f"{cluster_sizes.max()}"
    )
    print(
        f"Average size: "
        f"{cluster_sizes.mean():.2f}"
    )

    # -------------------------------------------------
    # Test different nprobe values
    # -------------------------------------------------

    query = vectors[0]

    print()
    print("IVF Search Tests")
    print("================")

    for nprobe in [1, 5, 10, 25, 100]:

        results = index.search(
            query,
            k=10,
            nprobe=nprobe,
        )

        print()
        print(
            f"nprobe = {nprobe}"
        )
        print("----------------")

        for vector_id, score in results[:3]:
            print(
                f"ID: {vector_id}, "
                f"similarity: {score:.6f}"
            )


if __name__ == "__main__":
    main()