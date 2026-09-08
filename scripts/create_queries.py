import numpy as np


def main():
    print("Loading benchmark embeddings...")

    vectors = np.load(
        "data/benchmark_embeddings.npy"
    )

    print(
        f"Dataset shape: {vectors.shape}"
    )

    rng = np.random.default_rng(42)

    # Select 500 vectors as queries.
    query_indices = rng.choice(
        len(vectors),
        size=500,
        replace=False,
    )

    queries = vectors[query_indices].copy()

    np.save(
        "data/benchmark_queries.npy",
        queries,
    )

    np.save(
        "data/benchmark_query_indices.npy",
        query_indices,
    )

    print()
    print("Query dataset created.")
    print(
        f"Query shape: {queries.shape}"
    )

    print(
        "Saved:"
    )
    print(
        "  data/benchmark_queries.npy"
    )
    print(
        "  data/benchmark_query_indices.npy"
    )


if __name__ == "__main__":
    main()