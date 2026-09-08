import time

import numpy as np

from vector_db.exact import ExactIndex


EMBEDDINGS_PATH = "data/benchmark_embeddings.npy"

NUM_QUERIES = 500
TOP_K = 10
SEED = 42


def main():
    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Dataset shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    index = ExactIndex(
        dimension=dimension
    )

    print("Building exact index...")

    start = time.perf_counter()

    for i, vector in enumerate(embeddings):
        index.insert(
            str(i),
            vector,
        )

    build_time = time.perf_counter() - start

    print(
        f"Build time: {build_time:.3f} seconds"
    )

    rng = np.random.default_rng(SEED)

    query_indices = rng.choice(
        len(embeddings),
        size=NUM_QUERIES,
        replace=False,
    )

    queries = embeddings[query_indices]

    print()
    print(
        f"Running {NUM_QUERIES} queries..."
    )

    latencies = []

    for query in queries:
        start = time.perf_counter()

        index.search(
            query,
            k=TOP_K,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

    latencies = np.asarray(latencies)

    print()
    print("Exact Search Results")
    print("--------------------")
    print(f"Queries:       {NUM_QUERIES}")
    print(f"Top-k:         {TOP_K}")
    print(f"Average:       {latencies.mean():.3f} ms")
    print(
        f"P50:           {np.percentile(latencies, 50):.3f} ms"
    )
    print(
        f"P95:           {np.percentile(latencies, 95):.3f} ms"
    )
    print(
        f"P99:           {np.percentile(latencies, 99):.3f} ms"
    )


if __name__ == "__main__":
    main()