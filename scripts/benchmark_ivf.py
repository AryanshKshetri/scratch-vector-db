import time

import numpy as np

from vector_db.ivf import IVFIndex
from vector_db.exact import ExactIndex


def percentile(values, percentile_value):
    return float(
        np.percentile(
            values,
            percentile_value,
        )
    )


def calculate_recall(
    exact_results,
    approximate_results,
):
    exact_ids = {
        result["id"]
        for result in exact_results
    }

    approximate_ids = {
        result[0]
        for result in approximate_results
    }

    if not exact_ids:
        return 0.0

    matches = exact_ids.intersection(
        approximate_ids
    )

    return len(matches) / len(exact_ids)


def main():

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    print("Loading benchmark vectors...")

    vectors = np.load(
        "data/benchmark_embeddings.npy"
    )

    queries = np.load(
        "data/benchmark_queries.npy"
    )

    print(
        f"Dataset shape: {vectors.shape}"
    )

    print(
        f"Query shape: {queries.shape}"
    )

    dimension = vectors.shape[1]

    # -------------------------------------------------
    # Build exact index
    # -------------------------------------------------

    print()
    print("Building exact index...")

    exact_index = ExactIndex(
        dimension=dimension
    )

    start = time.perf_counter()

    for i, vector in enumerate(vectors):
        exact_index.insert(
            str(i),
            vector,
        )

    exact_build_time = (
        time.perf_counter() - start
    )

    print(
        f"Exact build time: "
        f"{exact_build_time:.3f} seconds"
    )

    print(
        f"Exact index count: "
        f"{exact_index.count()}"
    )

    # -------------------------------------------------
    # Compute exact ground truth
    # -------------------------------------------------

    print()
    print("Computing exact ground truth...")

    ground_truth = []

    exact_latencies = []

    for query in queries:

        start = time.perf_counter()

        results = exact_index.search(
            query,
            k=10,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        exact_latencies.append(
            elapsed
        )

        ground_truth.append(
            results
        )

    exact_average = np.mean(
        exact_latencies
    )

    exact_p50 = percentile(
        exact_latencies,
        50,
    )

    exact_p95 = percentile(
        exact_latencies,
        95,
    )

    exact_p99 = percentile(
        exact_latencies,
        99,
    )

    print()
    print("Exact Baseline")
    print("--------------")
    print(
        f"Average: "
        f"{exact_average:.3f} ms"
    )

    print(
        f"P50:     "
        f"{exact_p50:.3f} ms"
    )

    print(
        f"P95:     "
        f"{exact_p95:.3f} ms"
    )

    print(
        f"P99:     "
        f"{exact_p99:.3f} ms"
    )

    # -------------------------------------------------
    # Build IVF index
    # -------------------------------------------------

    print()
    print("Building IVF index...")

    ivf_index = IVFIndex(
        dimension=dimension,
        n_clusters=100,
        seed=42,
    )

    train_start = time.perf_counter()

    ivf_index.train(
        vectors
    )

    train_time = (
        time.perf_counter()
        - train_start
    )

    print(
        f"IVF training time: "
        f"{train_time:.3f} seconds"
    )

    # -------------------------------------------------
    # Insert vectors into IVF
    # -------------------------------------------------

    print()
    print("Inserting vectors into IVF...")

    insert_start = time.perf_counter()

    for i, vector in enumerate(vectors):

        ivf_index.insert(
            str(i),
            vector,
        )

    insert_time = (
        time.perf_counter()
        - insert_start
    )

    print(
        f"IVF insert time: "
        f"{insert_time:.3f} seconds"
    )

    print(
        f"IVF index count: "
        f"{ivf_index.size()}"
    )

    # -------------------------------------------------
    # Benchmark IVF
    # -------------------------------------------------

    nprobe_values = [
        1,
        5,
        10,
        25,
        100,
    ]

    print()
    print("IVF Benchmark")
    print("=" * 75)

    print(
        f"{'nprobe':<10}"
        f"{'Avg ms':<15}"
        f"{'P50 ms':<15}"
        f"{'P95 ms':<15}"
        f"{'Recall@10':<15}"
        f"{'Speedup':<10}"
    )

    print("-" * 75)

    for nprobe in nprobe_values:

        latencies = []
        recalls = []

        for query_index, query in enumerate(
            queries
        ):

            start = time.perf_counter()

            results = ivf_index.search(
                query,
                k=10,
                nprobe=nprobe,
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            latencies.append(
                elapsed
            )

            recall = calculate_recall(
                ground_truth[
                    query_index
                ],
                results,
            )

            recalls.append(
                recall
            )

        average_latency = np.mean(
            latencies
        )

        p50_latency = percentile(
            latencies,
            50,
        )

        p95_latency = percentile(
            latencies,
            95,
        )

        average_recall = np.mean(
            recalls
        )

        speedup = (
            exact_average
            / average_latency
        )

        print(
            f"{nprobe:<10}"
            f"{average_latency:<15.3f}"
            f"{p50_latency:<15.3f}"
            f"{p95_latency:<15.3f}"
            f"{average_recall:<15.4f}"
            f"{speedup:<10.2f}x"
        )


if __name__ == "__main__":
    main()