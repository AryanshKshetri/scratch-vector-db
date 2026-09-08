from pathlib import Path

import numpy as np


INPUT_PATH = Path("data/embeddings.npy")
OUTPUT_PATH = Path("data/benchmark_embeddings.npy")

TARGET_SIZE = 50_000
SEED = 42


def build_benchmark():
    print("Loading base embeddings...")

    base_embeddings = np.load(INPUT_PATH)

    print(f"Base shape: {base_embeddings.shape}")

    rng = np.random.default_rng(SEED)

    num_base_vectors = len(base_embeddings)

    repeats = TARGET_SIZE // num_base_vectors
    remainder = TARGET_SIZE % num_base_vectors

    chunks = []

    for _ in range(repeats):
        noise = rng.normal(
            loc=0.0,
            scale=0.01,
            size=base_embeddings.shape,
        ).astype(np.float32)

        vectors = base_embeddings + noise

        # Normalize again after adding noise.
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        vectors = vectors / norms

        chunks.append(vectors)

    if remainder > 0:
        extra = base_embeddings[:remainder]

        noise = rng.normal(
            loc=0.0,
            scale=0.01,
            size=extra.shape,
        ).astype(np.float32)

        extra = extra + noise

        norms = np.linalg.norm(
            extra,
            axis=1,
            keepdims=True,
        )

        extra = extra / norms

        chunks.append(extra)

    benchmark_embeddings = np.vstack(chunks)

    np.save(
        OUTPUT_PATH,
        benchmark_embeddings,
    )

    print()
    print("Benchmark dataset created.")
    print(f"Shape: {benchmark_embeddings.shape}")
    print(f"Dtype: {benchmark_embeddings.dtype}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_benchmark()