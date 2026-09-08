import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters=100,
        max_iterations=20,
        seed=42,
    ):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.seed = seed

        self.centroids = None

    def fit(self, vectors):
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        n_samples = vectors.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(
                "Number of vectors must be >= number of clusters"
            )

        rng = np.random.default_rng(self.seed)

        # -------------------------------------------------
        # Initialize centroids using random data points
        # -------------------------------------------------

        indices = rng.choice(
            n_samples,
            size=self.n_clusters,
            replace=False,
        )

        centroids = vectors[indices].copy()

        # -------------------------------------------------
        # K-Means iterations
        # -------------------------------------------------

        for iteration in range(self.max_iterations):

            print(
                f"K-Means iteration {iteration + 1}/"
                f"{self.max_iterations}"
            )

            # Calculate squared Euclidean distance:
            #
            # ||x - c||²
            #
            # Using:
            #
            # ||x-c||² = ||x||² + ||c||² - 2x·c

            vector_norms = np.sum(
                vectors * vectors,
                axis=1,
                keepdims=True,
            )

            centroid_norms = np.sum(
                centroids * centroids,
                axis=1,
            )

            distances = (
                vector_norms
                + centroid_norms
                - 2 * vectors @ centroids.T
            )

            # Assign every vector to nearest centroid

            assignments = np.argmin(
                distances,
                axis=1,
            )

            # -------------------------------------------------
            # Recalculate centroids
            # -------------------------------------------------

            new_centroids = np.zeros_like(
                centroids
            )

            for cluster_id in range(
                self.n_clusters
            ):
                members = vectors[
                    assignments == cluster_id
                ]

                if len(members) == 0:
                    # Empty cluster:
                    # choose a random vector as centroid
                    random_index = rng.integers(
                        n_samples
                    )

                    new_centroids[
                        cluster_id
                    ] = vectors[random_index]

                else:
                    new_centroids[
                        cluster_id
                    ] = members.mean(axis=0)

            # -------------------------------------------------
            # Check convergence
            # -------------------------------------------------

            centroid_shift = np.linalg.norm(
                new_centroids - centroids
            )

            print(
                f"Centroid shift: {centroid_shift:.6f}"
            )

            centroids = new_centroids

            if centroid_shift < 1e-4:
                print("K-Means converged.")
                break

        self.centroids = centroids.astype(
            np.float32
        )

        return self

    def predict(self, vectors):
        if self.centroids is None:
            raise RuntimeError(
                "K-Means has not been fitted yet."
            )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        vector_norms = np.sum(
            vectors * vectors,
            axis=1,
            keepdims=True,
        )

        centroid_norms = np.sum(
            self.centroids * self.centroids,
            axis=1,
        )

        distances = (
            vector_norms
            + centroid_norms
            - 2 * vectors @ self.centroids.T
        )

        return np.argmin(
            distances,
            axis=1,
        )