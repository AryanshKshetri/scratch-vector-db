import numpy as np

from vector_db.kmeans import KMeans


class IVFIndex:
    def __init__(
        self,
        dimension,
        n_clusters=100,
        seed=42,
    ):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.seed = seed

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            max_iterations=10,
            seed=seed,
        )

        # Inverted lists:
        # cluster_id -> list of vector IDs
        self.inverted_lists = [
            [] for _ in range(n_clusters)
        ]

        # Store vectors by ID.
        self.vectors = {}

        # Store which cluster each vector belongs to.
        self.cluster_ids = {}

        self.is_trained = False

    def train(self, vectors):
        """
        Train the IVF index using K-Means.
        """

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                "Vector dimension does not match index."
            )

        print("Training IVF index...")

        self.kmeans.fit(vectors)

        self.is_trained = True

        print(
            f"IVF trained with "
            f"{self.n_clusters} clusters."
        )

    def insert(self, vector_id, vector):
        """
        Insert one vector into the IVF index.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Index must be trained before insertion."
            )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.shape != (self.dimension,):
            raise ValueError(
                "Vector has incorrect dimension."
            )

        if vector_id in self.vectors:
            raise ValueError(
                f"Vector ID already exists: {vector_id}"
            )

        # Find the nearest centroid.
        cluster_id = self._nearest_cluster(
            vector
        )

        # Store vector.
        self.vectors[vector_id] = vector

        # Store cluster assignment.
        self.cluster_ids[vector_id] = cluster_id

        # Add ID to inverted list.
        self.inverted_lists[
            cluster_id
        ].append(vector_id)

    def delete(self, vector_id):
        """
        Delete a vector from the IVF index.

        Returns True if deleted, False if the ID
        does not exist.
        """

        if vector_id not in self.vectors:
            return False

        cluster_id = self.cluster_ids[
            vector_id
        ]

        # Remove from inverted list.
        self.inverted_lists[
            cluster_id
        ].remove(vector_id)

        # Remove stored vector.
        del self.vectors[vector_id]

        # Remove cluster assignment.
        del self.cluster_ids[vector_id]

        return True

    def search(
        self,
        query,
        k=10,
        nprobe=5,
    ):
        """
        Approximate nearest-neighbor search.

        nprobe determines how many nearest clusters
        are searched.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Index must be trained before searching."
            )

        query = np.asarray(
            query,
            dtype=np.float32,
        )

        if query.shape != (self.dimension,):
            raise ValueError(
                "Query has incorrect dimension."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        if nprobe <= 0:
            raise ValueError(
                "nprobe must be greater than 0."
            )

        # Cannot probe more clusters than exist.
        nprobe = min(
            nprobe,
            self.n_clusters,
        )

        # -------------------------------------------------
        # Step 1:
        # Find distance from query to every centroid.
        # -------------------------------------------------

        centroids = self.kmeans.centroids

        query_squared_norm = np.sum(
            query * query
        )

        centroid_squared_norms = np.sum(
            centroids * centroids,
            axis=1,
        )

        distances = (
            query_squared_norm
            + centroid_squared_norms
            - 2 * (centroids @ query)
        )

        # -------------------------------------------------
        # Step 2:
        # Select nprobe nearest clusters.
        # -------------------------------------------------

        selected_clusters = np.argpartition(
            distances,
            nprobe - 1,
        )[:nprobe]

        # -------------------------------------------------
        # Step 3:
        # Collect candidate vectors from
        # the selected inverted lists.
        # -------------------------------------------------

        candidate_ids = []

        for cluster_id in selected_clusters:
            candidate_ids.extend(
                self.inverted_lists[cluster_id]
            )

        if not candidate_ids:
            return []

        # -------------------------------------------------
        # Step 4:
        # Convert candidate vectors into NumPy array.
        # -------------------------------------------------

        candidate_vectors = np.asarray(
            [
                self.vectors[vector_id]
                for vector_id in candidate_ids
            ],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Step 5:
        # Calculate cosine similarity.
        # -------------------------------------------------

        query_norm = np.linalg.norm(
            query
        )

        vector_norms = np.linalg.norm(
            candidate_vectors,
            axis=1,
        )

        similarities = (
            candidate_vectors @ query
        ) / (
            vector_norms * query_norm
            + 1e-10
        )

        # -------------------------------------------------
        # Step 6:
        # Select top-k candidates.
        # -------------------------------------------------

        k = min(
            k,
            len(candidate_ids),
        )

        top_indices = np.argpartition(
            -similarities,
            k - 1,
        )[:k]

        # Sort selected results by similarity.
        top_indices = top_indices[
            np.argsort(
                -similarities[top_indices]
            )
        ]

        # -------------------------------------------------
        # Step 7:
        # Return (ID, similarity) pairs.
        # -------------------------------------------------

        results = []

        for index in top_indices:
            results.append(
                (
                    candidate_ids[index],
                    float(
                        similarities[index]
                    ),
                )
            )

        return results

    def _nearest_cluster(self, vector):
        """
        Find the closest centroid for one vector.
        """

        centroids = self.kmeans.centroids

        vector_squared_norm = np.sum(
            vector * vector
        )

        centroid_squared_norms = np.sum(
            centroids * centroids,
            axis=1,
        )

        distances = (
            vector_squared_norm
            + centroid_squared_norms
            - 2 * (centroids @ vector)
        )

        return int(
            np.argmin(distances)
        )

    def size(self):
        """
        Return number of vectors stored.
        """

        return len(self.vectors)