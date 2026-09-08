import numpy as np


class ExactIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ids = []
        self.deleted = set()

    def insert(self, vector_id: str, vector: np.ndarray):
        vector = np.asarray(vector, dtype=np.float32)

        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected vector of dimension {self.dimension}, "
                f"got {vector.shape}"
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Cannot insert a zero vector.")

        vector = vector / norm

        self.vectors.append(vector)
        self.ids.append(vector_id)

        if vector_id in self.deleted:
            self.deleted.remove(vector_id)

    def search(self, query: np.ndarray, k: int = 10):
        query = np.asarray(query, dtype=np.float32)

        if query.shape != (self.dimension,):
            raise ValueError(
                f"Expected query of dimension {self.dimension}, "
                f"got {query.shape}"
            )

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Cannot search with a zero vector.")

        query = query / norm

        if not self.vectors:
            return []

        vectors = np.asarray(self.vectors)

        scores = vectors @ query

        for i, vector_id in enumerate(self.ids):
            if vector_id in self.deleted:
                scores[i] = -np.inf

        k = min(k, len(self.vectors))

        top_indices = np.argpartition(
            scores,
            -k
        )[-k:]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        results = []

        for index in top_indices:
            if scores[index] == -np.inf:
                continue

            results.append(
                {
                    "id": self.ids[index],
                    "score": float(scores[index]),
                }
            )

        return results

    def delete(self, vector_id: str):
        if vector_id not in self.ids:
            return False

        self.deleted.add(vector_id)
        return True

    def count(self):
        return len(self.vectors) - len(self.deleted)