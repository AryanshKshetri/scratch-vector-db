from typing import List

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from vector_db.exact import ExactIndex
from vector_db.ivf import IVFIndex


app = FastAPI(
    title="Scratch Vector Database",
    description="A vector database built from scratch using NumPy.",
    version="1.0.0",
)


DIMENSION = 384
N_CLUSTERS = 100


DIMENSION = 384
N_CLUSTERS = 100

MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    MODEL_NAME
)


def load_text_corpus():
    with open(
        "data/texts.txt",
        "r",
        encoding="utf-8",
    ) as file:
        texts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return texts


texts = load_text_corpus()


# -------------------------------------------------
# Request models
# -------------------------------------------------


class VectorRequest(BaseModel):
    id: str
    vector: List[float]


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    nprobe: int = 5


class TextSearchRequest(BaseModel):
    text: str
    k: int = 10
    nprobe: int = 5

# -------------------------------------------------
# In-memory indexes
# -------------------------------------------------


exact_index = ExactIndex(
    dimension=DIMENSION
)

ivf_index = IVFIndex(
    dimension=DIMENSION,
    n_clusters=N_CLUSTERS,
    seed=42,
)


# -------------------------------------------------
# Load benchmark vectors into IVF
# -------------------------------------------------


def load_ivf_index():
    print("Loading benchmark vectors...")

    vectors = np.load(
        "data/benchmark_embeddings.npy"
    )

    print(
        f"Dataset shape: {vectors.shape}"
    )

    if vectors.shape[1] != DIMENSION:
        raise ValueError(
            f"Expected dimension {DIMENSION}, "
            f"got {vectors.shape[1]}"
        )

    print("Training IVF index...")

    ivf_index.train(
        vectors
    )

    print("Inserting vectors into IVF...")

    for i, vector in enumerate(vectors):
        ivf_index.insert(
            str(i),
            vector,
        )

    print(
        f"IVF index ready with "
        f"{ivf_index.size()} vectors."
    )


# Load IVF when the application starts.
load_ivf_index()


# -------------------------------------------------
# Health check
# -------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Scratch Vector Database API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "exact_vectors": exact_index.count(),
        "ivf_vectors": ivf_index.size(),
    }


# -------------------------------------------------
# Exact search
# -------------------------------------------------


@app.post("/search/exact")
def search_exact(request: SearchRequest):

    try:
        results = exact_index.search(
            np.asarray(
                request.vector,
                dtype=np.float32,
            ),
            k=request.k,
        )

        return {
            "results": results,
            "count": len(results),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# -------------------------------------------------
# IVF search
# -------------------------------------------------


@app.post("/search/ivf")
def search_ivf(request: SearchRequest):

    try:
        results = ivf_index.search(
            np.asarray(
                request.vector,
                dtype=np.float32,
            ),
            k=request.k,
            nprobe=request.nprobe,
        )

        formatted_results = [
            {
                "id": vector_id,
                "score": score,
            }
            for vector_id, score in results
        ]

        return {
            "results": formatted_results,
            "count": len(formatted_results),
            "nprobe": request.nprobe,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/search/text")
def search_text(request: TextSearchRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:
        # Convert text into embedding.
        embedding = embedding_model.encode(
            request.text,
            normalize_embeddings=True,
        )

        # Search our IVF index.
        results = ivf_index.search(
            np.asarray(
                embedding,
                dtype=np.float32,
            ),
            k=request.k,
            nprobe=request.nprobe,
        )

        formatted_results = []

        for vector_id, score in results:

            index = int(vector_id)

            if index >= len(texts):
                continue

            formatted_results.append(
                {
                    "id": vector_id,
                    "text": texts[index],
                    "score": score,
                }
            )

        return {
            "query": request.text,
            "results": formatted_results,
            "count": len(formatted_results),
            "nprobe": request.nprobe,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

# -------------------------------------------------
# Exact insert
# -------------------------------------------------


@app.post("/vectors/exact")
def insert_exact(request: VectorRequest):

    try:
        exact_index.insert(
            request.id,
            np.asarray(
                request.vector,
                dtype=np.float32,
            ),
        )

        return {
            "message": "Vector inserted",
            "id": request.id,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# -------------------------------------------------
# Exact delete
# -------------------------------------------------


@app.delete("/vectors/exact/{vector_id}")
def delete_exact(vector_id: str):

    deleted = exact_index.delete(
        vector_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Vector not found",
        )

    return {
        "message": "Vector deleted",
        "id": vector_id,
    }