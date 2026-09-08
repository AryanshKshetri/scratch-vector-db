# Vector Database From Scratch

A vector database implemented from scratch using **Python and NumPy**, with a **FastAPI** REST API.

The goal of this project is to understand what happens inside a vector database instead of relying on libraries such as Pinecone, FAISS, Chroma, or `sklearn.neighbors`.

## Project Goals

- Implement exact brute-force vector search from scratch.
- Implement cosine similarity using NumPy.
- Implement an approximate IVF-Flat index from scratch.
- Implement K-Means clustering without sklearn.
- Support vector insertion, search, and deletion.
- Compare exact and approximate search.
- Measure Recall@10, latency, speedup, and search candidates.
- Expose the vector database through FastAPI.

## Architecture

```text
Client
   |
   v
FastAPI
   |
   v
Vector Database
   |
   +-------------------+
   |                   |
   v                   v
Exact Index         IVF-Flat
   |                   |
   |                   +--- K-Means
   |                   +--- Inverted Lists
   |
   v
NumPy
```

## Current Status

- [x] Project setup
- [x] Python virtual environment
- [x] FastAPI application
- [x] Health endpoint
- [x] Exact vector index
- [x] Exact index tests
- [x] Text embedding pipeline
- [x] Real 5,000-text corpus
- [x] 50,000+ vector dataset
- [x] K-Means implementation
- [x] IVF-Flat implementation
- [ ] FastAPI vector endpoints
- [ ] Benchmark suite
- [ ] Recall@10 evaluation
- [ ] Interactive search demo
- [ ] Final benchmark results

## Technology Stack

- **Python** — application language
- **FastAPI** — REST API layer
- **NumPy** — vector arithmetic and indexing operations
- **Sentence Transformers** — converting text into embeddings
- **Uvicorn** — ASGI server
- **Pytest** — testing

### What we are NOT using

The vector database implementation does not use:

- Pinecone
- FAISS
- Chroma
- `sklearn.neighbors`
- Any external nearest-neighbor/vector database implementation

The actual vector search and indexing algorithms are implemented ourselves using NumPy.

---

# Exact Index

The first component of the project is an exact brute-force vector index.

For every query, the index compares the query vector against every stored vector.

```text
Query
  |
  v
Normalize query
  |
  v
Compare against every vector
  |
  v
Calculate cosine similarity
  |
  v
Select top-k
```

Because every vector is examined, this provides the **ground truth** that will later be used to evaluate the approximate IVF-Flat index.

## Cosine Similarity

Vectors are normalized when they are inserted.

For normalized vectors, cosine similarity can be calculated using their dot product:

```text
cosine_similarity(q, v) = q · v
```

The implementation uses NumPy matrix multiplication for this calculation.

## Exact Index API

The current index supports:

```python
insert(vector_id, vector)
search(query, k=10)
delete(vector_id)
count()
```

Deletion currently uses a **tombstone** approach. Deleted vector IDs are tracked separately instead of immediately rebuilding the underlying vector storage.

---

# Embedding Pipeline

The project uses a Sentence Transformer model to convert text into dense vector representations.

Current model:

```text
all-MiniLM-L6-v2
```

The model produces:

```text
text
  ↓
384-dimensional embedding
```

The embeddings are stored as NumPy arrays.

Generated data is intentionally excluded from Git because embeddings can become large and can be regenerated from the source dataset.

---

# Development Workflow

The project is being built incrementally.

Each major milestone is committed to Git so that the development history shows how the vector database was constructed.

Current commits include:

```text
chore: initialize vector database project
feat: implement exact vector index
feat: add text embedding pipeline
```

---

# Running the Project

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the FastAPI server:

```powershell
uvicorn app.main:app --reload
```

The API is currently available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
GET /health
```

FastAPI's interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Testing

Run the current test suite with:

```powershell
pytest
```

The exact index currently has tests covering:

- Vector insertion
- Similarity search
- Top-k ordering
- Vector deletion

---

# Roadmap

## Phase 1 — Foundation

- Project setup
- FastAPI skeleton
- Exact vector index
- Tests

## Phase 2 — Data

- Collect 5,000 real short texts
- Generate embeddings
- Build a 50,000+ vector dataset

## Phase 3 — Approximate Search

- Implement K-Means from scratch
- Create IVF centroids
- Build inverted lists
- Implement `nprobe`
- Implement IVF-Flat search

## Phase 4 — API

- Insert endpoint
- Search endpoint
- Delete endpoint
- Database statistics endpoint

## Phase 5 — Evaluation

Compare exact search against IVF-Flat using:

- Recall@10
- Average latency
- P95 latency
- Speedup
- Number of vectors examined

## Phase 6 — Demo

Build a simple interface where a user can enter a natural-language query and compare:

```text
Exact Search
vs.
IVF-Flat Search
```

The final goal is to demonstrate the fundamental tradeoff of approximate nearest-neighbor search:

```text
More search effort
       ↓
Higher recall
       ↓
Higher latency
```

# Benchmark Dataset

The initial corpus contains 5,000 real text samples.

To satisfy the 50,000+ vector benchmark requirement, we derive
additional vectors by applying small controlled perturbations to
the real embeddings and renormalizing them.

This produces a clustered vector space suitable for evaluating
approximate nearest-neighbor search.

The derived benchmark contains:

- 50,000 vectors
- 384 dimensions
- float32 representation
- deterministic generation using seed 42
