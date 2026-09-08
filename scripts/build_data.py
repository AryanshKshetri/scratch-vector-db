from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_TEXTS = 5000
MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TEXTS_PATH = DATA_DIR / "texts.txt"


# ---------------------------------------------------------
# Create a small real-text corpus
# ---------------------------------------------------------

texts = [
    "The dog is playing in the park.",
    "A cat is sleeping on the sofa.",
    "The weather is sunny and warm today.",
    "Machine learning models learn patterns from data.",
    "Python is widely used for artificial intelligence.",
    "The football team won the championship.",
    "The restaurant serves excellent Italian food.",
    "A computer uses memory to store information.",
    "The ocean contains millions of different species.",
    "Scientists are studying climate change.",
]


# ---------------------------------------------------------
# Repeat the seed corpus to reach 5,000 texts
# ---------------------------------------------------------

texts = [
    texts[i % len(texts)] + f" Example number {i}."
    for i in range(NUM_TEXTS)
]


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print(f"Generating embeddings for {len(texts)} texts...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)

embeddings = embeddings.astype(np.float32)


# ---------------------------------------------------------
# Save data
# ---------------------------------------------------------

DATA_DIR.mkdir(exist_ok=True)

np.save(EMBEDDINGS_PATH, embeddings)

with open(TEXTS_PATH, "w", encoding="utf-8") as file:
    for text in texts:
        file.write(text + "\n")


print()
print("Dataset created successfully.")
print(f"Texts:      {len(texts)}")
print(f"Dimensions: {embeddings.shape[1]}")
print(f"Shape:      {embeddings.shape}")
print(f"Saved to:   {EMBEDDINGS_PATH}")