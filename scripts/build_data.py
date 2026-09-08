from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import brown
from sentence_transformers import SentenceTransformer


NUM_TEXTS = 5000
MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TEXTS_PATH = DATA_DIR / "texts.txt"


def load_texts():
    print("Loading Brown Corpus...")

    # Brown corpus is organized into sentences.
    sentences = brown.sents()

    texts = []

    for sentence in sentences:
        text = " ".join(sentence)

        # Ignore extremely short fragments.
        if len(text.split()) >= 8:
            texts.append(text)

        if len(texts) >= NUM_TEXTS:
            break

    print(f"Loaded {len(texts)} real text samples.")

    return texts


def create_embeddings(texts):
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def save_data(texts, embeddings):
    DATA_DIR.mkdir(exist_ok=True)

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    with open(
        TEXTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for text in texts:
            file.write(text + "\n")


def main():
    texts = load_texts()

    embeddings = create_embeddings(texts)

    save_data(texts, embeddings)

    print()
    print("Dataset created successfully.")
    print(f"Texts:      {len(texts)}")
    print(f"Dimensions: {embeddings.shape[1]}")
    print(f"Shape:      {embeddings.shape}")
    print(f"Saved to:   {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()