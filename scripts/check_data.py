import numpy as np


embeddings = np.load("data/embeddings.npy")

print("Shape:", embeddings.shape)
print("Data type:", embeddings.dtype)
print("First vector:")
print(embeddings[0])