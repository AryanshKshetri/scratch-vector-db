import numpy as np

from vector_db.exact import ExactIndex


def test_insert_and_search():
    index = ExactIndex(dimension=3)

    index.insert(
        "dog",
        np.array([1.0, 0.0, 0.0])
    )

    index.insert(
        "cat",
        np.array([0.9, 0.1, 0.0])
    )

    index.insert(
        "car",
        np.array([0.0, 0.0, 1.0])
    )

    query = np.array([1.0, 0.0, 0.0])

    results = index.search(query, k=2)

    assert results[0]["id"] == "dog"
    assert results[1]["id"] == "cat"


def test_delete():
    index = ExactIndex(dimension=3)

    index.insert(
        "dog",
        np.array([1.0, 0.0, 0.0])
    )

    index.insert(
        "cat",
        np.array([0.9, 0.1, 0.0])
    )

    index.delete("dog")

    results = index.search(
        np.array([1.0, 0.0, 0.0]),
        k=2
    )

    assert results[0]["id"] == "cat"