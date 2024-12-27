import numpy as np


def get_top(
    results: list[tuple[int, float, np.ndarray]],
        top_x: int) -> tuple[list[np.ndarray], list[int]]:
    top_results = sorted(results, key=lambda x: x[1])[-top_x:]
    top_weights = [_[2] for _ in top_results]
    id_of_top_results = [_[0] for _ in top_results]
    return (top_weights, id_of_top_results)
