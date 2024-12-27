from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


def get_top(
    results: list[tuple[int, float, "ndarray"]],
        top_x: int) -> tuple[list["ndarray"], list[int]]:
    top_results = sorted(results, key=lambda x: x[1])[-top_x:]
    top_weights = [_[2] for _ in top_results]
    id_of_top_results = [_[0] for _ in top_results]
    return (top_weights, id_of_top_results)
