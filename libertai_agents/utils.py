from typing import TypeVar, Callable

T = TypeVar("T")


def find(f: Callable[[T], bool], seq: list[T]) -> T | None:
    """Return first item in sequence where f(item) == True."""
    for item in seq:
        if f(item):
            return item
    return None
