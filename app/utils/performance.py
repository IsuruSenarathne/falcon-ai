import time
import functools
from typing import Callable, Any


def measure_time(prefix: str = "") -> Callable:
    """
    Decorator to measure and log function execution time.

    Args:
        prefix: Optional prefix for the log message (e.g., "🚀", "🤖")

    Returns:
        Decorated function that logs execution time

    Example:
        @measure_time("🤖 LLM inference")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            log_prefix = f"{prefix}: " if prefix else f"{func.__name__}: "
            print(f"{log_prefix}{elapsed:.2f}s")

            return result, elapsed
        return wrapper
    return decorator


class TimerContext:
    """Context manager for timing code blocks."""
    def __init__(self, label: str = ""):
        self.label = label
        self.start = None
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"{self.label}: {self.elapsed:.2f}s")


def timed_block(label: str = "") -> TimerContext:
    """
    Context manager for timing code blocks.

    Args:
        label: Optional label to print with elapsed time

    Returns:
        TimerContext that can be used as a context manager

    Example:
        with timed_block("KB retrieval") as timer:
            docs = vectorstore.similarity_search(query)
            print(f"Retrieved {len(docs)} documents in {timer.elapsed:.2f}s")
    """
    return TimerContext(label)
