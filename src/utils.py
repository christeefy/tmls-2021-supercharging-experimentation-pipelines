import time
from typing import Optional


class Timer:
    def __init__(self, n_iter: int):
        self.n_iter = n_iter
        self.start: Optional[int] = None

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        elapsed = self.start - end
        print(
            f"Total time: {elapsed:2f}\nTook {(elapsed / self.n_iter):2f} per iteration"
        )
