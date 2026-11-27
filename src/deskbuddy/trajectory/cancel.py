"""Cooperative cancellation for preview planning running in worker threads.

A newer click makes an in-flight preview stale. Planning code calls
``raise_if_cancelled()`` at loop boundaries so stale work ends within
milliseconds instead of occupying the single planning worker for seconds.
"""

import threading
from contextlib import contextmanager

_local = threading.local()


class PlanningCancelled(ValueError):
    """Raised inside a worker when its preview request has been superseded."""


@contextmanager
def planning_scope(cancelled):
    previous = getattr(_local, "cancelled", None)
    _local.cancelled = cancelled
    try:
        yield
    finally:
        _local.cancelled = previous


def raise_if_cancelled():
    check = getattr(_local, "cancelled", None)
    if check is not None and check():
        raise PlanningCancelled("Superseded by a newer target")
