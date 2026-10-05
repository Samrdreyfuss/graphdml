"""Warnings raised by graphdml."""

__all__ = ["GraphDMLWarning"]


class GraphDMLWarning(UserWarning):
    """A fitted estimate may be unreliable (small focal set, weak overlap, collinearity, ...)."""
