"""Interface for static models in the power system simulation library."""

from abc import ABC, abstractmethod


class StaticModelInterface(ABC):
    """Abstract base class for static models in the power system simulation library."""

    def __init__(self):
        """Initialize the static model interface."""
