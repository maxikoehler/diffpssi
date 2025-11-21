"""Models an interface for all block type classes."""

from abc import ABC, abstractmethod

from diffpssi.power_sim_lib.backend import *


class Block(ABC):
    """Interface class for Block objects."""

    def __init__(self):
        """Initialize a Block class object."""
