"""Models a short circuit event in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class ScEvent(StaticModelInterface):
    """
    Represent a short circuit event in a power system simulation.

    This class models a short circuit event at a specific bus within a specified time window,
    allowing the simulation of transient conditions and system response to faults.

    Attributes:
        start_time (float): The start time of the short circuit event.
        end_time (float): The end time of the short circuit event.
        bus (int): The index of the bus where the short circuit occurs.
    """

    def __init__(self, start_time, end_time, bus):
        """
        Initialize the ScEvent object with the start time, end time, and bus index.

        Args:
            start_time (float): The start time of the short circuit event.
            end_time (float): The end time of the short circuit event.
            bus (int): The index of the bus where the short circuit occurs.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.bus = bus

    def is_active(self, t):
        """
        Check if the short circuit event is active at a given time.

        Args:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        return bool(self.start_time < t <= self.end_time)
