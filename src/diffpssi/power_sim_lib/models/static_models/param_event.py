"""Models parameter changing events in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class ParamEvent(StaticModelInterface):
    """
    Represent a parameter event in a power system simulation.

    This class models a parameter event that changes the value of a parameter for a specific
    component within a specified time window, allowing the simulation of dynamic parameter
    changes in the system.

    Attributes:
        start_time (float): The start time of the parameter event.
        end_time (float): The end time of the parameter event.
        component (object): The component to which the parameter change applies.
        parameter (str): The name of the parameter to change.
        value (float): The new value of the parameter.
    """

    def __init__(self, start_time, model, param_name, value):
        """
        Initialize the ParameterEvent object with needed values.

        Args:
            start_time (float): The start time of the parameter event.
            parameter (str): The parameter to change
            value (float): The new value of the parameter.
        """
        self.start_time = start_time
        self.model = model
        self.param_name = param_name
        self.value = value
        self.handled = False

    def handle_event(self, t):
        """
        Check if the parameter event is active at a given time.

        Args:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        if not self.handled and t >= self.start_time:
            # set the attribute by name
            setattr(self.model, self.param_name, self.value)
            self.handled = True

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations for the bus.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.value = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.value


class ParamDependencyTime(ParamEvent):
    """
    Represent a parameter variation dependent on the time in a power system simulation.

    This class models a parameter event that changes the value of a parameter for a specific component
    within a specified time window, allowing the simulation of dynamic parameter changes in the system.

    Attributes:
        start_time (float): The start time of the parameter event.
        end_time (float): The end time of the parameter event.
        component (object): The component to which the parameter change applies.
        parameter (str): The name of the parameter to change.
        value (callable): The new value of the parameter.
    """

    def __init__(
        self, start_time: float, model: object, param_name: str, function: callable
    ):
        """
        Initialize a ParameterDependencyEvent object.

        Dependent on the start time, end time, component, parameter, and value.

        Args:
            start_time (float): The start time of the parameter event.
            parameter (str): The parameter to change
            value (float): The new value of the parameter.
        """
        self.start_time = start_time
        self.model = model
        self.param_name = param_name
        self.value = function
        self.handled = False

        self.parallel_sims = 1

    def handle_event(self, t):
        """
        Check if the parameter event is active at a given time.

        Args:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        if not self.handled and t >= self.start_time:
            # set the attribute by name
            setattr(
                self.model,
                self.param_name,
                self.value(t)
                * torch.ones((self.parallel_sims, 1), dtype=torch.complex128),
            )

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations for the bus.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.parallel_sims = parallel_sims
