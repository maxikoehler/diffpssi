"""Voltage controller classes for power system simulations."""

from abc import ABC
from math import floor

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, Lag, PT1Limited
from diffpssi.power_sim_lib.models.static_models import Transformer_Old


class Voltage_Controller(ABC):
    """
    Abstract base class for voltage controllers.

    Concrete voltage controllers (OLTC, FSM, etc.) should inherit from this
    class and implement the interface methods documented below.

    Args:
        N/A: Instances are created through concrete subclasses.
    """

    def __init__(self):
        """
        Initialize the voltage controller base class.

        Subclasses may extend initialization to set controller parameters.

        Returns:
            None
        """
        pass

    def differential(self):
        """
        Return the derivative vector of the controller's dynamic states.

        Subclasses with dynamic state (integrators, filters, etc.) should
        return a torch.Tensor containing the time derivatives arranged
        per parallel simulation instance.

        Returns:
            torch.Tensor or int: Derivative vector or 0 for stateless controllers.
        """
        pass

    def get_output(self):
        """
        Compute and return the controller output (e.g. tap ratio ``u_l``).

        Implementations should accept an optional measurement argument when
        required (e.g. measured bus voltage) and return the controller's
        current output for each parallel simulation.

        Returns:
            torch.Tensor or float: Controller output (per-simulation).
        """
        pass

    def get_state_vector(self):
        """
        Return the internal state vector of the controller.

        Returns:
            torch.Tensor or object: Controller-specific state representation.
        """
        pass

    def set_state_vector(self, x=None):
        """
        Set the controller's internal state vector from an external value.

        Args:
            x (optional): State vector or representation accepted by the
                controller (type is implementation-dependent).

        Returns:
            None
        """
        pass

    def enable_parallel_simulation(self, parallel_sims):
        """
        Prepare the controller for parallel simulation execution.

        Implementations should convert scalar parameters to tensors sized
        according to ``parallel_sims`` and prepare internal sub-blocks
        (integrators, filters) for batched operation.

        Args:
            parallel_sims (int): Number of parallel simulations to enable.

        Returns:
            None
        """
        pass

    def initialize(self, voltage=None):
        """
        Initialize controller state prior to simulation.

        This method is typically used to set initial references, initialize
        internal integrators/filters and to set the measurement state.

        Args:
            voltage (optional): Initial measured voltage or reference used for
                initialization (implementation-dependent).

        Returns:
            None
        """
        pass
