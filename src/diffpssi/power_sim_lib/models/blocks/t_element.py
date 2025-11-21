"""Models a TElement block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class TElement(Block):
    """Represent a T Element (time delay) in power system simulations.

    This class models a time delay element, which delays the input signal by a specified time constant.
    It is useful in control systems to simulate the effect of time delays in the system.
    Attributes:
        t_delay (float or torch.Tensor): Time constant of the time delay element.
        input (float or torch.Tensor): Current input to the TElement block.
        state_1 (float or torch.Tensor): Internal state of the TElement block.
    """

    def __init__(self, t_delay, t_i=0.005):
        """Initialize the TElement block with a specified time delay.

        Args:
            t_delay (float): Time constant of the time delay element.
        """
        self.integrator = Integrator(t_i)
        self.t_delay = t_delay

    def differential(self):
        """Compute the differential equations for the TElement block.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        dx1 = self.integrator.differential()
        return torch.stack(
            [
                dx1,
            ],
            axis=1,
        )

    def get_state_vector(self):
        """Retrieve the current state vector of the TElement block.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.stack(
            [
                self.integrator.state_1,
            ],
            axis=1,
        )

    def set_state_vector(self, x):
        """Set the state vector of the TElement block.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.integrator.state_1 = x[:, 0]

    def get_output(self, input_var):
        """Compute the output of the TElement block.

        Args:
            input_var: The current input to the model.
        Returns: The output of the model.
        """
        output = self.integrator.get_output(input_var)
        if output >= self.t_delay:
            self.integrator.set_state_vector(torch.zeros_like(self.integrator.state_1))
            return output
        else:
            return 0

    def enable_parallel_simulation(self, parallel_sims):
        """Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.t_delay = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_delay
        )

    def initialize(self):
        """Initialize the TElement block with a specified output."""
        self.integrator.initialize()

    def reset(self):
        """Reset the state vector of the TElement block to zero."""
        self.integrator.set_state_vector(torch.zeros_like(self.integrator.state_1))
