"""Models a PI controller block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class PIController(Block):
    """
    Represent a Proportional-Integral (PI) controller in power system simulations.

    This class models a PI controller, which is a common control algorithm used in power systems to regulate
    the output of a system. It adjusts the input signal based on the error between the desired output and the
    current output.

    Attributes:
        k_p (float or torch.Tensor): Proportional gain of the PI controller.
        k_i (float or torch.Tensor): Integral gain of the PI controller.
        input (float or torch.Tensor): Current input to the PI controller.
        state_1 (float or torch.Tensor): Internal state of the PI controller.
    """

    def __init__(self, k_p, k_i):
        """
        Initialize the PI controller with specified parameters.

        Args:
            k_p (float): Proportional gain of the PI controller.
            k_i (float): Integral gain of the PI controller.
            lim_min (float): Minimum limit for the output.
            lim_max (float): Maximum limit for the output.
        """
        self.k_p = k_p
        self.k_i = k_i

        self.input = 0
        self.state_1 = 0

    def differential(self):
        """
        Compute the differential equations for the PI controller.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        dx1 = self.input
        return torch.stack(
            [
                dx1,
            ],
            axis=1,
        )

    def get_state_vector(self):
        """
        Retrieve the current state vector of the PI controller.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.stack(
            [
                self.state_1,
            ],
            axis=1,
        )

    def set_state_vector(self, x):
        """
        Set the state vector of the PI controller.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Compute the output of the PI controller.

        Args:
            input_var: The current input to the model.

        Returns: The output of the model.
        """
        self.input = input_var
        output = self.k_p * self.input + self.k_i * self.state_1
        return output

    def initialize(self, out_wish):
        """
        Initialize the PI controller with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns:
            The desired input to the model.
        """
        self.out_wish = out_wish
        self.state_1 = out_wish / self.k_i
        return out_wish

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.k_p = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_p
        self.k_i = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_i
        self.state_1 = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.state_1
        )

    def reset(self):
        """
        Reset the PI controller to its initial state.
        """
        self.input = torch.zeros_like(self.input)
        self.state_1 = self.out_wish * torch.ones_like(self.state_1) / self.k_i
