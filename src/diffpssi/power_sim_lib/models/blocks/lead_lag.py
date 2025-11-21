"""Models a LeadLag block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class LeadLag(object):
    """
    Represent a Lead-Lag control block in power system simulations.

    This class models a Lead-Lag compensator, which is commonly used in control systems to improve
    the stability and speed of response. It adjusts the phase of a signal and can be used to
    compensate for delays in a control system.

    Args:
        t_1 (float or torch.Tensor): Time constant for the lead part of the block.
        t_2 (float or torch.Tensor): Time constant for the lag part of the block.
        input (float or torch.Tensor): Current input to the LeadLag block.
        state_1 (float or torch.Tensor): Internal state of the LeadLag block.
    """

    def __init__(self, t_1, t_2):
        """
        Initialize the LeadLag block with specified parameters.

        Args:
            t_1 (float): Time constant for the lead part of the block.
            t_2 (float): Time constant for the lag part of the block.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.t_1 = t_1
        self.t_2 = t_2

        self.input = 0
        self.state_1 = 0

    def differential(self):
        """
        Compute the differential equations for the LeadLag block.

        Returns:
            A tensor containing the derivatives of the state variables.
        """
        dx1 = (1 / self.t_2) * (self.input - self.state_1)
        return torch.stack(
            [
                dx1,
            ],
            axis=1,
        )

    def get_state_vector(self):
        """
        Retrieve the current state vector of the LeadLag block.

        Returns:
            The current state vector of the model.
        """
        return torch.stack(
            [
                self.state_1,
            ],
            axis=1,
        )

    def set_state_vector(self, x):
        """
        Set the state vector of the LeadLag block.

        Args:
            x (torch.tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Compute the output of the LeadLag block.

        Args:
            input_var: The current input to the model.

        Returns:
            The output of the model.
        """
        self.input = input_var
        output = (
            self.t_1 / self.t_2 * input_var + (1 - (self.t_1 / self.t_2)) * self.state_1
        )
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.t_1 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_1
        self.t_2 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_2

    def initialize(self, out_wish):
        """
        Initialize the LeadLag block with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns:
            The desired input to the model.
        """
        self.state_1 = out_wish
        return out_wish
