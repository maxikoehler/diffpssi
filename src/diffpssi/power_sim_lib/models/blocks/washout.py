"""Models a Washout block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class Washout(Block):
    """
    Represent a Washout filter in power system simulations.

    The Washout filter is a high-pass filter that allows signals with frequencies higher than a certain
    threshold to pass through while attenuating lower frequency signals. This filter is often used in
    control systems to isolate dynamic components of a signal.

    Args:
        k_w (float or torch.Tensor): Gain of the Washout filter.
        t_w (float or torch.Tensor): Time constant of the Washout filter.
        input (float or torch.Tensor): Current input to the Washout block.
        state_1 (float or torch.Tensor): Internal state of the Washout block.
    """

    def __init__(self, k_w, t_w):
        """
        Initialize the Washout filter with specified parameters.

        Args:
            k_w (float): Gain of the Washout filter.
            t_w (float): Time constant of the Washout filter.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.k_w = k_w
        self.t_w = t_w

        self.input = 0
        self.state_1 = 0

    def differential(self):
        """
        Compute the differential equations for the Washout filter.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        dx1 = 1 / self.t_w * (self.k_w * self.input - self.state_1)
        return torch.stack(
            [
                dx1,
            ],
            axis=1,
        )

    def get_state_vector(self):
        """
        Retrieve the current state vector of the Washout filter.

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
        Set the state vector of the Washout filter.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Compute the output of the Washout filter.

        Args:
            input_var: The current input to the model.

        Returns:
            The output of the model.
        """
        self.input = input_var
        output = 1 / self.t_w * (self.k_w * self.input - self.state_1)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.k_w = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_w
        self.t_w = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_w

    def initialize(self, out_wish):
        """
        Initialize the Washout filter with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns:
            The desired input to the model.
        """
        self.state_1 = self.k_w * out_wish
        return out_wish / self.k_w
