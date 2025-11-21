"""Models an Lag block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class Lag(Block):
    """Represent a Lag block in power system simulations.

    This class models a first-order lag system, which is commonly used in control systems to represent
    systems with a delay in response. It adjusts the input signal based on the time constant and gain.

    Args:
        ki (float or torch.Tensor): Gain of the Lag block.
        kd (float or torch.Tensor): Time constant of the Lag block.
        lim_min (float or torch.Tensor): Minimum limit for the output.
        lim_max (float or torch.Tensor): Maximum limit for the output.
        input (float or torch.Tensor): Current input to the Lag block.
        state_1 (float or torch.Tensor): Internal state of the Lag block.
    """

    def __init__(self, ki, kd, lim_min, lim_max):
        """Initialize the Lag block with specified parameters.

        Args:
            ki (float): Gain of the Lag block.
            kd (float): Time constant of the Lag block.
            lim_min (float): Minimum limit for the output.
            lim_max (float): Maximum limit for the output.
        """
        self.ki = ki
        self.kd = kd
        self.lim_min = lim_min
        self.lim_max = lim_max

        self.input = 0

        self.state_1 = 0

    def differential(self):
        """Compute the differential equations for the PT1Limited model.

        Returns: A tensor containing the derivatives of the state variables.
        """
        dx1 = -self.kd * self.state_1 + self.ki * self.input
        # noinspection PyTypeChecker
        dx1 = torch.where(
            torch.logical_or(
                torch.logical_and(self.state_1.real <= self.lim_min, dx1.real < 0),
                torch.logical_and(self.state_1.real >= self.lim_max, dx1.real > 0),
            ),
            0,
            dx1,
        )
        return torch.stack(
            [
                dx1,
            ],
            axis=1,
        )

    def get_state_vector(self):
        """Retrieve the current state vector of the PT1Limited model.

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
        """Set the state vector of the PT1Limited model.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """Compute the output of the PT1Limited model.

        Args:
            input_var (torch.Tensor): The current input to the model.

        Returns: The output of the model.
        """
        self.input = input_var * torch.ones_like(self.input)
        # noinspection PyTypeChecker
        output = torch.minimum(
            torch.maximum(self.state_1.real, self.lim_min), self.lim_max
        )
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.input = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.input
        self.state_1 = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.state_1
        )
        self.ki = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.ki
        self.kd = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.kd
        self.lim_min = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_min
        )
        self.lim_max = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_max
        )

    def initialize(self, out_wish):
        """Initialize the PT1Limited model with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.
        """
