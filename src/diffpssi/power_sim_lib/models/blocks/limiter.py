"""Models a limiter block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class Limiter(Block):
    """
    Represent a simple limiter block in power system simulations.

    This class is used to limit the output of a signal within a specified range. It is a basic yet crucial
    component in various control and simulation scenarios.

    Attributes:
        limit (float or torch.Tensor): The limit value for both positive and negative sides.
    """

    def __init__(self, limit):
        """
        Initialize the Limiter block with a specified limit.

        Args:
            limit (float): The limit value for both positive and negative sides.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.limit = limit

    def get_output(self, input_var):
        """
        Compute the output of the Limiter model.

        Args:
            input_var (torch.Tensor): The current input to the model.

        Returns:
            The output of the model.
        """
        # noinspection PyTypeChecker
        output = torch.minimum(torch.maximum(input_var.real, -self.limit), self.limit)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.limit = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.limit
