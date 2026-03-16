"""Models a Deadband block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class DeadBand(object):
    """Represent a Dead Band element in power system simulations.

    This class models a dead band element, which outputs a function if the input exceeds
    a certain threshold. It is useful in control systems to ignore small deviations and
    only respond to significant changes.

    Attributes:
        threshold (float or torch.Tensor): The threshold value for the dead band.
        output_function (callable): The function to apply to the input after exceeding the threshold.
    """

    def __init__(self, threshold):
        """Initialize the DeadBand block with a specified threshold and output function.

        Args:
            threshold (float): The threshold value for the dead band.
            output_function (callable): The function to apply to the input after exceeding the threshold.
        """
        self.threshold = threshold

    def get_output(self, input_var):
        """Compute the output of the DeadBand block.

        Args:
            input_var (torch.Tensor): The current input to the model.
        Returns: The output of the model.
        """
        return torch.where(
            torch.abs(input_var) >= self.threshold,
            input_var,
            torch.zeros_like(input_var),
        )

    def enable_parallel_simulation(self, parallel_sims):
        """Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.threshold = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.threshold
        )
