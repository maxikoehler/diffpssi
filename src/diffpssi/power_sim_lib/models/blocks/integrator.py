"""Models an Integrator block."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks.blocks_interface import Block


class Integrator(Block):
    """Represent an Integrator block in power system simulations.

    This class models an integrator, which accumulates the input signal over time. It is a fundamental
    component in control systems for integrating signals.

    Attributes:
        input (float or torch.Tensor): Current input to the Integrator block.
        state_1 (float or torch.Tensor): Internal state of the Integrator block.
    """

    def __init__(self, k_i, limiter=None):
        """Initialize the I controller with specified parameters.

        Args:
            k_i (float): Integral gain of the PI controller.
            lim_min (float): Minimum limit for the output.
            lim_max (float): Maximum limit for the output.
        """
        self.k_i = k_i

        self.input = 0
        self.state_1 = 0

        self.limiter = limiter

    def differential(self):
        """Compute the differential equations for the PI controller.

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
        """Retrieve the current state vector of the PI controller.

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
        """Set the state vector of the I controller.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """Compute the output of the PI controller.

        Args:
            input_var: The current input to the model.

        Returns: The output of the model.
        """
        if self.input.shape != input_var.shape:
            raise ValueError(
                f"Input shape mismatch in Integrator block: input_var: {input_var.shape} vs. {self.input.shape}"
            )

        self.input = self.k_i * input_var
        output = self.state_1

        if self.limiter is not None and self.state_1 >= self.limiter:
            self.state_1 = torch.zeros(self.state_1.shape)

        return output

    def initialize(self, out_wish):
        """Initialize the PI controller with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.
        """
        self.state_1 = out_wish / self.k_i
        return self.state_1

    def enable_parallel_simulation(self, parallel_sims):
        """Enable parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.k_i = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_i
        self.input = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.input
        self.state_1 = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.state_1
        )

    def reset(self):
        """Reset the state vector of the I controller to zero."""
        self.state_1 = torch.zeros(self.state_1.shape)
        self.input = torch.zeros(self.input.shape)
