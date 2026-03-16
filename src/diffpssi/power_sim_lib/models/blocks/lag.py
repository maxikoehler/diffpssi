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
        dx1 = -self.kd * (self.state_1 - 1) + self.ki * self.input
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
        # target_shape = (parallel_sims, 1)

        # def expand_param_inplace(param, needs_grad=False):
        #     if hasattr(param, 'shape') and param.shape == target_shape:
        #         return param
        #     val = param.detach() if hasattr(param, 'detach') else param
        #     if needs_grad:
        #         expanded = val.expand(target_shape).clone()
        #         expanded.requires_grad_(True)
        #         return expanded
        #     else:
        #         return torch.ones(target_shape, dtype=torch.float64) * val

        # ki_needs_grad = hasattr(self.ki, 'requires_grad') and self.ki.requires_grad
        # kd_needs_grad = hasattr(self.kd, 'requires_grad') and self.kd.requires_grad

        # if not (hasattr(self.input, 'shape') and self.input.shape == target_shape):
        #     self.input = expand_param_inplace(self.input, False)
        # if not (hasattr(self.state_1, 'shape') and self.state_1.shape == target_shape):
        #     self.state_1 = expand_param_inplace(self.state_1, False)
        # if not (hasattr(self.ki, 'shape') and self.ki.shape == target_shape):
        #     self.ki = expand_param_inplace(self.ki, ki_needs_grad)
        # if not (hasattr(self.kd, 'shape') and self.kd.shape == target_shape):
        #     self.kd = expand_param_inplace(self.kd, kd_needs_grad)
        # if not (hasattr(self.lim_min, 'shape') and self.lim_min.shape == target_shape):
        #     self.lim_min = expand_param_inplace(self.lim_min, False)
        # if not (hasattr(self.lim_max, 'shape') and self.lim_max.shape == target_shape):
        #     self.lim_max = expand_param_inplace(self.lim_max, False)

        self.ki = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.ki
        self.kd = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.kd
        self.lim_min = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_min
        )
        self.lim_max = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_max
        )
        self.input = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.input
        self.state_1 = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.state_1
        )

    def initialize(self, out_wish):
        """Initialize the PT1Limited model with a specified output.

        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.
        """
        self.out_wish = out_wish
        self.state_1 = out_wish
        return out_wish

    def reset(self):
        """Reset the internal state of the PT1Limited model."""
        self.input = torch.zeros_like(self.input)
        self.state_1 = self.out_wish * torch.ones_like(self.state_1)
