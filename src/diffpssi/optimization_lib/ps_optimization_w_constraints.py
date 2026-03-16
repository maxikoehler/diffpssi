"""Do optimization with constraints for power systems."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
import os
import time

import torch
from matplotlib import pyplot as plt

from diffpssi.optimization_lib.ps_optimization import PowerSystemOptimization


class PSOwithConstraints(PowerSystemOptimization):
    """Power System Optimization with Constraints.

    This class extends the PowerSystemOptimization class to include
    constraint handling during the optimization process.
    """

    def __init__(
        self,
        simulation,
        recorder_target,
        optimizer,
        loss_function,
        constraint_functions,
        constraint_penalties,
        max_iterations=100,
        tol=1e-6,
        save_path="./optimization_results",
        verbose=True,
    ):
        """
        Initialize the PSOwithConstraints class.

        Args:
            simulation: The power system simulation object.
            recorder_target: Target data for the recorder.
            optimizer: The optimizer to use for parameter updates.
            loss_function: The loss function to minimize.
            constraint_functions: List of functions representing constraints.
            constraint_penalties: List of penalties for each constraint.
            max_iterations (int, optional): Maximum number of optimization iterations.
            tol (float, optional): Tolerance for convergence.
            save_path (str, optional): Path to save optimization results.
            verbose (bool, optional): If True, print progress information.
        """
        super().__init__(
            simulation,
            recorder_target,
            optimizer,
            loss_function,
            max_iterations,
            tol,
            save_path,
            verbose,
        )
        self.constraint_functions = constraint_functions
        self.constraint_penalties = constraint_penalties

    def compute_objective(self, recorder):
        """
        Compute the total loss including constraints.

        Args:
            recorder: The recorded simulation data.

        Returns:
            torch.Tensor: The total loss value.
        """
        base_loss = self.loss_function(recorder, self.recorder_target)
        constraint_loss = 0.0

        for func, penalty in zip(self.constraint_functions, self.constraint_penalties):
            constraint_value = func(self.simulation)
            constraint_loss += penalty * torch.relu(constraint_value).pow(2).mean()

        total_loss = base_loss + constraint_loss
        return total_loss

    def plot_state(self, t, results, original_data, opt_step):
        """
        Plot the current state of the optimization.

        Args:
            t (int): Current iteration number.
            results: The recorded simulation data.
            loss: Current loss value.

        Returns:
            None
        """
        super().plot_state(
            t=t, results=results, original_data=original_data, opt_step=opt_step
        )

    def run(self, max_steps=100):
        """
        Run the optimization process with constraints.

        Args:
            max_steps (int, optional): Maximum number of optimization steps.

        Returns:
            None
        """
        if self.verbose:
            print("Starting optimization with constraints...")

        super().run(max_steps=max_steps)
