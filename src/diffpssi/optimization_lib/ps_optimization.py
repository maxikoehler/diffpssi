"""This file contains the optimization procedure of the power system parameters."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
import os
import time

import torch
from matplotlib import pyplot as plt

from diffpssi.optimization_lib.optimizers import CustomBFGSREALOptimizer

# currently only bfgs is supported, as it works best by far
optimizer_dict = {"bfgs": CustomBFGSREALOptimizer}


class PowerSystemOptimization:
    """Optimize the parameters of a power system simulation."""

    def __init__(
        self,
        sim,
        original_data,
        params_optimizable,
        param_names=None,
        optimizer="bfgs",
        params_original=None,
        max_step=0.1,
        decay=0.9,
        enable_plots=False,
        normalize_loss=True,
        loss_function=None,
        loss_threshold=None,
    ):
        """
        Initialize the PowerSystemOptimization class.

        Args:
            sim: PowerSystemSimulation object. Used to execute on a simulation object.
            original_data (torch.Tensor): Original data of shape (batch-size, timesteps, features).
            params_optimizable (list): List of parameters to optimize.
            param_names (list, optional): List of parameter names for display.
            optimizer (str, optional): Optimizer to use ('bfgs' supported).
            params_original (list, optional): List of original parameters for debugging.
            max_step (float or list, optional): Maximum relative step size for optimizer.
            decay (float, optional): Decay factor for the maximum step size (0-1).
            enable_plots (bool, optional): If True, plot results after each optimization step.
            normalize_loss (bool, optional): If True, normalize loss by min/max of original data.
            loss_function (callable, optional): Custom loss function. If None, use default.
            loss_threshold (float, optional): Stop optimization if loss falls below this value.

        Returns:
            None
        """
        self.sim = sim

        if sim.backend == "numpy":
            raise NotImplementedError(
                "Optimization is only supported for the PyTorch backend, not numpy. "
                "Please set the backend to PyTorch in power_sim_lib/backend.py"
            )

        self.target_data = original_data
        self.optimizer = optimizer_dict[optimizer](
            params_optimizable, max_step=max_step, decay=decay
        )

        self.params_original = params_original

        self.last_min_loss = None
        self.loss_threshold = loss_threshold if loss_threshold else 0
        if param_names:
            self.param_names = param_names
        else:
            self.param_names = [f"Param {i}" for i in range(len(params_optimizable))]

        self.enable_plots = enable_plots

        self.normalize_loss = normalize_loss

        if loss_function:
            self.loss_function = loss_function
        else:
            # use a default loss function:
            def default_loss_function(sim_result, target_data):
                """
                Calculate the mean absolute error between the simulation result and the target data.

                Args:
                    sim_result: The simulation result of the size (batch-size, timesteps, features)
                    target_data: The target data of the size (batch-size, timesteps, features)

                Returns:
                    A vector of the mean absolute error
                    for each batch element of the size (batch-size).
                """
                return torch.mean(
                    torch.sum(torch.abs(target_data - sim_result), dim=2), axis=1
                )

            self.loss_function = default_loss_function

        if self.normalize_loss:
            # normalize data
            # Determine the minimum and maximum values along dimension 1 of the target data
            self.min_values = torch.min(self.target_data, dim=1)[0].unsqueeze(1)
            self.max_values = torch.max(self.target_data, dim=1)[0].unsqueeze(1)
            self.range_values = self.max_values - self.min_values
            if torch.any(self.range_values == 0):
                # if this does not work, use simulation data to normalize.
                # Note: This is risky and can lead to errors.
                print("Using simulation data to normalize loss function")
            else:
                print("Using target data to normalize loss function")

        # save plot settings for easier comparison
        self.x_lims = None
        self.y_lims = None

    def plot_state(self, t, results, original_data, opt_step):
        """
        Plot the current best simulation result.

        Args:
            t (array-like): Timesteps.
            results (array-like): Simulation results.
            original_data (array-like): Original data.
            opt_step (int): Current optimization step.
        """
        # create as many subplots as we have signals to compare
        plt.figure()

        # set figure size
        plt.gcf().set_size_inches(10, 10)

        for i in range(len(results[0])):
            plt.subplot(len(results[0]), 1, i + 1)
            plt.plot(t, original_data[:, i], label="Original")
            plt.plot(t, results[:, i], label="Simulated", linestyle="--")

            if self.x_lims and self.y_lims:
                plt.xlim(self.x_lims[i])
                plt.ylim(self.y_lims[i])

            plt.ylabel(f"Signal {i}")
            plt.xlabel("Time [s]")

        plt.legend()
        plot_file = os.path.join(
            os.getcwd(), f"data/plots/optimization_step_{opt_step}.png"
        )
        plt.savefig(plot_file)

        # get xlims and ylims of all subplots
        if not self.x_lims or not self.y_lims:
            self.x_lims = []
            self.y_lims = []
            for ax in plt.gcf().axes:
                self.x_lims.append(ax.get_xlim())
                self.y_lims.append(ax.get_ylim())

        plt.close()

    def run(self, max_steps=100):
        """
        Run the configured optimization procedure.

        Args:
            max_steps (int, optional): Maximum number of optimization steps to perform.

        Returns:
            None
        """
        if os.environ.get("DIFFPSSI_FORCE_OPT_ITERS") is not None:
            max_steps = int(os.environ.get("DIFFPSSI_FORCE_OPT_ITERS"))
            print(
                f"WARNING: FORCING THE USE OF {os.environ.get('DIFFPSSI_FORCE_OPT_ITERS')}"
                "OPTIMIZATION ITERATION."
                "THIS SHOULD ONLY HAPPEN FOR UNITTESTS"
            )
        opt_start_time = time.time()

        min_loss_idx = None  # the index of the current best batch element
        results = None  # the simulation results
        t = None  # the timesteps
        opt_step = None  # the current optimization step

        for opt_step in range(max_steps):
            # opt_step_start = time.time()
            # set the gradients to zero in order to accumulate the
            self.optimizer.zero_grad()

            # first execute simulation with current parameters
            t, results = self.sim.run()

            if self.normalize_loss:
                if torch.any(self.range_values == 0):
                    min_values = torch.min(results, dim=1)[0].unsqueeze(1)
                    max_values = torch.max(results, dim=1)[0].unsqueeze(1)

                    # take the median of the min and max values along axis 0 to avoid outliers
                    # Also detach the values from the graph, because we will use them for scaling in
                    # later episodes as well
                    self.min_values = (
                        torch.median(min_values, dim=0)[0].unsqueeze(0).detach()
                    )
                    self.max_values = (
                        torch.median(max_values, dim=0)[0].unsqueeze(0).detach()
                    )
                    self.range_values = self.max_values - self.min_values

                target_norm = (self.target_data - self.min_values) / self.range_values
                res_norm = (results - self.min_values) / self.range_values
            else:
                target_norm = self.target_data
                res_norm = results

            # then calculate the loss, which corresponds to the mean absolute error
            # For this purpose the sum of all analyzed signals is calculated
            # and the mean along the time axis is taken
            # The result is a vector of the size (batch-size)
            loss = self.loss_function(res_norm, target_norm)

            # take the minimum loss for further analysis
            min_loss_val, min_loss_idx = torch.nan_to_num(loss, 100000).min(dim=0)

            # print the minimum loss and the corresponding idx
            print(
                f"Step: {opt_step}, Min. Loss Batch: {int(min_loss_idx)},"
                f" Min. Loss: {float(min_loss_val)}"
            )

            # calculate the gradients for the loss
            loss.sum().backward()

            # print the current best batch of parameters by comprehending them in a list
            print_list = [
                p[min_loss_idx].detach()
                for p in self.optimizer.param_groups[0]["params"]
            ]

            params_str = ", ".join(
                f"{self.param_names[i]}: {float(print_list[i].data.real):.3f}"
                for i in range(len(print_list))
            )
            print(f"Current Best Params: {params_str}")

            if self.params_original is not None:
                rel_errs = [
                    (float(print_list[i].data.real) - self.params_original[i])
                    / self.params_original[i]
                    * 100
                    for i in range(len(print_list))
                ]
                rel_errs_str = ", ".join(
                    f"{self.param_names[i]}: {rel_errs[i]:.2f}%"
                    for i in range(len(print_list))
                )
                print(f"Relative Errors in Percent: {rel_errs_str}")

            print(
                "---------------------------------------------------------------------------------"
            )

            if min_loss_val < self.loss_threshold:
                print("Loss threshold reached. Optimization stopped.")
                break

            # perform the optimization step in order to adapt the parameters using the gradients
            self.optimizer.step()

            if self.enable_plots:
                plt_original_data = self.target_data[min_loss_idx].detach().numpy()
                plt_results = results[min_loss_idx].detach().numpy()
                self.plot_state(t, plt_results, plt_original_data, opt_step)

            self.sim.reset()

            if self.last_min_loss and self.last_min_loss < min_loss_val:
                self.optimizer.decrease_step_size()

            self.last_min_loss = min_loss_val

        print(f"Optimization finished in {time.time() - opt_start_time:.2f} seconds")
        plt_original_data = self.target_data[min_loss_idx].detach().numpy()
        plt_results = results[min_loss_idx].detach().numpy()
        self.plot_state(t, plt_results, plt_original_data, opt_step)
