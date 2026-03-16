"""Implements a testbench for power system simulations.

Especially useful for testing models e.g., controller
blocks in isolation.
"""

import logging
import time

import numpy as np
from tqdm import tqdm

from .backend import *
from .recorder import Recorder
from .tb_solvers import tb_solver_dict

_logger = logging.getLogger(__name__)


class TestBench(object):
    """
    Lightweight testbench for running and characterizing control elements.

    The TestBench runs a collection of differential models with a provided
    input function and collects outputs via a recorder function. It is mainly
    used for controller or component inspection.

    Attributes:
        time (numpy.ndarray): Time vector for the test.
        time_step (float): Time step (seconds).
        diff_models (list): List of differential model instances.
        input_func (callable): Function returning input values for each time.
        record_func (callable): Recorder function used to collect outputs.
        solver (Solver): Time integration solver used to step the models.
        recorder (Recorder): Recorder instance configured for this testbench.
    """

    def __init__(
        self,
        time_step,
        sim_time,
        inspection_models,
        input_func,
        record_func,
        default_runner="run_oltc_control",
        solver="euler",
        verbose=True,
        parallel_sims=1,
    ):
        """
        Initialize the TestBench.

        Args:
            time_step (float): Simulation time step (seconds).
            sim_time (float): Total duration of the test (seconds).
            inspection_models (list): List of tuples (model, init_dict) describing models to inspect.
            input_func (callable): Function taking time and returning input to models.
            record_func (callable): Recorder function used to collect outputs.
            solver (str, optional): Solver key to select time integrator. Defaults to 'euler'.
            verbose (bool, optional): If True, enable progress output.

        Returns:
            None
        """
        self.sim_time = sim_time
        self.time_step = time_step
        self.time = np.arange(0, self.sim_time + self.time_step, self.time_step)

        self.diff_models = []
        self.init_dict = []
        for i in range(len(inspection_models)):
            self.diff_models.append(inspection_models[i][0])
            self.init_dict.append(inspection_models[i][1])

        self.input_func = input_func
        self.model_return = torch.zeros(
            (len(self.time), len(self.diff_models), parallel_sims, 1),
            dtype=torch.float64,
            # requires_grad=True,
        )

        self.solver = tb_solver_dict[solver]()

        self.parallel_sims = parallel_sims
        self.verbose = verbose
        self.backend = BACKEND

        # Flag to track if parallel simulation has been enabled (to avoid re-expansion)
        self._parallel_sim_enabled = False

        run_functions = {
            "run_oltc_control": self.run_oltc_control,
            "run_control_element": self.run_control_element,
        }
        if default_runner in run_functions:
            self.default_runner = run_functions[default_runner]
        else:
            msg = f"{default_runner} is not a valid runner function."
            raise TypeError(msg)

        self.recorder = Recorder(sim=self, recorder_dict=record_func)
        self.record_func = self.recorder.record_fun

    def run(self):
        """Wrap the desired run function for mathod compatibility.

        Returns:
            np.array: Simulation time stamps, Tensor: Result Tensor in the
            form [batch, model, timestep].
        """
        _, result = self.default_runner()
        return self.time, result

    def initialize(self):
        """
        Initialize all differential models and enable batching.

        Returns:
            None
        """
        if not self._parallel_sim_enabled:
            for i, model in enumerate(self.diff_models):
                model.initialize(self.init_dict[i])
            self.enable_parallel_simulation(self.parallel_sims)
            self._parallel_sim_enabled = True

    def record_list(self):
        """
        Return the list of recorded data descriptions from the recorder.

        Returns:
            list: Descriptions of recorded fields.
        """
        return self.recorder.record_list()

    def set_record_func(self, record_func):
        """
        Set a custom recorder function for the TestBench.

        Args:
            record_func (callable): Recorder function compatible with Recorder.

        Returns:
            None
        """
        self.recorder.set_record_func(record_func)
        self.record_func = record_func
        return

    def run_control_element(self):
        """
        Run the testbench for control element inspection.

        The testbench repeatedly calls ``input_func`` and updates the
        differential models. Outputs are collected by the configured recorder.

        Returns:
            tuple: (time_vector, recorded_tensor)
        """
        self.initialize()

        start_time = time.time()

        if self.verbose:
            iterator = tqdm(self.time)
        else:
            iterator = self.time

        # format should be [model, timestep, value]
        recorder_list = []

        for t in iterator:

            input_val = self.input_func(t)

            for model in self.diff_models:
                model.get_output(input_val)

            self.solver.step(self)

            # Record the state of the system; all desired parameters
            try:
                recorder_list.append(torch.stack(self.record_func(self)))
            except TypeError as e:
                _logger.warning("No record function specified: %s", e)

        # Format shall be [batch, timestep, value]
        try:
            return_tensor = torch.swapaxes(
                torch.stack(recorder_list, axis=1), 0, 2
            ).squeeze(-1)
        except:
            return_tensor = torch.stack(recorder_list, axis=1)

        if self.verbose:
            end_time = time.time()
            print(
                "Dynamic simulation finished in {:.2f} seconds".format(
                    end_time - start_time
                )
            )

        return self.time, return_tensor

    def run_oltc_control(self):
        """
        Run the testbench with OLTC-style control elements.

        Similar to ``run_control_element`` but supports chained/feedback inputs
        between models. Returns the recorded outputs.

        Returns:
            tuple: (time_vector, recorded_tensor)
        """
        self.initialize()

        start_time = time.time()

        if self.verbose:
            iterator = tqdm(self.time)
        else:
            iterator = self.time

        # format should be [model, timestep, value]
        recorder_list = []

        for j, t in enumerate(iterator):

            for i, model in enumerate(self.diff_models):
                # [timestep, model]
                input_val = self.input_func(t) * self.model_return[j, i]
                # model.update_vref(input_val)
                model_output = model.get_output(input_val)
                # Detach to prevent building up of computational graph across iterations
                # This prevents "backward through graph a second time" errors
                if t < self.time[-1]:
                    self.model_return[j + 1, i] = (
                        model_output.detach()
                        if hasattr(model_output, "detach")
                        else model_output
                    )

            self.solver.step(self)

            # Record the state of the system; all desired parameters
            try:
                recorder_list.append(torch.stack(self.record_func(self)))
            except TypeError as e:
                _logger.warning("No record function specified: %s", e)

        # Format shall be [batch, timestep, value]
        try:
            return_tensor = torch.swapaxes(
                torch.stack(recorder_list, axis=1), 0, 2
            ).squeeze(-1)
        except:
            return_tensor = torch.stack(recorder_list, axis=1)

        if self.verbose:
            end_time = time.time()
            print(
                "Dynamic simulation finished in {:.2f} seconds".format(
                    end_time - start_time
                )
            )

        return self.time, return_tensor

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel (batched) simulation for all contained models.

        Args:
            parallel_sims (int): Number of parallel simulations to run (batch size).

        Returns:
            None
        """
        for model in self.diff_models:
            model.enable_parallel_simulation(parallel_sims)

        self.time_step = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.time_step
        )

    def reset(self):
        """
        Reset the TestBench to its initial state.

        This will call ``reset`` on all models, reset the solver and clear
        cached admittance matrices.

        Returns:
            None
        """
        self.time_step = self.time_step * torch.ones(
            (self.parallel_sims, 1), dtype=torch.float64
        )
        self.model_return = torch.zeros_like(self.model_return, dtype=torch.float64)

        # reset all model states
        for model in self.diff_models:
            model.reset()

        self.solver.reset()
