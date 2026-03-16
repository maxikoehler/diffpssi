"""
Main simulation class for power system simulation.

This module provides the main simulation classes used for dynamic and test-bench
simulations of power systems. It contains the :class:`PowerSystemSimulation`, a
simple :class:`Recorder`, and a :class:`TestBench` utility used for component
characterization and controller tests.

Typical usage::
    sim = PowerSystemSimulation(time_step=0.01, sim_time=10, parallel_sims=1, ...)
    sim.create_grid(grid_data)
    sim.set_record_function(my_recorder)
    time, data = sim.run()

The public classes expose helper methods to add buses, lines, transformers,
generators and control models. Heavy numerical work is delegated to model and
solver implementations in other modules.
"""

import logging
import os
import time

import numpy as np
from tqdm import tqdm

from .backend import *
from .load_flow import do_load_flow
from .models.exciters import SEXS
from .models.governors import TGOV1
from .models.stabilizers import STAB1
from .models.static_models import *
from .models.synchronous_machine import SynchMachine
from .models.transformer import transformer_type_dict
from .recorder import Recorder
from .solvers import solver_dict

_logger = logging.getLogger(__name__)


class PowerSystemSimulation(object):
    """
    Representation of a dynamic power system simulation.

    This class aggregates network elements (buses, lines, transformers), dynamic
    models (generators, exciters, governors, loads) and the numerical solver used
    to advance the simulation in time.

    Attributes:
        time (numpy.ndarray): Time vector used by the simulation.
        time_step (float or torch.Tensor): Simulation time step (may be batched).
        busses (list): List of Bus objects in the network.
        lines (list): List of Line objects in the network.
        trafos (list): List of transformer objects.
        base_mva (float): System base power in MVA.
        base_voltage (float): System nominal voltage.
        fn (float): Nominal system frequency in Hz.
        parallel_sims (int): Number of parallel simulations (batch size).
        solver (Solver): Numerical integrator/solver instance used for stepping.
        record_func (callable): Optional function used to collect outputs each step.

    The class provides convenience methods to construct the grid from a data
    structure, add elements and control models, initialize model states and run
    the time-domain simulation.
    """

    def __init__(
        self,
        time_step,
        sim_time,
        parallel_sims,
        solver="heun",
        trans_model="AM",
        backend="numpy",
        grid_data=None,
        jacobi_calculation=False,
        verbose=True,
    ):
        """
        Initialize a PowerSystemSimulation.

        For testing purposes the transformer modelling approach can be selected
        via ``trans_model`` (e.g. 'CIM', 'AM' or 'old').

        Args:
            time_step (float): Simulation time step (seconds).
            sim_time (float): Total simulation duration (seconds).
            parallel_sims (int): Number of parallel simulations (batch size).
            solver (str): Key for the numerical solver (used with solver_dict).
            trans_model (str): Transformer modelling approach ('CIM', 'AM', 'old').
            backend (str): Computational backend name (currently not directly used).
            grid_data (dict, optional): Optional grid specification to create the grid on init.
            jacobi_calculation (bool, optional): If True, compute jacobian during run.
            verbose (bool, optional): If True, enable progress output.

        Returns:
            None
        """
        # setting mode of the transformer model
        self.trans_model = trans_model

        # Add timestep in the end because the first step does not have a value
        self.time = np.arange(0, sim_time, time_step)
        self.time_step = time_step
        self.t = 0

        self.busses = []
        self.non_slack_busses = []
        self.bus_idxs = {}
        self.lines = []
        self.trafos = []

        self.inverse_dynamic_y_matrix = None
        self.static_y_matrix = None

        self.sc_events = []
        self.param_events = []
        self.parallel_sims = parallel_sims
        self.record_func = None
        self.verbose = verbose
        self.jacobian_calculation = jacobi_calculation
        self.jacobian_matrix = None

        if os.environ.get("DIFFPSSI_FORCE_INTEGRATOR") is not None:
            # this should only be used for integration tests
            self.solver = solver_dict[os.environ.get("DIFFPSSI_FORCE_INTEGRATOR")]()
            _logger.warning(
                "WARNING: FORCING THE USE OF THE {} INTEGRATOR. "
                "THIS SHOULD ONLY HAPPEN FOR UNITTESTS".format(
                    os.environ.get("DIFFPSSI_FORCE_INTEGRATOR")
                )
            )
        else:
            self.solver = solver_dict[solver]()

        self.base_voltage = None
        self.base_mva = None
        self.fn = None

        if grid_data is not None:
            self.create_grid(grid_data)

        self.backend = BACKEND

    def get_generator_by_name(self, name):
        """
        Return a generator model by name.

        Args:
            name (str): Generator model name to search for.

        Returns:
            object or None: The matching generator model, or None if not found.
        """
        for bus in self.busses:
            for model in bus.models:
                if model.name == name:
                    return model
        return None

    def create_grid(self, grid_data):
        """
        Create and populate the simulation grid from a data dictionary.

        The provided ``grid_data`` is transformed into internal structures and
        used to instantiate buses, generators, loads, lines, transformers and
        control models (AVR, governors, PSS). The method finally sets the slack bus.

        Args:
            grid_data (dict): Grid specification used to construct buses and models.

        Returns:
            None
        """
        self.fn = grid_data["f"]
        self.base_mva = grid_data["base_mva"]
        self.base_voltage = grid_data["base_voltage"]

        transformed_data = {}
        # first transform the data to a more convenient format
        for key, value in grid_data.items():
            # Check if the value is a list of lists
            if isinstance(value, list) and all(
                isinstance(item, list) for item in value
            ):
                # Use the first sublist as keys, and transform the remaining sublists into dictionaries
                keys = value[0]
                transformed_data[key] = [dict(zip(keys, v)) for v in value[1:]]
            elif isinstance(value, dict):
                # If the value is a dictionary, apply the transformation to each key within the dictionary
                transformed_data[key] = {
                    sub_key: [
                        dict(zip(value[sub_key][0], v)) for v in value[sub_key][1:]
                    ]
                    for sub_key in value
                }
            else:
                # Copy the value as is
                transformed_data[key] = value
        grid_data = transformed_data

        for bus_dict in grid_data["busses"]:
            bus_model = Bus(param_dict=bus_dict)
            bus_model.enable_parallel_simulation(self.parallel_sims)
            self.add_bus(bus_model)

        generators = grid_data.get("generators", [])
        if isinstance(generators, dict):
            for gen_dict in generators["GEN"]:
                generator_model = SynchMachine(
                    param_dict=gen_dict,
                    s_n_sys=self.base_mva,
                    v_n_sys=self.base_voltage,
                    f_n_sys=self.fn,
                )
                self.add_generator(generator_model)

        loads = grid_data.get("loads", [])
        if isinstance(loads, dict):
            for load_dict in loads["ZIP"]:
                load_model = Load(param_dict=load_dict, s_n_sys=self.base_mva)
                self.add_load(load_model)

        for line_dict in grid_data.get("lines", []):
            line_model = Line(
                param_dict=line_dict, s_n_sys=self.base_mva, v_n_sys=self.base_voltage
            )
            self.add_line(line_model)

        for transformer_dict in grid_data.get("transformers", []):
            trans_type = transformer_dict.get("type", "simple")
            transformer_model = transformer_type_dict[trans_type](
                sim=self,
                parallel_sims=self.parallel_sims,
                s_n_sys=self.base_mva,
                trans_model=self.trans_model,
                param_dict=transformer_dict,
            )
            self.add_transformer(transformer_model)

        exciters = grid_data.get("avr", [])
        if isinstance(exciters, dict):
            for sexs_dict in exciters["SEXS"]:
                exciter_model = SEXS(param_dict=sexs_dict)
                self.add_exciter(exciter_model)

        governors = grid_data.get("gov", [])
        if isinstance(governors, dict):
            for gov_dict in governors["TGOV1"]:
                gov_model = TGOV1(param_dict=gov_dict)
                self.add_governor(gov_model)

        psss = grid_data.get("pss", [])
        if isinstance(psss, dict):
            for pss_dict in psss["STAB1"]:
                pss_model = STAB1(param_dict=pss_dict)
                self.add_pss(pss_model)

        self.set_slack_bus(grid_data["slack_bus"])

    def set_slack_bus(self, slack_bus):
        """
        Set the slack/reference bus for the system.

        Args:
            slack_bus (str): Name of the bus to set as slack (must exist in bus_idxs).

        Returns:
            None
        """
        slack_bus_idx = self.bus_idxs[slack_bus]
        self.busses[slack_bus_idx].lf_type = "SL"

    def add_bus(self, bus_model):
        """
        Add a bus object to the simulation and enable batching if required.

        Args:
            bus_model (Bus): Bus instance to append to the simulation.

        Returns:
            None
        """
        # Assign an index to the bus and add it to the list of buses
        self.bus_idxs[bus_model.name] = len(self.busses)
        bus_model.enable_parallel_simulation(self.parallel_sims)
        self.busses.append(bus_model)

    def add_generator(self, generator_model):
        """
        Add a synchronous generator model to its configured bus.

        This will append the model to the bus, enable parallel simulation for
        the generator and set the bus type to PV. The bus voltage is adjusted
        to the generator's setpoint.

        Args:
            generator_model (SynchMachine): Generator instance with attribute .bus and .v_soll.

        Returns:
            None
        """
        bus = self.busses[self.bus_idxs[generator_model.bus]]
        generator_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(generator_model)
        # fit the voltage of the bus to the generator
        bus.update_voltages(bus.models[-1].v_soll)
        bus.lf_type = "PV"

    def add_inverter(self, inverter_model):
        """
        Add an inverter model to the specified bus and mark bus as PQ.

        Args:
            inverter_model (object): Inverter instance (must provide .bus attribute).

        Returns:
            None
        """
        bus = self.busses[self.bus_idxs[inverter_model.bus]]
        inverter_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(inverter_model)

        bus.lf_type = "PQ"

    def add_load(self, load_model):
        """
        Add a static or dynamic load model to its bus.

        Args:
            load_model (Load): Load instance with attribute .bus.

        Returns:
            None
        """
        bus = self.busses[self.bus_idxs[load_model.bus]]
        load_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(load_model)

    def add_shunt(self, shunt_model):
        """
        Add a shunt element to its bus.

        Args:
            shunt_model (Shunt): Shunt instance with attribute .bus.

        Returns:
            None
        """
        bus = self.busses[self.bus_idxs[shunt_model.bus]]
        shunt_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(shunt_model)

    def add_line(self, line_model):
        """
        Add a transmission line to the network and set internal bus indices.

        Args:
            line_model (Line): Line instance containing from_bus_name and to_bus_name.

        Returns:
            None
        """
        bus_from = self.bus_idxs[line_model.from_bus_name]
        bus_to = self.bus_idxs[line_model.to_bus_name]

        line_model.from_bus_id = bus_from
        line_model.to_bus_id = bus_to

        line_model.enable_parallel_simulation(self.parallel_sims)

        self.lines.append(line_model)

    def add_transformer(self, transformer_model):
        """
        Add a transformer to the network and set endpoint indices.

        Args:
            transformer_model (Transformer): Transformer instance with .from_bus_name and .to_bus_name.

        Returns:
            None
        """
        bus_from = self.bus_idxs[transformer_model.from_bus_name]
        bus_to = self.bus_idxs[transformer_model.to_bus_name]

        transformer_model.from_bus_id = bus_from
        transformer_model.to_bus_id = bus_to

        transformer_model.enable_parallel_simulation(self.parallel_sims)

        self.trafos.append(transformer_model)

    def add_exciter(self, exciter_model):
        """
        Attach an exciter model to its generator and set the voltage setpoint.

        Args:
            exciter_model (object): Exciter instance (must include attribute .gen).

        Returns:
            None
        """
        generator = self.get_generator_by_name(exciter_model.gen)
        exciter_model.v_setpoint = generator.v_soll
        exciter_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_exciter(exciter_model)

    def add_governor(self, governor_model):
        """
        Attach a governor model to its generator.

        Args:
            governor_model (object): Governor instance (must include attribute .gen).

        Returns:
            None
        """
        generator = self.get_generator_by_name(governor_model.gen)
        governor_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_governor(governor_model)

    def add_pss(self, pss_model):
        """
        Attach a power system stabilizer (PSS) to its generator.

        Args:
            pss_model (object): PSS instance (must include attribute .gen).

        Returns:
            None
        """
        generator = self.get_generator_by_name(pss_model.gen)
        pss_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_pss(pss_model)

    def add_voltage_controller(self, voltage_controller_model):
        """
        Add a voltage controller to the simulation.

        Args:
            voltage_controller_model (object): Controller instance to add.

        Returns:
            None
        """
        pass

    def inverse_dyn_admittance_matrix(self):
        """
        Compute and return the inverse dynamic admittance matrix.

        The admittance matrix is assembled from line, transformer and bus
        contributions; the function returns the inverse of the assembled
        dynamic bus admittance matrix for each parallel simulation.

        Returns:
            torch.Tensor: Batched inverse dynamic admittance matrix with complex dtype.
        """
        if True:
            # reconstruct the dynamic y_matrix
            # y_matrix shape: (parallel_sims, n_busses, n_busses)
            dynamic_y_matrix = torch.zeros(
                (self.parallel_sims, len(self.busses), len(self.busses)),
                dtype=torch.complex128,
            )
            for line in self.lines:
                dynamic_y_matrix[
                    :, line.from_bus_id, line.to_bus_id
                ] += line.get_admittance_off_diagonal()
                dynamic_y_matrix[
                    :, line.to_bus_id, line.from_bus_id
                ] += line.get_admittance_off_diagonal()
                dynamic_y_matrix[
                    :, line.from_bus_id, line.from_bus_id
                ] += line.get_admittance_diagonal()
                dynamic_y_matrix[
                    :, line.to_bus_id, line.to_bus_id
                ] += line.get_admittance_diagonal()

            # TRANSFORMER section
            for transformer in self.trafos:
                trafo_adm = transformer.calc_admittance(return_need=True)
                dynamic_y_matrix[
                    :, transformer.from_bus_id, transformer.to_bus_id
                ] += trafo_adm[:, 0, 1]
                dynamic_y_matrix[
                    :, transformer.to_bus_id, transformer.from_bus_id
                ] += trafo_adm[:, 1, 0]
                dynamic_y_matrix[
                    :, transformer.from_bus_id, transformer.from_bus_id
                ] += trafo_adm[:, 0, 0]
                dynamic_y_matrix[
                    :, transformer.to_bus_id, transformer.to_bus_id
                ] += trafo_adm[:, 1, 1]

            # BUS section
            for i, bus in enumerate(self.busses):
                for model in bus.models:
                    dynamic_y_matrix[:, i, i] += model.get_admittance(
                        dyn=True
                    ).squeeze()

            self.inverse_dynamic_y_matrix = torch.linalg.inv(dynamic_y_matrix)
            return self.inverse_dynamic_y_matrix

    def lf_admittance_matrix(self):
        """
        Compute and return the static (load-flow) admittance matrix.

        This function caches the static admittance matrix in ``self.static_y_matrix``
        to avoid repeated re-assembly.

        Returns:
            torch.Tensor: Batched static admittance matrix with complex dtype.
        """
        if self.static_y_matrix is not None:
            # get the previously computed static y_matrix
            return self.static_y_matrix
        # reconstruct the static y_matrix
        # y_matrix shape: (parallel_sims, n_busses, n_busses)
        static_y_matrix = torch.zeros(
            (self.parallel_sims, len(self.busses), len(self.busses)),
            dtype=torch.complex128,
        )
        for line in self.lines:
            static_y_matrix[
                :, line.from_bus_id, line.to_bus_id
            ] += line.get_admittance_off_diagonal()
            static_y_matrix[
                :, line.to_bus_id, line.from_bus_id
            ] += line.get_admittance_off_diagonal()
            static_y_matrix[
                :, line.from_bus_id, line.from_bus_id
            ] += line.get_admittance_diagonal()
            static_y_matrix[
                :, line.to_bus_id, line.to_bus_id
            ] += line.get_admittance_diagonal()

        for transformer in self.trafos:
            trafo_adm = transformer.calc_admittance_static(return_need=True)
            static_y_matrix[
                :, transformer.from_bus_id, transformer.to_bus_id
            ] += trafo_adm[:, 0, 1]
            static_y_matrix[
                :, transformer.to_bus_id, transformer.from_bus_id
            ] += trafo_adm[:, 1, 0]
            static_y_matrix[
                :, transformer.from_bus_id, transformer.from_bus_id
            ] += trafo_adm[:, 0, 0]
            static_y_matrix[
                :, transformer.to_bus_id, transformer.to_bus_id
            ] += trafo_adm[:, 1, 1]

        for i, bus in enumerate(self.busses):
            for model in bus.models:
                static_y_matrix[:, i, i] += model.get_admittance(dyn=False).squeeze()

        self.static_y_matrix = static_y_matrix

        return self.static_y_matrix

    def current_injections(self):
        """
        Return the current injections for all buses in the system.

        Each bus computes its own injection (via its models) and the results are
        stacked along the bus axis.

        Returns:
            torch.Tensor: Batched tensor containing current injections.
        """
        return torch.stack(
            [bus.get_current_injections() for bus in self.busses], axis=1
        )

    def initialize(self):
        """
        Initialize model states and compute initial algebraic variables.

        This runs a load-flow (via :func:`do_load_flow`) to determine initial
        power injections, calls ``initialize`` on models where provided and
        computes initial bus voltages.

        Returns:
            None
        """
        power_inj = do_load_flow(self)
        for i, bus in enumerate(self.busses):
            for model in bus.models:
                try:
                    model.initialize(power_inj[:, i], bus.voltage)
                except AttributeError:
                    raise AttributeError(
                        f"Model {model} at bus {bus} has no initialize method."
                    )

        for i, trafo in enumerate(self.trafos):
            try:
                trafo.initialize()
            except AttributeError:
                raise AttributeError("Transformer model has no initialize method.")

        # calculate bus voltages
        voltages = torch.matmul(
            self.inverse_dyn_admittance_matrix(), self.current_injections()
        )

        bus.update_voltages(voltages[:, i])

    def add_sc_event(self, start_time, end_time, bus):
        """
        Add a short-circuit event (temporary fault) affecting a bus.

        Args:
            start_time (float): Fault start time (seconds).
            end_time (float): Fault end time (seconds).
            bus (str): Name of the bus where the fault occurs.

        Returns:
            None
        """
        bus_idx = self.bus_idxs[bus]
        self.sc_events.append(ScEvent(start_time, end_time, bus_idx))

    def add_param_event(self, timestep, model, parameter, new_val):
        """
        Schedule a discrete parameter change at a given timestep.

        Args:
            timestep (float): Time at which the parameter is changed.
            model (object): Model instance whose attribute will be modified.
            parameter (str): Name of the attribute/parameter to change.
            new_val (Any): New value to assign to the parameter.

        Returns:
            None
        """
        param_event = ParamEvent(timestep, model, parameter, new_val)
        param_event.enable_parallel_simulation(self.parallel_sims)
        self.param_events.append(param_event)

    def add_param_dependency(self, timestep, model, parameter, function):
        """
        Schedule a time-dependent parameter update using a callable.

        Args:
            timestep (float): Time resolution or step used to evaluate the function.
            model (object): Model instance whose attribute will be updated.
            parameter (str): Name of the attribute to update.
            function (callable): Callable taking a time argument and returning the new value.

        Returns:
            None
        """
        param_event = ParamDependencyTime(timestep, model, parameter, function)
        param_event.enable_parallel_simulation(self.parallel_sims)
        self.param_events.append(param_event)

    def set_record_function(self, record_func):
        """
        Set a custom function used to record simulation data each timestep.

        Args:
            record_func (callable): Callable that accepts the simulation instance
                and returns a sequence of tensors/values to record.

        Returns:
            None
        """
        self.record_func = record_func

    def reset(self):
        """
        Reset the simulation to its initial state.

        This will call ``reset`` on all buses/models, reset the solver and clear
        cached admittance matrices.

        Returns:
            None
        """
        # reset all model states
        for bus in self.busses:
            bus.reset()

        self.solver.reset()
        self.static_y_matrix = None
        self.inverse_dynamic_y_matrix = None

    def run(self):
        """
        Run the time-domain simulation and collect recorded outputs.

        The routine initializes the system, steps through the configured time
        vector and calls the configured solver. If a ``record_func`` is set, its
        outputs are collected and returned as a tensor.

        Returns:
            tuple: ``(time_vector, recorded_tensor)`` where ``time_vector`` is a
            numpy array and ``recorded_tensor`` contains the recorded outputs.
        """
        self.initialize()

        start_time = time.time()

        recorder_list = []

        if self.verbose:
            iterator = tqdm(self.time)
        else:
            iterator = self.time

        for t in iterator:
            if self.jacobian_calculation:
                self.jacobian_matrix = torch.linalg.det(
                    self.construct_jacobian_matrix()
                ) * torch.ones((self.parallel_sims, 1), dtype=torch.float64)

            # Parameter event handler has to be BEFORE y_matrix calculation
            # -> Respect changes in models in the y_matrix
            for param_event in self.param_events:
                param_event.handle_event(t)

            # TRANSFORMER MODEL differences
            dynamic_y_matrix = torch.linalg.inv(self.inverse_dyn_admittance_matrix())

            # SC event maipulations have to be done AFTER the calculation of new y_matrix
            for sc_event in self.sc_events:
                if sc_event.is_active(t):
                    dynamic_y_matrix[:, sc_event.bus, sc_event.bus] = 1e6

            self.inverse_dynamic_y_matrix = torch.linalg.inv(dynamic_y_matrix)

            # do a step with the solver
            self.solver.step(self)

            # Record the state of the system; all desired parameters
            try:
                recorder_list.append(torch.stack(self.record_func(self)))
            except TypeError as e:
                _logger.warning("No record function specified: %s", e)

            self.t += self.time_step

        # Format shall be [batch, timestep, value]
        try:
            return_tensor = torch.swapaxes(
                torch.stack(recorder_list, axis=1), 0, 2
            ).squeeze(-1)
        except:
            return_tensor = torch.stack(recorder_list, axis=1)

        if self.verbose:
            end_time = time.time()
            print("=" * 50)
            print(
                "Dynamic simulation finished in {:.2f} seconds".format(
                    end_time - start_time
                )
            )

        return self.time, return_tensor
