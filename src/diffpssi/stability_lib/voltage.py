"""Methods and tools for voltage stability analysis.

Some of the methods and features are highly compatible with generic data from a simulation program.
Other are highly specific to the DiffPSSI simulation framework and its syntax.
"""

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.load_flow import do_load_flow
from diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss
from diffpssi.tools import *

# pylint: disable=too-few-public-methods, too-many-locals


class VoltageIndex(ABC):
    """Abstract Base Class for all Voltage Stability Indices."""

    def __init__(self):
        """Initialize the VoltageIndex base class."""


# Trajectory Violation Integral: Violation of Envelopes around a certain Bus
class ViolationIntegral(VoltageIndex):
    """Calculate the 'Trajectory Violation Integral (TVI)' based on existing time series data.

    This represents an envelope around a defined stable convergent, and analyzes the area of
    exceeding this border. Another formulation would be the difference between the envelope
    and the time series voltage outside that envelope times the violated time.

    Attributes:
        env (str): Type of the desired envelope (defaults to "TVI").
        env_params (dict): Variable settings for the envelope 'TVI'. Can be reset with
            :meth:`set_env_params`.
        result (complex|None): Storage for the computed result (initially ``None``).
    """

    def __init__(self, envelope: str = "TVI"):
        """Initialize a ViolationIntegral calculation object.

        Args:
            envelope (str): Used envelope for defining stable dynamic voltage behavior.
                Accessible values:
                  - 'FRT': Medium Voltage Fault-Ride Through curves for type II machines.
                  - 'TVI': Scientific exponential decaying approach. Defaults to 'TVI'.
        """
        self.env = envelope
        self.env_params = {"beta": 0.05, "v_st": 0.9, "t_start": 1}
        self.result = None

    def get_result(self, time: list, v_bb: list) -> complex:
        """Calculate the integral of the voltage differences.

        The method selects the envelope type set in the instance and delegates to the
        corresponding envelope functions to compute the violation integral.

        Args:
            time (list): Time vector of the time series computation result.
            v_bb (list): Bus voltage vector of the time series computation.

        Returns:
            complex: The integral of the bus voltage violation over time.

        Raises:
            TypeError: If an unsupported envelope type is configured.
        """
        if self.env == "TVI":
            self.result = self.__calc_env(time, v_bb, self.t_upp, self.t_low)
        elif self.env == "FRT":
            self.result = self.__calc_env(time, v_bb, self.hvrt, self.lvrt)
        else:
            raise TypeError(f"{self.env} is not supported as envelope for the index.")

        return self.result

    def get_env(self, time: list) -> list[list]:
        """Return discrete envelope values for the provided time vector.

        Args:
            time (list): Time vector of the time series computation result.

        Returns:
            list[list]: A time-dependent voltage vector containing the upper and lower
                envelope as [upper, lower].

        Raises:
            TypeError: If an unsupported envelope type is configured.
        """
        if self.env == "TVI":
            upper = self.t_upp(time)
            lower = self.t_low(time)
        elif self.env == "FRT":
            upper = self.hvrt(time)
            lower = self.lvrt(time)
        else:
            raise TypeError(f"{self.env} is not supported as envelope for the index.")

        return [upper, lower]

    def set_env_params(self, env_params: dict) -> None:
        """Set the envelope parameters.

        Args:
            env_params (dict): New envelope parameters dictionary.
        """
        self.env_params = env_params

    def __calc_env(
        self, time: list, v_bb: list, func_up: callable, func_low: callable
    ) -> complex:
        """Calculate the envelope violation integral using a Riemann sum.

        Args:
            time (list): Time vector of the time series computation.
            v_bb (list): Bus voltage result vector from the time series computation.
            func_up (callable): Callable returning the upper voltage envelope for a
                given time vector.
            func_low (callable): Callable returning the lower voltage envelope for a
                given time vector.

        Returns:
            complex: Riemann sum of the voltage differences (area outside the envelope).
        """
        sum_v_diffs = 0
        delta_t = time[1] - time[0]
        up = func_up(time)
        low = func_low(time)
        for i in range(len(time)):
            if v_bb[i] < low[i]:
                sum_v_diffs += (low[i] - v_bb[i]) * delta_t
            elif v_bb[i] > up[i]:
                sum_v_diffs += (v_bb[i] - up[i]) * delta_t
            else:
                sum_v_diffs += 0

        return sum_v_diffs

    def hvrt(self, time: list) -> list:
        """Calculate the upper envelope for Fault-Ride-Through (FRT) behavior.

        The envelope follows parameters from the Technical Connection Guidelines in
        Europe for Type II generation units in medium voltage grids.

        Args:
            time (list): Target time vector, usually the time series computation one.

        Returns:
            list: Voltage vector for the upper FRT envelope.
        """
        t_start = self.env_params["t_start"]
        u_c = self.env_params["v_st"]

        u_0 = 1.25
        t_1 = 0.1
        u_1 = 1.2
        t_2 = 5
        u_2 = 1.15
        t_3 = 60
        u_3 = 1.1
        output = []

        for t in time:
            if t - t_start <= 0:
                output.append(u_0 * u_c)
            elif t - t_start > 0 and t - t_start <= t_1:
                output.append(u_1 * u_c)
            elif t - t_start > t_1 and t - t_start <= t_2:
                output.append(u_2 * u_c)
            elif t - t_start > t_2 and t - t_start <= t_3:
                output.append(u_2 * u_c)
            elif t - t_start > t_3:
                output.append(u_3 * u_c)

        return output

    def lvrt(self, time: list, t_start: float = 1, u_c: complex = 1) -> list:
        """Calculate the lower envelope for Fault-Ride-Through (FRT) behavior.

        The envelope follows parameters from the Technical Connection Guidelines in
        Europe for Type II generation units in medium voltage grids.

        Args:
            time (list): Time vector of the time series computation.
            t_start (float, optional): Start time of the fault scenario. Defaults to 1.
            u_c (complex, optional): Characteristic voltage reference (defaults to 1).

        Returns:
            list: Voltage vector for the lower FRT envelope.
        """
        t_start = self.env_params["t_start"]
        u_c = self.env_params["v_st"]

        u_0 = 0.15
        t_1 = 0.15

        def u_1(t):
            return (14 / 57) * t + (43 / 380)

        t_2 = 3
        u_2 = 0.85
        t_3 = 60
        u_3 = 0.9
        output = []

        for t in time:
            if t - t_start <= t_1:
                output.append(u_0 * u_c)
            elif t - t_start > t_1 and t - t_start <= t_2:
                output.append(u_1(t - t_start) * u_c)
            elif t - t_start > t_2 and t - t_start <= t_3:
                output.append(u_2 * u_c)
            elif t - t_start > t_3:
                output.append(u_3 * u_c)

        return output

    def t_low(self, time: list) -> list:
        """Calculate the lower envelope using the scientific (exponential) description.

        Args:
            time (list): Time vector of the time series computation.

        Returns:
            list: Voltage vector for the scientific lower voltage envelope.
        """
        t_low = []
        t_end = time[-1]

        for t in time:
            if t < self.env_params["t_start"]:
                t_low.append(np.zeros_like(t))
            else:
                t_low.append(
                    (
                        (
                            (t - self.env_params["t_start"])
                            / t_end
                            * np.exp((t - self.env_params["t_start"]) / t_end)
                        )
                        ** self.env_params["beta"]
                    )
                    / np.exp(self.env_params["beta"])
                    * self.env_params["v_st"]
                )

        return t_low

    def t_upp(self, time: list) -> list:
        """Calculate the upper envelope using the scientific (exponential) description.

        Args:
            time (list): Time vector of the time series computation.

        Returns:
            list: Voltage vector for the scientific upper voltage envelope.
        """
        return 2 * np.ones_like(time) - self.t_low(time)


# Nose Curve Calculation
class NoseCurve:
    """Nose Curve calculation based on iterative load flows.

    The class runs load flow calculations with varying load parameters on a provided grid.
    It can also be used to compute nose curves for a grid with a given parameter set.

    Attributes:
        ps_sim (Pss): Referenced PowerSystemSimulation object used for load flows.
        result (dict): Dictionary storing results per bus as pandas DataFrames.
        res_variation (dict): Storage for variation results.
    """

    def __init__(
        self,
        load_model: callable,
        loading: dict[list],
        ps_sim: Pss = None,
    ):
        """Initialize a NoseCurve calculation and plotting object.

        Args:
            load_model (callable): Callable for the load model used to create the grid.
                The callable must accept kwargs in the form ``bus_name=[active_power, reactive_power]``.
            loading (dict[list]): Vectors for active power ('p') and power factor tangent ('tan_phi').
            ps_sim (Pss, optional): PowerSystemSimulation object to use. If ``None``,
                a default Pss is created internally.
        """
        if ps_sim is None:
            # Just do a basic setup
            self.ps_sim = Pss(
                parallel_sims=1,
                sim_time=1,
                time_step=0.005,
                verbose=False,
                solver="heun",
                grid_data=load_model(),
            )
        elif isinstance(ps_sim, Pss):
            # If 'ps_sim' is a Pss object, use it
            self.ps_sim = ps_sim
        else:
            raise ValueError(
                "Invalid value for 'ps_sim'. Please provide a Pss object or leave this setting."
            )

        if load_model is not None:
            self.__load_model = load_model
        else:
            raise ValueError("No load model provided. Please provide a load model.")

        self.__p_vector = []
        for key in loading.keys():
            if key == "p":
                self.__p_vector = loading[key]
            elif key == "tan_phi":
                self.__phi_vector = loading[key]
            else:
                raise ValueError(
                    f"Loading parameter '{key}' not found in load model attributes."
                )

        if self.__p_vector is None or self.__phi_vector is None:
            raise ValueError(
                "Not sufficient loading parameters provided. Please provide a load model with "
                "'p' and 'tan_phi' attributes."
            )

        self.result = {}
        self.res_variation = {}

    def run_calculation(self, bus: list[str]) -> dict[pd.DataFrame]:
        """Run iterative load flow calculations to build nose curves for given buses.

        The method iterates over the configured ``p`` and ``tan_phi`` vectors, creates
        grids using the provided load model, and runs load flows until convergence
        fails (interpreted as the critical point).

        Args:
            bus (list[str]): List of bus names to calculate nose curves for.

        Returns:
            dict[pd.DataFrame]: Dictionary mapping bus names to DataFrames with columns
                [p, q, tan_phi, v].
        """
        # Check if bus is in the grid and translate to bus indices
        bus_indices = []
        for b in bus:
            if b not in self.ps_sim.bus_idxs.keys():
                raise ValueError(
                    f"Bus '{b}' not found in grid. Please provide a valid bus name."
                )
            bus_indices.append(self.ps_sim.bus_idxs[b])

        # Save ps_sim state
        ps_state = self.__reset_sim_parameters()

        # Run load flow calculation for each bus
        for num, bus_index in enumerate(bus_indices):
            ind_result = pd.DataFrame(columns=["p", "q", "tan_phi", "v"])
            for i, phi in enumerate(self.__phi_vector):
                for j, p in enumerate(self.__p_vector):
                    # Reset parameters of the load model
                    self.__reset_sim_parameters()

                    load = {bus[num]: [p, p * phi]}

                    # Create a new grid
                    try:
                        grid_data = self.__load_model(**load)
                    except ValueError as exc:
                        raise ValueError(
                            "Error in load model interface. Please check the load models "
                            "loads are parameterized correctly. Expectation is a list and "
                            "the bus name as variable (as Bus_name=[p, q])."
                        ) from exc
                    self.ps_sim.create_grid(grid_data)
                    # Run load flow calculation until convergence is not possible anymore
                    # -> Critical Point
                    try:
                        s_calc = do_load_flow(self.ps_sim)
                    except:
                        break

                    ind_result.loc[len(ind_result)] = [
                        -s_calc[0, bus_index].real * self.ps_sim.base_mva,
                        -s_calc[0, bus_index].imag * self.ps_sim.base_mva,
                        phi,
                        self.ps_sim.busses[bus_index].voltage,
                    ]

            # Restore the previous state of the PowerSystemSimulation object
            self.__restore_ps_state(ps_state)

            # Append the results for each BUS as seperate dataframe
            self.result[bus[num]] = ind_result
            del ind_result

        return self.result

    def plot_nose_curve(
        self,
        busses: list[str],
        size: tuple = (12, 6),
        title: bool = True,
        save_path: str = None,
    ) -> plt.Axes:
        """Plot nose curves for the specified buses.

        Args:
            busses (list[str]): List of bus names to plot nose curves for.
            size (tuple, optional): Figure size (width, height). Defaults to (12, 6).
            title (bool, optional): If True, add a title to each plot. Defaults to True.
            save_path (str, optional): If provided, save plots as PDF files to this path.

        Returns:
            plt.Axes: The last matplotlib Axes object created.
        """
        for bus in busses:
            id_result = self.result[bus]

            # plt.figure(figsize=size)
            fig, ax = plt.subplots(figsize=size)
            for i, phi in enumerate(id_result["tan_phi"].unique()):
                ax.plot(
                    id_result[id_result["tan_phi"] == phi]["p"],
                    np.abs(id_result[id_result["tan_phi"] == phi]["v"]),
                    label=r"$\tan(\phi)={}$".format(phi),
                )

            ax.set_xlabel("Active Power in MW")
            ax.set_ylabel("Voltage in p.u.")

            if title:
                ax.set_title("Nose Curves for Bus {}".format(bus))
            ax.legend()
            ax.grid()

            if save_path is not None:
                plt.savefig(save_path + f"{bus}_nose_curve.pdf")
            else:
                pass

        return ax

    def get_max_loadings(self, busses: list[str]) -> dict[dict[pd.DataFrame]]:
        """Return maximum loadings (p, q, v) for each bus and power angle.

        Args:
            busses (list[str]): List of bus names to compute maximum loadings for.

        Returns:
            dict[dict[pd.DataFrame]]: Nested dictionary mapping each bus to a dict of
                power-angle entries containing DataFrames with columns [p, q, v].
        """
        max_loadings = {}
        for bus in busses:
            id_result = self.result[f"{bus}"]
            max_loadings[bus] = {}
            for phi in self.__phi_vector:
                entry = pd.DataFrame(columns=["p", "q", "v"])
                entry.loc[len(entry)] = [
                    np.abs(id_result[id_result["tan_phi"] == phi]["p"]).max().squeeze(),
                    np.abs(id_result[id_result["tan_phi"] == phi]["q"]).max().squeeze(),
                    np.abs(id_result[id_result["tan_phi"] == phi]["v"]).max().squeeze(),
                ]
                max_loadings[bus][phi] = entry
                del entry

        return max_loadings

    def run_variation_calculation(
        self,
        bus: str,
        variation_parameter: callable,
        variation_values: list,
    ) -> dict[pd.DataFrame]:
        """Run nose curve calculations while varying a simulation parameter.

        Only a single bus and a single variation parameter are supported per call.

        Args:
            bus (str): Bus name for which the NoseCurve is calculated.
            variation_parameter (callable): Callable which applies the variation to the
                simulation (e.g., a setter that accepts ``(ps_sim, value)``).
            variation_values (list): List of values to iterate for the variation parameter.

        Returns:
            dict[pd.DataFrame]: Dictionary mapping each variation value to a DataFrame
                with columns [p, q, v].

        Raises:
            ValueError: If the provided load model is not correctly parameterized
                (expects arguments like ``BusName=[P, Q]``).
        """
        res_store = self.result
        bus_index = self.ps_sim.bus_idxs[bus]

        var = np.round(variation_values, 2)

        for i, value in enumerate(var):
            ind_result = pd.DataFrame(columns=["p", "q", "v"])
            for j, p in enumerate(self.__p_vector):
                # Reset parameters of the load model
                self.__reset_sim_parameters()
                self.ps_sim.static_y_matrix = None

                load = {bus: [p, 0]}

                # Create a new grid
                try:
                    grid_data = self.__load_model(**load, u_l=value)
                except:
                    raise ValueError(
                        f"Error in load model interface. Please check the load models loads are "
                        f"parameterized correctly. Expectation is a list and the bus name as "
                        "variable (as Bus_name=[p, q])."
                    )
                self.ps_sim.create_grid(grid_data)

                # Set the new parameter
                variation_parameter(self.ps_sim, value)
                # Run load flow calculation until convergence is not possible anymore
                # -> Critical Point
                try:
                    s_calc = do_load_flow(self.ps_sim)
                except:
                    break

                ind_result.loc[len(ind_result)] = [
                    -s_calc[0, bus_index].real * self.ps_sim.base_mva,
                    -s_calc[0, bus_index].imag * self.ps_sim.base_mva,
                    self.ps_sim.busses[bus_index].voltage,
                ]

            # Save each result
            self.res_variation[f"{value}"] = ind_result
            del ind_result

        # Restore the previous result variable
        self.result = res_store

        return self.res_variation

    def plot_nose_curve_variation(
        self,
        param_var_dict: dict[pd.DataFrame],
        current_plot: plt.Axes,
        label_args: dict = {"labels": ["OLTC Ratios"]},
        plot_args: dict = {"linestyle": "dashed", "color": ees_red},
    ) -> plt.Axes:
        """Insert nose-curve variation lines into an existing plot.

        Args:
            param_var_dict (dict[pd.DataFrame]): Dictionary of nose curve results keyed by
                variation value.
            current_plot (plt.Axes): Existing axes to which curves should be added.
            label_args (dict, optional): Label settings to append to legend (default: {'labels': ['OLTC Ratios']}).
            plot_args (dict, optional): Plotting kwargs for the inserted curves (default style provided).

        Returns:
            plt.Axes: The updated axes object with inserted curves and updated legend.
        """
        handles, labels = current_plot.get_legend_handles_labels()

        for key in param_var_dict:
            if key is not str("1.0"):
                current_plot.plot(
                    param_var_dict[key]["p"],
                    np.abs(param_var_dict[key]["v"]),
                    **plot_args,
                )

        if label_args is not None:
            labels += label_args["labels"]
            current_plot.legend(labels=labels)

        return current_plot

    def add_load_to_plot(
        self,
        load: list[float, float],
        bus: str,
        current_plot: plt.Axes,
        load_model: dict[float] = {"z": 1, "i": 0, "p": 0},
        y_lims: tuple = (0.6, 1.2),
        plot_args: dict = {},
    ) -> plt.Axes:
        """Add a specified load curve to an existing plot.

        Args:
            load (list[float, float]): Load parameter as [P, Q].
            bus (str): Bus name where the load is added.
            current_plot (plt.Axes): Existing axes to which the load curve will be added.
            load_model (dict, optional): ZIP model shares as a dict like {'z': 1, 'i': 0, 'p': 0}.
            y_lims (tuple, optional): Y-axis limits for the curve generation. Defaults to (0.6, 1.2).
            plot_args (dict, optional): Additional matplotlib plotting kwargs.

        Returns:
            plt.Axes: The axes object with the inserted load curve.

        Raises:
            ValueError: If the load model callable is not parameterized correctly
                (expects kwargs like ``BusName=[P, Q]``).
        """
        self.__reset_sim_parameters()

        load_sim = {bus: load}

        # Create a new grid
        try:
            grid_data = self.__load_model(**load_sim)
        except:
            raise ValueError(
                f"Error in load model interface. Please check the load models loads are "
                f"parameterized correctly. Expectation is a list and the bus name as "
                f"variable (as Bus_name=[p, q])."
            )
        self.ps_sim.create_grid(grid_data)

        # Run load flow calculation until convergence is not possible anymore -> Critical Point
        try:
            s_calc = do_load_flow(self.ps_sim)
        except:
            pass

        v_0 = self.ps_sim.busses[self.ps_sim.bus_idxs[bus]].voltage.squeeze()

        # Calculate plotting values
        v = np.linspace(y_lims[0], y_lims[1], 100)
        s = self.__solve_load_function(v=v, v_0=v_0, s_0=load, load_model=load_model)

        # Add the load curve to the plot
        if "label" not in plot_args.keys():
            plot_args["label"] = (f"Load Curve $P={load[0]}$ MW",)
        current_plot.plot(np.abs(s), v, **plot_args)

        return current_plot

    def __solve_load_function(
        self,
        v: list[float],
        v_0: complex,
        s_0: list[float],
        load_model: dict[float],
    ) -> list[float]:
        """Compute the complex power values for a given voltage vector and ZIP model.

        Args:
            v (list[float]): Vector of voltages to evaluate.
            v_0 (complex): Reference voltage for the accounted load.
            s_0 (list[float]): Base complex power as [P, Q].
            load_model (dict[float]): ZIP model shares as {'z': ..., 'i': ..., 'p': ...}.

        Returns:
            list[float]: Complex power vector corresponding to voltages ``v``.
        """
        s = (s_0[0] + 1j * s_0[1]) * (
            load_model["z"] * (v / v_0) ** 2
            + load_model["i"] * v / v_0
            + load_model["p"] * np.ones(v.shape)
        )

        return s

    def __reset_sim_parameters(self) -> None:
        """Reset the simulation grid parameters and return the previous state.

        Returns:
            dict: The power system state before the reset (keys: 'busses', 'non_slack_busses',
                'bus_idxs', 'lines', 'trafos').
        """
        # Get the current state of the PowerSystemSimulation object
        ps_state = {
            "busses": self.ps_sim.busses,
            "non_slack_busses": self.ps_sim.non_slack_busses,
            "bus_idxs": self.ps_sim.bus_idxs,
            "lines": self.ps_sim.lines,
            "trafos": self.ps_sim.trafos,
        }

        # Reset the simulation parameters
        self.ps_sim.busses = []
        self.ps_sim.non_slack_busses = []
        self.ps_sim.bus_idxs = {}
        self.ps_sim.lines = []
        self.ps_sim.trafos = []

        return ps_state

    def __restore_ps_state(self, ps_state: dict) -> None:
        """Restore the PowerSystemSimulation object to the provided state.

        Args:
            ps_state (dict): Power system state produced by :meth:`__reset_sim_parameters`.
        """
        self.ps_sim.busses = ps_state["busses"]
        self.ps_sim.non_slack_busses = ps_state["non_slack_busses"]
        self.ps_sim.bus_idxs = ps_state["bus_idxs"]
        self.ps_sim.lines = ps_state["lines"]
        self.ps_sim.trafos = ps_state["trafos"]
        return


# Calculation of Critical Time points in the simulation
class CrtiticalTimes:
    """Calculate critical timestamps where voltages leave configured envelopes.

    The class checks for envelope violations (e.g., TVI or FRT) and records the
    timestamps where the bus voltages cross the envelope boundaries outwardly.

    Attributes:
        calculation_dict (list): List of envelope types to check (e.g., ['tvi', 'frt']).
        results (dict): Recorded timestamps per bus and envelope type.
    """

    def __init__(self, calculation_dict: list = ["tvi"]):
        """Initialize a CrtiticalTimes analyzer.

        Args:
            calculation_dict (list, optional): List of envelope types to check. Supported:
                'tvi', 'frt'. Defaults to ['tvi'].
        """
        self.calculation_dict = calculation_dict
        self.frt_env = ViolationIntegral(envelope="FRT")
        self.frt_env.set_env_params(
            env_params={"beta": 0.05, "v_st": 0.9, "t_start": 1}
        )
        self.tvi_env = ViolationIntegral(envelope="TVI")

        self.frt_time = None
        self.tvi_time = None

        self.results = {}

    def get_result(self, time: list, voltages: list[list]) -> dict:
        """Compute critical timestamps for envelope violations for each bus.

        Args:
            time (list): Time vector of the time series computation.
            voltages (list[list]): Voltage time series. Expected shape is (time, buses)
                or an array that can be transposed to iterate per bus.

        Returns:
            dict: Dictionary keyed by bus (e.g., 'bus_0') containing lists of timestamps
                for each enabled envelope type and other metadata.
        """
        self.add_env_data(time)
        # Calculate envelope violations
        for i, v_bb in enumerate(np.swapaxes(voltages, 0, 1)):
            self.results[f"bus_{i}"] = {}
            if "tvi" in self.calculation_dict:
                self.results[f"bus_{i}"]["tvi"] = []
            if "frt" in self.calculation_dict:
                self.results[f"bus_{i}"]["frt"] = []
            self.results[f"bus_{i}"]["max_loading"] = []
            tvi_flag = False
            frt_flag = False
            max_loading_flag = False
            for j, v in enumerate(v_bb):
                # tvi testing
                if "tvi" in self.calculation_dict:
                    if (
                        v > self.tvi_time[0][j] or v < self.tvi_time[1][j]
                    ) and tvi_flag == False:
                        self.results[f"bus_{i}"]["tvi"].append(
                            float(np.round(time[j], 4))
                        )
                        tvi_flag = True
                    elif (
                        v <= self.tvi_time[0][j] and v >= self.tvi_time[1][j]
                    ) and tvi_flag == True:
                        tvi_flag = False
                # frt testing
                if "frt" in self.calculation_dict:
                    if (
                        v > self.frt_time[0][j] or v < self.frt_time[1][j]
                    ) and frt_flag == False:
                        self.results[f"bus_{i}"]["frt"].append(
                            float(np.round(time[j], 4))
                        )
                        frt_flag = True
                    elif (
                        v <= self.frt_time[0][j] and v >= self.frt_time[1][j]
                    ) and frt_flag == True:
                        frt_flag = False

        return self.results

    def add_env_data(self, time: list) -> None:
        """Precompute and store envelope data for the provided time vector.

        Args:
            time (list): Time vector of the time series computation.
        """
        self.frt_time = self.frt_env.get_env(time)
        self.tvi_time = self.tvi_env.get_env(time)


voltage_index_dict = {"VI": ViolationIntegral}
