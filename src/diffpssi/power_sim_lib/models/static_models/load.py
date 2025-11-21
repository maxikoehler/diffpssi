"""Models a load in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class Load(StaticModelInterface):
    """
    Represent a load in the power system simulation.

    This class models an electrical load, characterized by its active and reactive power demand,
    at a bus in the power system.

    Attributes:
        p_soll_mw (float): Desired active power (MW) of the load.
        q_soll_mvar (float): Desired reactive power (MVAR) of the load.
    """

    def __init__(
        self, s_n_sys, param_dict=None, name=None, bus=None, p=None, q=None, model=None
    ):
        """
        Initialize the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the load.
            name (str, optional): The name of the load.
            bus (str, optional): The name of the bus where the load is connected.
            p (float, optional): The active power demand of the load in MW.
            q (float, optional): The reactive power demand of the load in MVAR.
            model (str, optional): The model of the load. Currently only 'Z' is supported.
        """
        if param_dict is None:
            param_dict = {
                "name": name,
                "bus": bus,
                "P": p,
                "Q": q,
                "model": model,
            }
        self.name = param_dict["name"]
        self.bus = param_dict["bus"]
        self.p_soll_mw = param_dict["P"]
        self.q_soll_mvar = param_dict["Q"]
        self.p_atm = self.p_soll_mw
        self.q_atm = self.q_soll_mvar
        self.model = param_dict["model"]

        self.v_atm = 0
        self.v_init = 0

        if self.model not in ["Z", "I", "P", "T", "ZIP"]:
            raise ValueError("The load model is not supported.")
            # raise ValueError('Only Z model is supported for loads')

        if self.model == "Z":
            self.z = 1
            self.i = 0
            self.p = 0
        elif self.model == "I":
            self.z = 0
            self.i = 1
            self.p = 0
        elif self.model == "P":
            self.z = 0
            self.i = 0
            self.p = 1
        elif self.model == "ZIP":
            # Better if ONE preset is set and a Warning is raised instead of an error?
            if load_dict is None:
                raise ValueError("ZIP model needs a specified load_dict.")
            self.z = load_dict["z"]
            self.i = load_dict["i"]
            self.p = load_dict["p"]

        self.s_n_sys = s_n_sys

        self.y_load = 0
        self.i_inj = 0

    def get_lf_power(self):
        """
        Calculate and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power
        (sum of active and reactive power) of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """
        return -(self.p_soll_mw + 1j * self.q_soll_mvar) / self.s_n_sys

    def get_admittance(self, dyn):
        """
        Compute and returns the admittance of the Load.

        The admittance is computed based on dynamic or static analysis, determined
        by the 'dyn' parameter. If dynamic analysis is selected, the existing load
        admittance value is returned. Otherwise, a zero admittance tensor is returned
        for static analysis.

        Args:
            dyn (bool): Flag indicating whether dynamic (True) or static (False)
            analysis is to be used.

        Returns:
            torch.Tensor: The calculated admittance tensor.
        """
        if dyn:
            s_load = (self.p_soll_mw + 1j * self.q_soll_mvar) / self.s_n_sys
            z_load = torch.conj(torch.abs(self.v_init) ** 2 / s_load)

            self.i_inj = s_load * self.s_n_sys / torch.conj(self.v_atm)

            self.y_load = 1 / z_load

            return self.y_load

        return torch.zeros_like(self.p_soll_mw)

    # noinspection PyUnusedLocal
    def initialize(self, s_calc, v_bb):
        """
        Initialize the Load by calculating its admittance.

        This method computes the Load's admittance based on the calculated complex power
        and the voltage base values.

        Args:
            s_calc (torch.Tensor): Calculated apparent power.
            v_bb (torch.Tensor): Busbar voltage value.
        """
        self.v_atm = v_bb
        self.v_init = v_bb
        s_load = (self.p_soll_mw + 1j * self.q_soll_mvar) / self.s_n_sys

        self.i_inj = torch.conj(-s_load * self.s_n_sys / self.v_atm)

        z_load = torch.conj(torch.abs(v_bb) ** 2 / s_load)
        self.y_load = 1 / z_load

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulation for the Load.

        This method sets up the load for parallel simulations by initializing the active and
        reactive power demands as tensors with dimensions corresponding to the number of
        parallel simulations.

        Args:
            parallel_sims (int): The number of parallel simulations to enable.
        """
        self.p_soll_mw = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.p_soll_mw
        )
        self.q_soll_mvar = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.q_soll_mvar
        )
        self.v_atm = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_atm
        self.v_init = (
            torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_init
        )

        self.i_inj = (
            torch.zeros((parallel_sims, 1), dtype=torch.complex128) * self.i_inj
        )

    def calc_current_injections(self):
        """
        Calculate and returns the current injections for the Load.

        This method is used in parallel simulations to compute the current injections.
        It currently returns a zero tensor.

        Returns:
            torch.Tensor: A tensor of zero current injections for the parallel simulations.
        """
        return torch.zeros_like(self.p_soll_mw)

    def update_internal_vars(self, v_bb):
        """
        Update internal variables of the Load based on the given voltage base values.

        This method is a placeholder for updating any internal variables of the Load object,
        based on the provided voltage base values.

        Args:
            v_bb (float): Base voltage value.
        """
        self.v_atm = v_bb

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Get additional values of Load objects.

        Args:
            value (str): Descriptions of the relavant parameter.
        Returns:
            list: Array of desired values
        """
        if sim is None:
            raise ValueError("Simulation object has to be handed over.")

        current = (
            sim.busses[self.from_bus_id].voltage - sim.busses[self.from_bus_id].voltage
        ) * self.y_load

        if value == "current":
            return current
        elif value == "current_mag":
            return torch.abs(current)
        elif value == "current_angle":
            if angle_type == "deg":
                return torch.angle(current) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(current)
        elif value == "P_atm":
            return self.p_atm
        elif value == "Q_atm":
            return self.q_atm
        elif value == "S_atm":
            return self.p_atm + 1j * self.q_atm
        elif value == "phase_angle":
            if angle_type == "deg":
                return (
                    (
                        torch.angle(sim.busses[sim.bus_idxs[self.bus]].voltage)
                        - torch.angle(current)
                    )
                    / torch.pi
                    * 180
                )
            if angle_type == "rad":
                return torch.angle(
                    sim.busses[sim.bus_idxs[self.bus]].voltage
                ) - torch.angle(current)
        else:
            raise ValueError("Invalid value parameter")
