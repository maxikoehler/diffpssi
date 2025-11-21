"""Models a shunt in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class Shunt(StaticModelInterface):
    """
    Represent a shunt in the power system simulation.

    This class models an electrical load, characterized by its active and reactive power demand,
    at a bus in the power system.

    Attributes:
        q_soll_mvar (float): Desired reactive power (MVAR) of the load.
    """

    def __init__(
        self,
        s_n_sys,
        param_dict=None,
        name=None,
        bus=None,
        v_n=None,
        q=None,
        model=None,
    ):
        """
        Initialize the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the load.
            name (str, optional): The name of the load.
            bus (str, optional): The name of the bus where the load is connected.
            v_n (float, optional): The nominal voltage of the load in kV.
            q (float, optional): The reactive power demand of the load in MVAR.
            model (str, optional): The model of the load. Currently only 'Z' is supported.
        """
        if param_dict is None:
            param_dict = {
                "name": name,
                "bus": bus,
                "V_n": v_n,
                "Q": q,
                "model": model,
            }
        self.name = param_dict["name"]
        self.bus = param_dict["bus"]
        self.v_n = param_dict["V_n"]
        self.q_soll_mvar = param_dict["Q"]
        self.model = param_dict["model"]

        if not self.model == "Z":
            raise ValueError("Only Z model is supported for shunts")

        self.s_n_sys = s_n_sys

        s_shunt = torch.tensor(-1j * self.q_soll_mvar / self.s_n_sys)
        z = torch.conj(1 / s_shunt)
        self.y_shunt = 1 / z

    def get_lf_power(self):
        """
        Calculate and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power
        (sum of active and reactive power) of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """
        return torch.zeros_like(self.q_soll_mvar)

    # noinspection PyUnusedLocal
    def get_admittance(self, dyn):
        """
        Compute and returns the admittance of the Load.

        The admittance is computed based on dynamic or static analysis, determined
        by the 'dyn' parameter. Currently, the same admittance is returned for both
        dynamic and static analysis.

        Args:
            dyn (bool): Flag indicating whether dynamic (True) or static (False)
            analysis is to be used.

        Returns:
            torch.Tensor: The calculated admittance tensor.
        """
        # parameter dyn is not used here, but is required for compatibility with other models
        return self.y_shunt

    def initialize(self, s_calc, v_bb):
        """
        Initialize the Load (theoretically). Currently not used.

        Args:
            s_calc: The calculated complex power.
            v_bb: Voltage at the busbar.
        """
        return

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulation for the Load.

        This method sets up the load for parallel simulations by initializing the active
        and reactive power demands as tensors with dimensions corresponding to the number
        of parallel simulations.

        Args:
            parallel_sims (int): The number of parallel simulations to enable.
        """
        self.q_soll_mvar = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.q_soll_mvar
        )
        self.y_shunt = (
            torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.y_shunt
        )

    def calc_current_injections(self):
        """
        Calculate and returns the current injections for the Load.

        This method is used in parallel simulations to compute the current injections.
        It currently returns a zero tensor.

        Returns:
            torch.Tensor: A tensor of zero current injections for the parallel simulations.
        """
        return torch.zeros_like(self.q_soll_mvar)

    def update_internal_vars(self, v_bb):
        """
        Update internal variables of the Load.

        Basis on the given voltage at the busbar (theoretically).
        Currently not used and only for compatibility with other models.

        Args:
            v_bb (float): Busbar voltage.
        """
        return

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Get additional values of Shunt objects.

        Args:
            value (str): Descriptions of the relavant parameter.
        Returns:
            list: Array of desired values
        """
        if sim is None:
            raise ValueError("Simulation object has to be handed over.")

        current = (
            sim.busses[self.from_bus_id].voltage - sim.busses[self.from_bus_id].voltage
        ) * self.y_shunt

        if value == "current":
            return current
        elif value == "current_mag":
            return torch.abs(current)
        elif value == "current_angle":
            if angle_type == "deg":
                return torch.angle(current) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(current)
        elif value == "P":
            return torch.real(sim.busses[sim.bus_idxs[self.bus]] * torch.conj(current))
        elif value == "Q":
            return torch.imag(sim.busses[sim.bus_idxs[self.bus]] * torch.conj(current))
        elif value == "S":
            return sim.busses[sim.bus_idxs[self.bus]] * torch.conj(current)
        elif value == "phase_angle":
            if angle_type == "deg":
                pass
            if angle_type == "rad":
                pass
        else:
            raise ValueError("Invalid value parameter")
