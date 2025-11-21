"""Models a very basic transformer in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class Transformer_Old(StaticModelInterface):
    """
    Represent a transformer in the power system simulation.

    This class models the electrical characteristics of a transformer, including resistance,
    reactance, and connection between two buses in the power system.

    Attributes:
        from_bus_id (int): The index of the primary side bus of the transformer.
        to_bus_id (int): The index of the secondary side bus of the transformer.
        r (torch.Tensor): Resistance of the transformer in p.u.
        x (torch.Tensor): Reactance of the transformer in p.u.
        b (torch.Tensor): Susceptance of the transformer in p.u.
    """

    def __init__(
        self,
        s_n_sys,
        param_dict=None,
        name=None,
        from_bus=None,
        to_bus=None,
        s_n=None,
        r=None,
        x=None,
        v_n_from=None,
        v_n_to=None,
        b=0,
    ):
        """
        Initialize the Transformer object with electrical parameters and connected buses.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the transformer.
            name (str, optional): The name of the transformer.
            from_bus (str, optional): The name of the primary side bus of the transformer.
            to_bus (str, optional): The name of the secondary side bus of the transformer.
            s_n (float, optional): The nominal power of the transformer in MVA.
            r (float, optional): The resistance of the transformer in p.u.
            x (float, optional): The reactance of the transformer in p.u.
            v_n_from (float, optional): The nominal voltage of the primary side of the
            transformer in kV.
            v_n_to (float, optional): The nominal voltage of the secondary side of the
            transformer in kV.
            b (float, optional): The susceptance of the transformer in p.u.
        """
        if param_dict is None:
            param_dict = {
                "name": name,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "S_n": s_n,
                "R": r,
                "X": x,
                "V_n_from": v_n_from,
                "V_n_to": v_n_to,
                "B": b,
            }
        self.name = param_dict["name"]
        self.from_bus_name = param_dict["from_bus"]
        self.to_bus_name = param_dict["to_bus"]

        self.from_bus_id = None
        self.to_bus_id = None

        self.s_n = param_dict["S_n"]
        self.r = param_dict["R"]
        self.x = param_dict["X"]
        self.v_n_from = param_dict["V_n_from"]
        self.v_n_to = param_dict["V_n_to"]
        self.b = param_dict.get("B", 0)

        self.s_n_sys = s_n_sys

    def differential(self):
        """Get the differential equations of the transformer model (old)."""
        return torch.zeros((1, 1), dtype=torch.complex128)

    def get_state_vector(self):
        """Get the state vector of the transformer model (old)."""
        return torch.zeros((1, 1), dtype=torch.complex128)

    def set_state_vector(self, state_vector):
        """Set the state vector of the transformer model (old)."""
        pass

    def calc_admittance(self, return_need=True, init_proc=False) -> None:
        """
        Calculate the admittance matrix of the transformer as a two-port network (4x4 tensor).

        Returns:
            Two-port admittance matrix of the transformer (4x4 matrix).
        """
        # calculate all contributions to the admittance matrix
        Y_diagonal = (
            (1 / (self.r + 1j * self.x) + 1j * self.b / 2) * self.s_n / self.s_n_sys
        )
        Y_off_diagonal = -1 / (self.r + 1j * self.x) * self.s_n / self.s_n_sys

        # if a return of the function is needed, return the y_matrix as well
        if return_need:
            # returning the admittance matrix
            return torch.tensor(
                [[Y_diagonal, Y_off_diagonal], [Y_off_diagonal, Y_diagonal]]
            )

    def calc_admittance_static(self, return_need=True) -> None:
        """
        Calculate the admittance matrix of the transformer as a two-port network (4x4 tensor).

        Returns:
            Two-port admittance matrix of the transformer (4x4 matrix).
        """
        # calculate all contributions to the admittance matrix
        Y_diagonal = (
            (1 / (self.r + 1j * self.x) + 1j * self.b / 2) * self.s_n / self.s_n_sys
        )
        Y_off_diagonal = -1 / (self.r + 1j * self.x) * self.s_n / self.s_n_sys

        # if a return of the function is needed, return the y_matrix as well
        if return_need:
            # returning the admittance matrix
            return torch.tensor(
                [[Y_diagonal, Y_off_diagonal], [Y_off_diagonal, Y_diagonal]]
            )

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations for the transformer.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.from_bus_id = (
            torch.ones((parallel_sims, 1), dtype=torch.int32) * self.from_bus_id
        )
        self.to_bus_id = (
            torch.ones((parallel_sims, 1), dtype=torch.int32) * self.to_bus_id
        )
        self.r = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.r
        self.x = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.x
        self.v_n_from = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_from
        )
        self.v_n_to = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_to

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Get additional values for the transformer (old).

        Args:
            value (str): Description of the relevant parameter.

        Returns:
            Array of desired values
        """
        if sim is None:
            raise ValueError("Simulation object has to be handed over.")

        voltage = (
            sim.busses[sim.bus_idxs[self.from_bus_name]].voltage
            - sim.busses[sim.bus_idxs[self.to_bus_name]].voltage
        )
        current = (voltage) * self.calc_admittance()[0, 0]

        if value == "current":
            return current
        elif value == "current_mag":
            return torch.abs(current)
        elif value == "current_angle":
            if angle_type == "deg":
                return torch.angle(current) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(current)
        elif value == "phase_angle":
            if angle_type == "rad":
                return torch.angle(voltage) - torch.angle(current)
            if angle_type == "deg":
                return (torch.angle(voltage) - torch.angle(current)) / torch.pi * 180
            elif value == "P":
                return torch.real(voltage) * torch.conj(current)
            elif value == "Q":
                return torch.imag(voltage) * torch.conj(current)
            elif value == "S":
                return voltage * torch.conj(current)
        else:
            raise ValueError("Invalid value parameter")

    def initialize(self, from_voltage, to_voltage):
        """Initialize the transformer (old)."""
        pass
