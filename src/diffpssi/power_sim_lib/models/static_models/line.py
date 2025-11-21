"""Models a line in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class Line(StaticModelInterface):
    """
    Represent a transmission line in the power system simulation.

    This class models the electrical characteristics of a transmission line, including resistance,
    reactance, and susceptance, between two buses in the power system.

    Attributes:
        from_bus_id (int): The index of the starting bus of the line.
        to_bus_id (int): The index of the ending bus of the line.
        r (torch.Tensor): Resistance of the line.
        x (torch.Tensor): Reactance of the line.
        b (torch.Tensor): Susceptance of the line.
    """

    def __init__(
        self,
        s_n_sys,
        v_n_sys,
        param_dict=None,
        name=None,
        from_bus=None,
        to_bus=None,
        length=None,
        s_n=None,
        v_n=None,
        unit=None,
        r=None,
        x=None,
        b=None,
    ):
        """
        Initialize the Line object with the specified electrical parameters and connected buses.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the line.
            name (str, optional): The name of the line.
            from_bus (str, optional): The name of the starting bus of the line.
            to_bus (str, optional): The name of the ending bus of the line.
            length (float, optional): The length of the line in km.
            s_n (float, optional): The nominal power of the line in MVA.
            v_n (float, optional): The nominal voltage of the line in kV.
            unit (str, optional): The unit of the line parameters. Either 'Ohm' or 'p.u.'.
            r (float, optional): The resistance of the line (either in Ohm or p.u.)/length.
            x (float, optional): The reactance of the line (either in Ohm or p.u.)/length.
            b (float, optional): The susceptance of the line (either in Ohm or p.u.)/length.
        """
        if param_dict is None:
            param_dict = {
                "name": name,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "length": length,
                "S_n": s_n,
                "V_n": v_n,
                "unit": unit,
                "R": r,
                "X": x,
                "B": b,
            }
        self.name = param_dict["name"]
        self.from_bus_name = param_dict["from_bus"]
        self.to_bus_name = param_dict["to_bus"]

        self.from_bus_id = None
        self.to_bus_id = None

        length = param_dict["length"]
        s_n = param_dict.get("S_n", s_n_sys)
        v_n = param_dict.get("V_n", v_n_sys)
        unit = param_dict["unit"]
        r = param_dict["R"]
        x = param_dict["X"]
        b = param_dict["B"]

        z_n_sys = v_n_sys**2 / s_n_sys
        z_n = v_n**2 / s_n

        if unit == "Ohm":
            self.r = r * length / z_n_sys
            self.x = x * length / z_n_sys
            self.b = b * length * z_n_sys
        elif unit == "p.u.":
            self.r = r * z_n / z_n_sys * length
            self.x = x * z_n / z_n_sys * length
            self.b = b / z_n * z_n_sys * length
        else:
            raise ValueError("Unit not supported")

    def get_admittance_diagonal(self):
        """
        Calculate the diagonal admittance value of the line.

        Returns:
            torch.Tensor: Diagonal admittance of the line.
        """
        return 1 / (self.r + 1j * self.x) + 1j * self.b / 2

    def get_admittance_off_diagonal(self):
        """
        Calculate the off-diagonal admittance value of the line.

        Returns:
            torch.Tensor: Off-diagonal admittance of the line.
        """
        return -1 / (self.r + 1j * self.x)

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations for the line.

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
        self.b = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.b

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Get additional values for Line objects.

        Args:
            value (str): Descriptions of the relevant parameter.

        Returns:
            list: Array of desired values
        """
        if sim is None:
            raise ValueError("Simulation object has to be handed over.")

        current = (
            sim.busses[sim.bus_idxs[self.from_bus_name]].voltage
            - sim.busses[sim.bus_idxs[self.to_bus_name]].voltage
        ) * self.get_admittance_diagonal()

        if value == "current":
            return current
        elif value == "current_mag":
            return torch.abs(current)
        elif value == "current_angle":
            if angle_type == "deg":
                return torch.angle(current) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(current)
            elif value == "utilization":
                pass
            elif value == "phase_angle":
                if angle_type == "deg":
                    pass
                if angle_type == "rad":
                    pass
        else:
            raise ValueError("Invalid value parameter")
