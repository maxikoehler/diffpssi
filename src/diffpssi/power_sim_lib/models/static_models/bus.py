"""Models a bus in a power system."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)


class Bus(StaticModelInterface):
    """
    Represent a bus in the power system simulation.

    A bus is a node at which power system components such as generators, loads,
    and lines are connected. This class models the electrical behavior of the bus and interactions
    with connected components.

    Attributes:
        name (str): The name of the bus.
        lf_type (str): The load flow type of the bus.
        v_n (torch.Tensor): Nominal voltage at the bus.
        models (list): List of models (generators, loads, etc.) connected to the bus.
        voltage (torch.Tensor): Current voltage at the bus.
    """

    def __init__(self, param_dict=None, name=None, v_n=None):
        """
        Initialize the Bus object with the given name, load flow type, and nominal voltage.

        Args:
            param_dict (dict): Dictionary of parameters for the bus.
            name (str): The name of the bus.
            v_n (float): Nominal voltage at the bus in kV.
        """
        self.parallel_sims = None
        if param_dict is None:
            param_dict = {
                "name": name,
                "V_n": v_n,
            }

        self.name = param_dict["name"]
        self.v_n = param_dict["V_n"]

        self.models = []
        self.diff_models = []
        self.lf_type = "PQ"
        self.voltage = 1.0

    def get_current_injections(self):
        """
        Calculate the total current injection at the bus from all connected models.

        Returns:
            torch.Tensor: The total current injection at the bus.
        """
        current_inj = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        for model in self.models:
            try:
                current_inj += model.calc_current_injections()
            except AttributeError:
                pass

        return current_inj

    def update_voltages(self, v_bb):
        """
        Update the voltage at the bus and propagates the update to all connected models.

        Args:
            v_bb (torch.Tensor): The new busbar voltage value.
        """
        self.voltage = v_bb

        for model in self.models:
            try:
                model.update_internal_vars(v_bb)
            except AttributeError:
                pass

    def add_model(self, model):
        """
        Add a power system model (e.g., generator, load) to the bus.

        Args:
            model (GenericModel): The model to add to the bus.
        """
        self.models.append(model)
        if hasattr(model, "differential"):
            self.diff_models.append(model)

    def add_transformer(self, model):
        """
        Add a tranformer model (any Transformer model) to the bus.

        Args:
            model (Transformer): The transformer to add to the bus.
        """
        self.trafos.append(model)

    def get_lf_power(self):
        """
        Calculate the total load flow power at the bus from all connected models.

        Returns:
            torch.Tensor: The total load flow power at the bus.
        """
        s_soll = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        for model in self.models:
            try:
                s_soll += model.get_lf_power()
            except AttributeError:
                pass
        return s_soll

    def reset(self):
        """Reset the bus to its initial state."""
        try:
            self.update_voltages(self.models[-1].v_soll)
        # except index and attribute error if no models are connected
        except (IndexError, AttributeError):
            self.update_voltages(
                torch.ones((self.parallel_sims, 1), dtype=torch.complex128)
            )

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulations for the bus.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.v_n = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_n
        self.voltage = (
            torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.voltage
        )
        self.parallel_sims = parallel_sims

    def get_value(self, value, angle_type="deg"):
        """
        Get desired values at the Busbar.

        Args:
            value (str): Descriptions of the relavant parameter.
        Returns:
            list: Array of desired values
        """
        if value == "voltage":
            return self.voltage
        elif value == "voltage_mag":
            return torch.abs(self.voltage)
        elif value == "voltage_angle":
            if angle_type == "deg":
                return torch.angle(self.voltage) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(self.voltage)
        elif value == "current":
            return self.get_current_injections()
        elif value == "current_mag":
            return torch.abs(self.get_current_injections())
        elif value == "current_angle":
            if angle_type == "deg":
                return torch.angle(self.get_current_injections()) / torch.pi * 180
            if angle_type == "rad":
                return torch.angle(self.get_current_injections())
        elif value == "phase_angle":
            if angle_type == "rad":
                return torch.angle(self.voltage) - torch.angle(
                    self.get_current_injections()
                )
            if angle_type == "deg":
                return (
                    (
                        torch.angle(self.voltage)
                        - torch.angle(self.get_current_injections())
                    )
                    / torch.pi
                    * 180
                )
        elif value == "S":
            current = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
            for model in self.models:
                try:
                    current += model.calc_current_injections()
                except AttributeError:
                    current += model.i_inj

            return torch.conj(current) * self.voltage
        else:
            raise ValueError("Invalid value parameter")
