"""OLTC Transformer model implementation."""

import logging

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.transformer.transformer_interface import Transformer
from diffpssi.power_sim_lib.models.voltage_controller import voltage_control_dict

_logger = logging.getLogger(__name__)


class OLTC_Transformer(Transformer):
    """
    Transformer with a longitudinal on-load tap-changing controller (OLTC).

    This subclass integrates a voltage controller instance (from
    ``voltage_control_dict``) and exposes methods to drive and query the
    controller state.
    """

    def __init__(
        self,
        sim,
        s_n_sys,
        param_dict,
        parallel_sims,
        oltc="oltc",
        trans_model="AM",
        name=None,
    ):
        """
        Initialize an OLTC_Transformer instance.

        Args:
            sim (object): Simulation instance the transformer belongs to.
            s_n_sys (float): System base apparent power.
            param_dict (dict): Transformer and controller configuration. See base
                class for transformer keys and ``param_dict_oltc`` for controller
                parameters when applicable.
            oltc (str or object): OLTC controller identifier or controller object.
            trans_model (str): Transformer model identifier.
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        super().__init__(s_n_sys, param_dict, parallel_sims, trans_model)

        self.sim = sim

        self.from_voltage = 0
        self.to_voltage = 0

        # OLTC related characteristics
        self.tap_side = param_dict.get("tap_side", "hv")

        # if oltc controller is a global controller, then no new instance has to be created
        oltc_controller = param_dict.get("control", "oltc")
        if isinstance(oltc, str):
            self.oltc = voltage_control_dict[oltc_controller](
                name="OLTC1",
                trafo=self,
                param_dict=param_dict.get("param_dict_oltc", None),
            )
        elif isinstance(oltc, object):
            self.oltc = oltc_controller
        elif oltc_controller is None:
            self.oltc = None
            warning = (
                f"No OLTC controller defined for OLTC transformer {self.name}. "
                "Please provide a valid controller by adding it after initialization.."
            )
            _logger.warning(warning)

        # get the initial state of the OLTC controller
        self.u_l = self.oltc.u_l
        self.u = self.u_l
        # normally: self.u_l * torch.exp(1j * (self.theta / 180) * torch.pi)
        # but somehow phase shifting not working in simulation

        self.parallel_sims = parallel_sims

    def differential(self):
        """
        Return the concatenated differential vector provided by the OLTC.

        Returns:
            torch.Tensor: Concatenated derivative vector produced by the
            OLTC controller.

        Raises:
            AttributeError: If the OLTC controller does not implement
                ``differential()``.
        """
        try:
            deriv_vec = torch.concatenate(
                [
                    self.oltc.differential(),
                ],
                axis=1,
            )
        except Exception:
            raise AttributeError(
                "No differential function for the OLTC controller is defined."
            )

        return deriv_vec

    def calc_admittance(self, return_need):
        """
        Compute admittance matrix while updating OLTC controller output.

        The method measures voltage at the configured measurement bus, obtains
        the OLTC output ratio, updates internal tap ratio and then delegates to
        :py:meth:`Transformer.calc_admittance` to compute the admittance matrix.

        Args:
            return_need (bool): If True return the computed admittance matrix.

        Returns:
            torch.Tensor or None: See :py:meth:`Transformer.calc_admittance`.
        """
        if self.measure_bus == self.from_bus:
            v_measure = self.sim.busses[self.sim.bus_idxs[self.from_bus_name]].voltage
        else:
            v_measure = self.sim.busses[self.sim.bus_idxs[self.to_bus_name]].voltage

        self.u_l = self.oltc.get_output(v_measure)

        self.u = self.u_l * torch.exp(1j * (self.theta / 180) * torch.pi)
        # rest identical to super-class
        return super().calc_admittance(return_need)

    def calc_admittance_static(self, return_need):
        """
        Compute the static admittance matrix for the OLTC transformer.

        Delegates to :py:meth:`Transformer.calc_admittance_static`.
        """
        return super().calc_admittance_static(return_need)

    def calc_current_injections(self):
        """
        Compute current injections for the OLTC transformer.

        Delegates to the base class implementation.
        """
        return super().calc_current_injections()

    def set_oltc_controller(self, oltc_model):
        """
        Associate or replace the OLTC controller for this transformer.

        Args:
            oltc_model (str or object): Controller identifier or controller
                instance to associate with the transformer.

        Returns:
            None
        """
        if isinstance(oltc_model, str):
            self.oltc = voltage_control_dict[oltc_model]()
        elif isinstance(oltc_model, object):
            self.oltc = oltc_model
        return

    def set_state_vector(self, state) -> None:
        """
        Set the current state for the OLTC transformer.

        The state is forwarded to the attached OLTC controller instance.

        Args:
            state (dict): State dictionary accepted by the OLTC controller.

        Raises:
            AttributeError: If the OLTC controller does not implement
                ``set_state_vector``.
        """
        try:
            self.oltc.set_state_vector(state)
        except Exception as exs:
            raise AttributeError(
                "No state vector for the OLTC controller is defined."
            ) from exs

    def get_state_vector(self):
        """
        Return the current state of the OLTC transformer.

        Returns:
            object: Controller-specific state (for example tap position).

        Raises:
            AttributeError: If the OLTC controller does not implement
                ``get_state_vector``.
        """
        # The actual state is defined with the position of the tap changer
        try:
            return self.oltc.get_state_vector()
        except Exception as exs:
            raise AttributeError(
                "No state vector for the OLTC controller is defined."
            ) from exs

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulation mode for the OLTC transformer and its controller.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        super().enable_parallel_simulation(parallel_sims)
        self.oltc.enable_parallel_simulation(parallel_sims)

    def get_value(self, value, angle_type="deg"):
        """
        Return derived quantities for the OLTC transformer.

        Delegates to the base class implementation. See
        :py:meth:`Transformer.get_value` for details.
        """
        return super().get_value(value, angle_type)

    def initialize(self):
        """
        Initialize the OLTC transformer and its controller.

        The measurement bus is used to obtain a pre-control voltage which is
        passed to the OLTC controller's ``initialize`` method.
        """
        if self.measure_bus == self.from_bus:
            v_pre = self.sim.busses[self.sim.bus_idxs[self.from_bus_name]].voltage
        else:
            v_pre = self.sim.busses[self.sim.bus_idxs[self.to_bus_name]].voltage

        self.oltc.initialize(v_pre)
        return
