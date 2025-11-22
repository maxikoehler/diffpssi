"""Transformer interface model implementation."""

import logging
from abc import ABC, abstractmethod

from diffpssi.power_sim_lib.backend import *

_logger = logging.getLogger(__name__)


class Transformer(ABC):
    """
    Interface class for different transformer types.

    The abstract base class encapsulates common transformer parameters and
    electrical computations such as admittance matrix calculation and current
    injections. Subclasses implement or extend behavior for specific
    transformer/controller combinations (e.g., simple transformer, OLTC).

    Args:
        N/A: Instances are created via concrete subclasses.
    """

    def __init__(
        self, s_n_sys, param_dict, parallel_sims, sim, trans_model="AM", name=None
    ):
        """
        Initialize base transformer parameters.

        The constructor extracts electrical and connection parameters from
        ``param_dict`` and computes per-unit impedance/admittance scaled to the
        system base power ``s_n_sys``.

        Args:
            s_n_sys (float): System base apparent power used for per-unit scaling.
            param_dict (dict): Dictionary containing transformer parameters. Expected
                keys include: ``name``, ``S_n``, ``R``, ``X``, ``V_n_from``, ``V_n_to``,
                ``from_bus``, ``to_bus`` and optional keys ``B``, ``theta``, ``u_l``,
                ``tap_side``, ``measure_side``.
            trans_model (str): Transformer model identifier (default: "AM").
            parallel_sims (int): Number of parallel simulations (default: 1). When
                enabled numerical attributes are converted to tensors sized for
                parallel execution.

        Returns:
            None: Initializes instance attributes in-place.
        """
        # Basic characteristics and relations
        self.name = param_dict["name"]
        self.s_n = param_dict["S_n"]
        self.s_n_sys = s_n_sys
        self.r = param_dict["R"]
        self.x = param_dict["X"]
        self.b = param_dict.get("B", 0)

        self.z_t = (self.r + 1j * self.x) * self.s_n_sys / self.s_n
        self.y_t = 1 / self.z_t

        self.from_bus_name = param_dict["from_bus"]
        self.to_bus_name = param_dict["to_bus"]
        self.from_bus_id = None
        self.to_bus_id = None
        self.v_n_from = param_dict["V_n_from"]
        self.v_n_to = param_dict["V_n_to"]
        self.from_voltage = None
        self.to_voltage = None

        if self.v_n_from > self.v_n_to:
            self.from_bus = "hv"
        elif self.v_n_from <= self.v_n_to:
            self.from_bus = "lv"

        self.tap_side = param_dict.get("tap_side", "hv")
        self.measure_bus = param_dict.get("measure_side", "hv")

        self.theta = param_dict.get("theta", 0)
        self.u_l = param_dict.get("u_l", 1)
        self.admittance_matrix = torch.zeros(
            (parallel_sims, 2, 2), dtype=torch.complex128
        )
        self.trans_model = trans_model
        self.parallel_sims = parallel_sims

    def calc_admittance(self, return_need) -> None:
        """
        Compute and set the transformer's two-port admittance matrix.

        The admittance is built from the transformer's series impedance and the
        current tap/phase ratio. The method sets ``self.admittance_matrix`` and
        optionally returns it when ``return_need`` is True.

        Args:
            return_need (bool): If True the method returns the admittance
                matrix in addition to setting ``self.admittance_matrix``.

        Returns:
            torch.Tensor or None: The 2x2 admittance matrix if ``return_need`` is
            True, otherwise None.
        """
        self.u = self.u_l * torch.exp(-1j * (self.theta / 360) * 2 * torch.pi)

        # Model on the HV side; from LV to HV
        # This should lead to the exact same result; -> Just angle shifting?
        y_11 = self.y_t * (1 / (torch.conj(self.u) * self.u))
        y_12 = -self.y_t * (1 / torch.conj(self.u))
        y_21 = -self.y_t * (1 / self.u)
        y_22 = self.y_t

        # setting the new admittance matrix in correct bus relation
        # This determines, where the ratio theta is located
        if self.tap_side == self.from_bus:
            self.admittance_matrix[:, 0, 0] = y_11.reshape(1, -1)
            self.admittance_matrix[:, 0, 1] = y_12.reshape(1, -1)
            self.admittance_matrix[:, 1, 0] = y_21.reshape(1, -1)
            self.admittance_matrix[:, 1, 1] = y_22.reshape(1, -1)
            # self.admittance_matrix = torch.tensor([[y_11, y_12], [y_21, y_22]])
        elif self.tap_side != self.from_bus:
            self.admittance_matrix[:, 0, 0] = y_22.reshape(1, -1)
            self.admittance_matrix[:, 0, 1] = y_21.reshape(1, -1)
            self.admittance_matrix[:, 1, 0] = y_12.reshape(1, -1)
            self.admittance_matrix[:, 1, 1] = y_11.reshape(1, -1)

        # if a return of the function is needed, return the y_matrix as well
        if return_need:
            return self.admittance_matrix

    def calc_admittance_static(self, return_need) -> None:
        """
        Compute the static admittance matrix.

        This is an alias to :py:meth:`calc_admittance` kept for API symmetry with
        static model implementations.

        Args:
            return_need (bool): If True return the computed admittance matrix.

        Returns:
            torch.Tensor or None: See :py:meth:`calc_admittance`.
        """
        return self.calc_admittance(return_need)

    def calc_current_injections(self):
        """
        Compute current injection from the transformer's shunt branch.

        The implementation depends on the selected ``trans_model``. For the
        default analytical model ("AM") no shunt injection is present. The
        returned current is scaled to the system base power.

        Args:
            None

        Returns:
            torch.Tensor: Current injection scaled to the system base.
        """
        if self.trans_model == "AM":
            i_inj = 0
        elif self.trans_model == "CIM":
            # currently not represented
            u = self.factor * torch.exp(1j * self.theta)
            i_inj = 0
            _logger.warning("CIM model not yet implemented, defaulting to 0 injection.")
        else:
            raise ValueError("Invalid transformer model selected.")

        # transform it to the base system. sn/vn is local per unit and
        # sys_s_n/sys_v_n is global per unit
        i_inj = i_inj * (self.s_n / self.s_n_sys)
        return i_inj

    def enable_parallel_simulation(self, parallel_sims):
        """
        Convert scalar attributes to tensors for parallel simulations.

        This method replaces numeric scalar attributes (IDs, base values,
        impedances, taps, etc.) with tensors sized for ``parallel_sims``.

        Args:
            parallel_sims (int): Number of parallel simulations to enable.

        Returns:
            None: Attributes are modified in-place.
        """
        self.s_n = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.s_n
        self.s_n_sys = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.s_n_sys
        )
        self.r = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.r
        self.x = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.x
        self.b = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.b
        self.v_n_from = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_from
        )
        self.v_n_to = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_to
        self.theta = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.theta
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l
        self.z_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.z_t
        self.y_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.y_t

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Return selected derived quantities for the transformer.

        Supported ``value`` strings are ``'current'`` and ``'voltage'``. The
        method requires a simulation object to access bus voltages.

        Args:
            value (str): Name of the requested quantity. Must be either
                ``'current'`` or ``'voltage'``.
            sim (object): Simulation instance providing bus objects and indices.
            angle_type (str): Angle representation for returned values (unused
                by current implementation, default: "deg").

        Returns:
            torch.Tensor: Requested per-simulation array for the chosen quantity.

        Raises:
            ValueError: If ``sim`` is None or ``value`` is not supported.
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
        elif value == "voltage":
            return voltage
        else:
            raise ValueError("Invalid value parameter")

    def initialize(self):
        """
        Perform any transformer-specific initialization prior to simulation.

        Subclasses may override this method to initialize controller state,
        compute derived values, or perform other setup steps.

        Args:
            None

        Returns:
            None
        """
        pass

    def differential(self):
        """
        Return the time-derivative vector for dynamic states.

        Default implementation returns 0 and is expected to be overridden by
        subclasses that expose dynamic states (for example OLTC controllers).

        Returns:
            int or torch.Tensor: Zero or a tensor containing derivatives.
        """
        return 0

    def get_state_vector(self):
        """
        Return the current state vector for the transformer.

        The default implementation returns 0. Subclasses that have internal
        dynamic state (e.g., tap positions) should override this method.

        Returns:
            object: Current state representation (type depends on subclass).
        """
        return 0

    def set_state_vector(self, x=None):
        """
        Set the transformer's state vector.

        Subclasses with mutable internal state (for example an OLTC controller)
        should implement this to accept an appropriate state representation.

        Args:
            x (optional): New state vector or dictionary representing the state.

        Returns:
            None
        """
        pass
