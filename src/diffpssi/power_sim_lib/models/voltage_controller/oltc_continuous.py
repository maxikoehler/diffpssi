"""Implements a continuous OLTC voltage controller model."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import (
    DeadBand,
    Integrator,
    Lag,
    Limiter,
    PIController,
    PT1Limited,
)
from diffpssi.power_sim_lib.models.voltage_controller.voltage_controller_interface import (
    Voltage_Controller,
)


class OLTC_Continuous(Voltage_Controller):
    """
    Continuous OLTC controller.

    Uses a PT1Limited measurement/filter block and (optionally) a deadband to
    compute a continuous tap ratio output. Suitable for electronic or
    continuously actuated OLTC models.

    Args:
        N/A: See :py:meth:`__init__` for constructor parameters.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
    ):
        """
        Initialize a continuous OLTC controller.

        Args:
            name (str, optional): Controller name.
            trafo (object, optional): Associated transformer instance.
            param_dict (dict, optional): Parameter dictionary. Expected keys
                include ``db``, ``m_max``, ``m_min``, ``v_ref``, and ``dir``.
            t_1 (float): Time constant used for PT1 filter / switching duration.
            parallel_sims (int): Number of parallel simulations.
            db (float, optional): Deadband value. If None no deadband is used.

        Returns:
            None
        """
        self.name = name
        self.trafo = trafo

        self.m_max = param_dict.get("m_max", 1.1)
        self.m_min = param_dict.get("m_min", 0.9)

        self.u_l = 1.0  # initial oltc ratio
        self.m = 0  # tap change signal
        self.dir = param_dict.get("dir", -1)

        self.t_1 = param_dict.get("t_1", 5)
        self.gain_pt1 = param_dict.get("gain_pt1", 1)

        self.pt1 = PT1Limited(
            t_pt1=self.t_1,
            gain_pt1=self.gain_pt1,
            lim_min=self.m_min,
            lim_max=self.m_max,
        )

        if param_dict.get("db", None) != None:
            self.deadband = DeadBand(param_dict.get("db"))
        else:
            self.deadband = None

        self.error = 0  # 1e-2
        # reference voltage for the controller;
        # typically set at the beginning load flow analysis
        self.v_ref = param_dict.get("v_ref", 1.0)
        self.parallel_sims = param_dict.get("parallel_sims", 1)

    def get_state_vector(self):
        """
        Return the PT1 filter's internal state vector.

        Returns:
            torch.Tensor: PT1 state vector shaped (parallel_sims, n_states).
        """
        return self.pt1.get_state_vector()

    def set_state_vector(self, x):
        """
        Set the PT1 filter's internal state vector.

        Args:
            x (torch.Tensor): State vector matching the layout returned by
                :py:meth:`get_state_vector`.

        Returns:
            None
        """
        self.pt1.set_state_vector(x)

    def get_output(self, vbb=None):
        """
        Compute the continuous controller output for a measured voltage.

        The result is produced by the PT1 block with optional deadband
        processing. Small errors below ``self.error`` are treated as zero.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: Continuous tap ratio output ``u_l``.
        """
        self.v_diff = torch.abs(self.v_ref) - torch.abs(vbb)

        u = self.pt1.get_output(self.v_ref + self.dir * self.v_diff)

        if self.deadband != None:
            self.u_l = self.deadband.get_output(u)
        else:
            self.u_l = u

        return self.u_l

    def differential(self):
        """
        Return derivative vector for dynamic PT1 block.

        Returns:
            torch.Tensor: PT1 differential vector.

        Raises:
            NotImplementedError: If PT1 differential is not implemented.
        """
        try:
            deriv_vec = torch.concatenate(
                [
                    self.pt1.differential(),
                ],
                axis=1,
            )
        except Exception:
            raise NotImplementedError("Differential not implemented for integrator.")

        return deriv_vec

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel/batched execution for the continuous OLTC controller.

        Converts scalars to per-simulation tensors and configures PT1 for
        batched operation.

        Args:
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        if self.deadband != None:
            self.deadband.enable_parallel_simulation(parallel_sims)
        self.pt1.enable_parallel_simulation(parallel_sims)
        self.m_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_max
        self.m_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_min
        self.m = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l
        # self.dir = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.dir
        # self.error = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.error
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref

    def initialize(self, voltage):
        """
        Initialize the continuous OLTC controller state using an initial voltage.

        Args:
            voltage (torch.Tensor or float): Initial measured/reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        self.pt1.initialize(self.v_ref)
        return

    def update_vref(self, voltage):
        """
        Update the controller reference voltage.

        Args:
            voltage (torch.Tensor or float): New reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        return

    def reset(self):
        """
        Reset the OLTC controller to its initial state.

        Returns:
            None
        """
        self.pt1.reset()


class OLTC_Continuous_Milano(Voltage_Controller):
    """
    Continuous OLTC controller following Milano's formulation.

    Uses a lag transfer block to compute a continuous tap
    ratio output. Suitable for electronic or continuously actuated OLTC models.

    Args:
        N/A: See :py:meth:`__init__` for constructor parameters.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
    ):
        """
        Initialize a continuous OLTC controller.

        Args:
            name (str, optional): Controller name.
            trafo (object, optional): Associated transformer instance.
            param_dict (dict, optional): Parameter dictionary. Expected keys
                include ``m_max``, ``m_min``, ``v_ref``, and ``dir``.
            t_1 (float): Time constant used for lag filter / switching duration.
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        self.name = name
        self.trafo = trafo

        self.m_max = param_dict.get("m_max", 1.1)
        self.m_min = param_dict.get("m_min", 0.9)

        self.u_l = 1.0  # initial oltc ratio
        self.dir = param_dict.get("dir", 1)

        self.kd = param_dict.get("kd", 1)
        self.ki = param_dict.get("ki", 1)

        self.lag = Lag(
            kd=self.kd,
            ki=self.ki,
            lim_min=self.m_min,
            lim_max=self.m_max,
        )

        if param_dict.get("db", None) != None:
            self.deadband = DeadBand(param_dict.get("db"))
        else:
            self.deadband = None

        # reference voltage for the controller;
        # typically set at the beginning load flow analysis
        self.v_ref = param_dict.get("v_ref", 1.0)
        self.parallel_sims = param_dict.get("parallel_sims", 1)

    def get_state_vector(self):
        """
        Return the PT1 filter's internal state vector.

        Returns:
            torch.Tensor: PT1 state vector shaped (parallel_sims, n_states).
        """
        return self.lag.get_state_vector()

    def set_state_vector(self, x):
        """
        Set the PT1 filter's internal state vector.

        Args:
            x (torch.Tensor): State vector matching the layout returned by
                :py:meth:`get_state_vector`.

        Returns:
            None
        """
        self.lag.set_state_vector(x)

    def get_output(self, vbb=None):
        """
        Compute the continuous controller output for a measured voltage.

        The result is produced by the PT1 block with optional deadband
        processing. Small errors below ``self.error`` are treated as zero.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: Continuous tap ratio output ``u_l``.
        """
        self.v_diff = torch.abs(self.v_ref) - torch.abs(vbb)

        u = self.lag.get_output(self.dir * self.v_diff)

        # discretizing
        # u = torch.floor_divide(u, 0.02) * 0.02

        if self.deadband != None:
            self.u_l = self.deadband.get_output(u)
        else:
            self.u_l = u

        return self.u_l

    def differential(self):
        """
        Return derivative vector for dynamic PT1 block.

        Returns:
            torch.Tensor: PT1 differential vector.

        Raises:
            NotImplementedError: If PT1 differential is not implemented.
        """
        try:
            deriv_vec = torch.concatenate(
                [
                    self.lag.differential(),
                ],
                axis=1,
            )
        except Exception:
            raise NotImplementedError("Differential not implemented for integrator.")

        return deriv_vec

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel/batched execution for the continuous OLTC controller.

        Converts scalars to per-simulation tensors and configures PT1 for
        batched operation.

        Args:
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        if self.deadband != None:
            self.deadband.enable_parallel_simulation(parallel_sims)
        self.lag.enable_parallel_simulation(parallel_sims)
        self.m_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_max
        self.m_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_min
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l
        self.dir = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.dir
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref

    def initialize(self, voltage):
        """
        Initialize the continuous OLTC controller state using an initial voltage.

        Args:
            voltage (torch.Tensor or float): Initial measured/reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        self.lag.initialize(self.v_ref)
        return

    def update_vref(self, voltage):
        """
        Update the controller reference voltage.

        Args:
            voltage (torch.Tensor or float): New reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        return

    def reset(self):
        """
        Reset the OLTC controller to its initial state.

        Returns:
            None
        """
        tensor_size = torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        self.u_l = 1.0 * tensor_size
        self.lag.reset()


class OLTC_Continuous_PI(Voltage_Controller):
    """
    Continuous OLTC controller with PI control.

    Uses an integrator block to compute a continuous tap
    ratio output. Suitable for electronic or continuously actuated OLTC models.

    Args:
        N/A: See :py:meth:`__init__` for constructor parameters.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
    ):
        """
        Initialize a continuous OLTC controller.

        Args:
            name (str, optional): Controller name.
            trafo (object, optional): Associated transformer instance.
            param_dict (dict, optional): Parameter dictionary. Expected keys
                include ``m_max``, ``m_min``, ``v_ref``, and ``dir``.
            t_1 (float): Time constant used for lag filter / switching duration.
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        self.name = name
        self.trafo = trafo

        self.lim = param_dict.get("lim", 0.1)

        self.u_l = 1.0  # initial oltc ratio
        self.dir = param_dict.get("dir", -1)

        self.kp = param_dict.get("kp", 0)
        self.ki = param_dict.get("ki", 0.5)

        self.pi = PIController(
            k_p=self.kp,
            k_i=self.ki,
        )

        self.limiter = Limiter(
            limit=self.lim,
        )

        if param_dict.get("db", None) != None:
            self.deadband = DeadBand(param_dict.get("db"))
        else:
            self.deadband = None

        # reference voltage for the controller;
        # typically set at the beginning load flow analysis
        self.v_ref = param_dict.get("v_ref", 1.0)
        self.parallel_sims = param_dict.get("parallel_sims", 1)

    def get_state_vector(self):
        """
        Return the PT1 filter's internal state vector.

        Returns:
            torch.Tensor: PT1 state vector shaped (parallel_sims, n_states).
        """
        return self.pi.get_state_vector()

    def set_state_vector(self, x):
        """
        Set the PT1 filter's internal state vector.

        Args:
            x (torch.Tensor): State vector matching the layout returned by
                :py:meth:`get_state_vector`.

        Returns:
            None
        """
        self.pi.set_state_vector(x)

    def get_output(self, vbb=None):
        """
        Compute the continuous controller output for a measured voltage.

        The result is produced by the PT1 block with optional deadband
        processing. Small errors below ``self.error`` are treated as zero.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: Continuous tap ratio output ``u_l``.
        """
        self.v_diff = torch.abs(self.v_ref) - torch.abs(vbb)

        u = self.pi.get_output(self.dir * self.v_diff)

        if self.deadband != None:
            self.u_l = self.deadband.get_output(u)
        else:
            self.u_l = u

        return self.u_l

    def differential(self):
        """
        Return derivative vector for dynamic PT1 block.

        Returns:
            torch.Tensor: PT1 differential vector.

        Raises:
            NotImplementedError: If PT1 differential is not implemented.
        """
        try:
            deriv_vec = torch.concatenate(
                [
                    self.pi.differential(),
                ],
                axis=1,
            )
        except Exception:
            raise NotImplementedError("Differential not implemented for integrator.")

        return deriv_vec

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel/batched execution for the continuous OLTC controller.

        Converts scalars to per-simulation tensors and configures PT1 for
        batched operation.

        Args:
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        if self.deadband != None:
            self.deadband.enable_parallel_simulation(parallel_sims)
        self.limiter.enable_parallel_simulation(parallel_sims)
        self.lim = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim
        self.pi.enable_parallel_simulation(parallel_sims)
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l
        self.dir = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.dir
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref

    def initialize(self, voltage):
        """
        Initialize the continuous OLTC controller state using an initial voltage.

        Args:
            voltage (torch.Tensor or float): Initial measured/reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        self.pi.initialize(self.v_ref)
        return

    def update_vref(self, voltage):
        """
        Update the controller reference voltage.

        Args:
            voltage (torch.Tensor or float): New reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        return

    def reset(self):
        """
        Reset the OLTC controller to its initial state.

        Returns:
            None
        """
        tensor_size = torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        self.u_l = 1.0 * tensor_size
        self.pi.reset()
