"""Implements a continuous OLTC voltage controller model."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, PT1Limited
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
        t_1=1,
        parallel_sims=1,
        db=None,
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
        if param_dict is None:
            param_dict = {
                "db": 0.05,
                "m_max": 1.1,
                "m_min": 0.9,
                "v_ref": 1,
                "dir": 1,
            }

        self.name = name
        self.trafo = trafo

        self.m_max = param_dict["m_max"]
        self.m_min = param_dict["m_min"]

        # Was ist mit nicht passenden Zahlen, sodass nicht integer werte in der Rechnung auftreten?
        self.u_l = 1.0  # initial oltc ratio
        self.m = 0  # tap change signal
        self.dir = param_dict.get("dir", 1)

        if t_1 is None:
            self.t_1 = 5  # time constant for the integrator: duration of one switching operation
        else:
            self.t_1 = t_1

        self.pt1 = PT1Limited(
            t_pt1=self.t_1, gain_pt1=1, lim_min=self.m_min, lim_max=self.m_max
        )

        if db != None:
            self.deadband = DeadBand(db)
        else:
            self.deadband = None

        self.error = 1e-2
        self.v_ref = param_dict[
            "v_ref"
        ]  # reference voltage for the controller; typically set at the beginning load flow analysis
        self.parallel_sims = parallel_sims

    def get_state_vector(self):
        """
        Return the PT1 filter's internal state vector.

        Returns:
            torch.Tensor: PT1 state vector shaped (parallel_sims, n_states).
        """
        return (
            self.pt1.get_state_vector()
        )  # torch.stack([self.integrator.get_state_vector(), ], axis=1)

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
        self.v_diff = torch.abs(self.v_ref) - torch.abs(
            vbb
        )  # normally magnitude instead of real part
        if self.deadband != None:
            self.v_diff = self.deadband.get_output(self.v_diff)

        if torch.abs(self.v_diff) < self.error:
            self.v_diff = 0 * self.v_diff

        self.u_l = self.pt1.get_output(self.v_ref + self.dir * 3 * self.v_diff)

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
        self.pt1.enable_parallel_simulation(parallel_sims)
        self.m_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_max
        self.m_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_min
        self.m = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l
        # self.n_taps = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.n_taps
        # self.taps = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.taps
        # self.tap_pos = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.tap_pos
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref

    def initialize(self, voltage):
        """
        Initialize the continuous OLTC controller state using an initial voltage.

        Args:
            voltage (torch.Tensor or float): Initial measured/reference voltage.

        Returns:
            None
        """
        self.v_ref = torch.abs(voltage)
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
