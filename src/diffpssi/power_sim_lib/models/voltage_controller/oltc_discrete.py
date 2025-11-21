"""Implementation of a discrete OLTC voltage controller model."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, PT1Limited
from diffpssi.power_sim_lib.models.voltage_controller.voltage_controller_interface import (
    Voltage_Controller,
)


class OLTC_Discrete(Voltage_Controller):
    """
    Discrete OLTC controller (single-module).

    The controller implements a discrete on-load tap changer with a deadband
    and a single time constant. It may only switch by one tap step per
    switching operation (adjacent tap switching).

    Args:
        N/A: See :py:meth:`__init__` for constructor arguments.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
        parallel_sims=1,
        t_1=5,
        db=0.05,
        delta_m=0.02,
        m_max=1.1,
        m_min=0.9,
        v_ref=1,
    ):
        """
        Initialize a discrete single-module OLTC controller.

        Args:
            name (str, optional): Controller name. Defaults to None.
            trafo (object, optional): Associated transformer instance. Defaults to None.
            param_dict (dict, optional): Parameter dictionary. If omitted the
                individual keyword arguments are used to build a default
                parameter set. Expected keys: ``t_1``, ``db``, ``delta_m``,
                ``m_max``, ``m_min``, ``v_ref``.
            parallel_sims (int): Number of parallel simulations (default: 1).
            t_1 (float): Integrator limiter / switching time window (default: 5).
            db (float): Deadband width (default: 0.05).
            delta_m (float): Tap step magnitude (default: 0.02).
            m_max (float): Maximum tap multiplier (default: 1.1).
            m_min (float): Minimum tap multiplier (default: 0.9).
            v_ref (float): Reference voltage (default: 1).

        Returns:
            None
        """
        if param_dict is None:
            param_dict = {
                "t_1": t_1,
                "db": db,
                "delta_m": delta_m,
                "m_max": m_max,
                "m_min": m_min,
                "v_ref": v_ref,
            }

        self.name = name
        self.trafo = trafo

        self.db = param_dict["db"]  # Dead band range
        self.delta_m = param_dict["delta_m"]  # Tap changing rate
        self.m_max = param_dict["m_max"]
        self.m_min = param_dict["m_min"]
        self.t_1 = param_dict["t_1"]
        self.pt1_const = param_dict.get(
            "pt_1", 0
        )  # time constant for measurement filter

        # Was ist mit nicht passenden Zahlen, sodass nicht integer werte in der Rechnung auftreten?
        self.n_taps = int((self.m_max - self.m_min) / self.delta_m)  # number of taps
        self.taps = torch.arange(0, self.n_taps, 1)  # tap positions
        self.tap_pos_m = param_dict.get(
            "tap_pos_m", int(self.n_taps / 2)
        )  # initial tap position
        self.u_l = self.m_min + self.tap_pos_m * self.delta_m  # initial oltc ratio
        self.m = 0  # tap change signal
        self.v_diff = 0
        self.v_dead = 0
        self.v_measure = 0
        self.integ = 0
        self.dir = 1

        if self.pt1_const != 0:
            self.pt1 = PT1Limited(t_pt1=self.pt1_const, gain_pt1=1)
        else:
            self.pt1 = None
        self.deadband = DeadBand(self.db)
        self.integrator_m = Integrator(k_i=1, limiter=t_1)

        self.v_ref = param_dict[
            "v_ref"
        ]  # reference voltage for the controller; typically set at the beginning load flow analysis
        self.parallel_sims = parallel_sims

    def get_state_vector(self):
        """
        Return the controller's internal state vector.

        For the discrete OLTC this is the integrator state used to time the
        switching operation.

        Returns:
            torch.Tensor: State vector shaped (parallel_sims, n_states).
        """
        return (
            self.integrator_m.get_state_vector()
        )  # torch.stack([self.integrator.get_state_vector(), ], axis=1)

    def set_state_vector(self, x):
        """
        Set the controller's internal state vector.

        Args:
            x (torch.Tensor): State vector in the same layout as returned by
                :py:meth:`get_state_vector`.

        Returns:
            None
        """
        self.integrator_m.set_state_vector(x)

    def get_output(self, vbb=None):
        """
        Compute the controller output (tap ratio) for a given measured voltage.

        The method optionally filters the measurement through a PT1 block,
        applies deadband logic and integrator timing to determine if a
        switching operation must be performed.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: New tap ratio ``u_l`` after potential switching.
        """
        # Add possible filter for the measurements... -> PT1 block
        if self.pt1 is not None:
            self.v_measure = self.pt1.get_output(torch.abs(vbb))
        else:
            self.v_measure = torch.abs(vbb)

        self.v_diff = torch.abs(self.v_ref) - torch.abs(
            vbb
        )  # * torch.ones((self.parallel_sims, 1), dtype=torch.float64)

        v_dead_prev = self.v_dead
        self.v_dead = self.deadband.get_output(self.v_diff)

        # reset the integrator, if the v_diff falls under the deadband (could as well be == 0)
        # OR if the sign suddenly changing
        if torch.abs(self.v_dead) == torch.zeros_like(self.v_dead):
            self.integrator_m.reset()
        elif torch.sign(self.v_dead) != torch.sign(v_dead_prev):
            self.integrator_m.reset()

        self.integ = self.integrator_m.get_output(
            torch.abs(torch.sign(self.v_diff))
        )  # * torch.ones((self.parallel_sims, 1))) # or v_dead/self.v_diff for variable gain

        if self.integ > self.t_1:
            # reset the integrator after the switching operation mandatory for remaining operatbility of t_1 s window
            # if the deadband is continuously overstepped for more than t_1 s, switching operation is triggered
            self.m = self.switching(self.dir * self.v_diff)
            self.integrator_m.reset()
            self.u_l += self.delta_m * self.m

        return self.u_l

    def differential(self):
        """
        Return derivative(s) of the controller internal dynamic blocks.

        Returns:
            torch.Tensor: Concatenated derivative vector from internal blocks
            (e.g. integrator).

        Raises:
            NotImplementedError: If an internal dynamic block does not expose
                a differential implementation.
        """
        try:
            deriv_vec = torch.concatenate(
                [
                    self.integrator_m.differential(),
                ],
                axis=1,
            )
        except Exception:
            raise NotImplementedError("Differential not implemented for integrator.")

        return deriv_vec

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable batched/parallel execution for the controller.

        Converts scalar attributes to tensors sized by ``parallel_sims`` and
        configures internal sub-blocks for batched operation.

        Args:
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        self.integrator_m.enable_parallel_simulation(parallel_sims)
        self.deadband.enable_parallel_simulation(parallel_sims)

        self.db = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.db
        self.delta_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.delta_m
        )
        self.m_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_max
        self.m_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_min
        self.n_taps = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.n_taps
        self.tap_pos_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.tap_pos_m
        )
        self.t_1 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_1
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref
        self.v_measure = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_measure
        )
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l

    def initialize(self, voltage):
        """
        Initialize controller internal state using a reference/measured voltage.

        Args:
            voltage (torch.Tensor or float): Voltage to be used as initial
                reference and measurement.

        Returns:
            None
        """
        self.v_ref = voltage
        self.v_measure = voltage
        self.integrator_m.initialize(
            torch.zeros((self.parallel_sims, 1), dtype=torch.float64)
        )
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

    def switching(self, v_diff):
        """
        Determine discrete switching action based on the voltage difference.

        The method updates the tap position and returns the tap change ``m``
        which is applied to the tap ratio.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation from
                reference used to decide switching direction.

        Returns:
            torch.Tensor: Tap change step(s) applied (per parallel simulation).
        """
        if v_diff > 0 and self.tap_pos_m > self.taps[0]:
            m = -1 * torch.ones((self.parallel_sims, 1), dtype=torch.float64)
            self.tap_pos_m -= torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        elif v_diff < 0 and self.tap_pos_m <= self.taps[-1]:
            m = 1 * torch.ones((self.parallel_sims, 1), dtype=torch.float64)
            self.tap_pos_m += torch.ones((self.parallel_sims, 1), dtype=torch.float64)

        else:
            m = torch.zeros((self.parallel_sims, 1), dtype=torch.float64)

        return m
