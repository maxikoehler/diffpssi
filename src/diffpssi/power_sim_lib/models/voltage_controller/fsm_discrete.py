"""Voltage controller classes for power system simulations."""

import numpy as np

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, PT1Limited
from diffpssi.power_sim_lib.models.voltage_controller.voltage_controller_interface import (
    Voltage_Controller,
)


class FSM_Discrete(Voltage_Controller):
    """
    Finite-state machine (FSM) discrete voltage controller.

    Combines a finite-state machine (FSM) logic and an OLTC-like module to
    implement tap-changing behavior with separate m/k branches. The controller
    supports deadband, integrators and optional measurement filtering.

    Args:
        N/A: See :py:meth:`__init__` for constructor arguments.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
        parallel_sims=1,
        t_m=0.2,
        t_k=5,
        db=0.025,
        delta_m=2,
        delta_k=0.02,
        m_max=4,
        m_min=-4,
        k_max=5,
        k_min=-5,
        gamma=4,
        v_ref=1,
    ):
        """
        Initialize an FSM_Discrete controller.

        Args:
            name (str, optional): Controller name.
            trafo (object, optional): Associated transformer instance.
            param_dict (dict, optional): Parameter dictionary. If omitted the
                provided defaults are used. Expected keys include timing,
                deadband and tap ranges for both FSM and OLTC modules.
            parallel_sims (int): Number of parallel simulations.
            t_m (float): Time constant for m-branch integrator.
            t_k (float): Time constant for k-branch integrator.
            db (float): Deadband width.
            delta_m (float): m-branch tap step.
            delta_k (float): k-branch tap step.
            m_max, m_min, k_max, k_min (int/float): Tap limits.
            gamma (int): Maximum tap-skips parameter.
            v_ref (float): Reference voltage.

        Returns:
            None
        """
        if param_dict is None:
            param_dict = {
                "t_m": t_m,
                "t_k": t_k,
                "db": db,
                "delta_m": delta_m,
                "delta_k": delta_k,
                "m_max": m_max,
                "m_min": m_min,
                "k_max": k_max,
                "k_min": k_min,
                "v_ref": v_ref,
                "gamma_max": gamma,
            }

        self.name = name
        self.trafo = trafo

        self.db = param_dict["db"]  # Dead band range
        self.v_ref = param_dict[
            "v_ref"
        ]  # reference voltage for the controller; typically set at the beginning load flow analysis
        self.ul_max = 1.1
        self.ul_min = 0.9
        self.pt1_const = param_dict.get(
            "pt_1", 0
        )  # time constant for measurement filter
        self.v_measure = self.v_ref

        # Parameters in m; FSM module
        self.delta_m = param_dict["delta_m"]  # Tap changing rate
        self.m_max = param_dict["m_max"]
        self.m_min = param_dict["m_min"]
        self.t_m = param_dict["t_m"]
        self.n_taps_m = int(
            (self.m_max - self.m_min)
        )  # / self.delta_m) # number of taps in m; OLTC module
        self.taps_m = torch.arange(
            self.m_min, self.m_max + 1, 1
        )  # tap positions in m; OLTC module
        self.tap_pos_m = (
            0  # int(self.n_taps_m / 2) # initial tap position in m; OLTC module
        )

        # Parameters in k; OLTC module
        self.delta_k = param_dict["delta_k"]  # Tap changing rate
        self.k_max = param_dict["k_max"]
        self.k_min = param_dict["k_min"]
        self.t_k = param_dict["t_k"]
        self.n_taps_k = int(
            (self.k_max - self.k_min)
        )  # / self.delta_k) # number of taps in k; FSM module
        self.taps_k = torch.arange(
            self.k_min, self.k_max + 1, 1
        )  # tap positions in k; FSM module
        self.tap_pos_k = (
            0  # int(self.n_taps_k / 2) # initial tap position in k; FSM module
        )

        self.gamma = gamma
        self.dir = 1

        self.u_l = 1 + self.delta_k * (
            self.tap_pos_k + self.tap_pos_m * self.delta_m
        )  # initial oltc ratio

        # initials / state variables
        self.m = 0  # tap change signal in m; OLTC module
        self.k = 0  # tap change signal in k; FSM module
        self.v_diff = 0
        self.v_dead = 0
        self.integ_m = 0
        self.integ_k = 0

        self.e = 0
        self.e_k = 0
        self.eta = 0

        if self.pt1_const != 0:
            self.pt1 = PT1Limited(
                t_pt1=self.pt1_const, gain_pt1=1, lim_min=0, lim_max=1
            )
        else:
            self.pt1 = None
        self.deadband = DeadBand(self.db)
        self.integrator_m = Integrator(k_i=1 / self.t_m, limiter=None)
        self.integrator_k = Integrator(k_i=1 / self.t_k, limiter=None)

        self.parallel_sims = parallel_sims

    def get_state_vector(self):
        """
        Return the FSM_Discrete controller's internal state vector.

        The returned vector concatenates integrator states and optional PT1
        filter states in a consistent order used by :py:meth:`set_state_vector`.

        Returns:
            torch.Tensor: State vector shaped (parallel_sims, n_states).
        """
        if self.pt1 is not None:
            return torch.concatenate(
                [
                    self.integrator_m.get_state_vector(),
                    self.integrator_k.get_state_vector(),
                    self.pt1.get_state_vector(),
                ],
                axis=1,
            )
        else:
            return torch.concatenate(
                [
                    self.integrator_m.get_state_vector(),
                    self.integrator_k.get_state_vector(),
                ],
                axis=1,
            )

    def set_state_vector(self, x):
        """
        Restore internal states from a provided state vector.

        Args:
            x (torch.Tensor): State vector with columns matching the order from
                :py:meth:`get_state_vector`.

        Returns:
            None
        """
        self.integrator_m.set_state_vector(x[:, 0:1])
        self.integrator_k.set_state_vector(x[:, 1:2])
        if self.pt1 is not None:
            self.pt1.set_state_vector(x[:, 2:3])

    def get_output(self, vbb=None):
        """
        Compute the controller output (tap ratio) for the FSM_Discrete controller.

        The method filters the measured voltage (optional PT1), applies the
        deadband, integrates the error, evaluates tap-skips and performs
        discrete switching actions when integrator thresholds are exceeded.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: Tap ratio ``u_l`` after any performed switching.
        """
        # Add possible filter for the measurements... -> PT1 block
        if self.pt1 is not None:
            self.v_measure = self.pt1.get_output(torch.abs(vbb))
        else:
            self.v_measure = torch.abs(vbb)

        self.v_diff = (
            torch.abs(self.v_ref) - self.v_measure
        )  # * torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        self.e = torch.sign(self.deadband.get_output(self.v_diff))
        self.e_k = self.e

        # If one wants to favor the FSM
        # if self.tap_pos_m in [self.taps_m[0], self.taps_m[-1]]:
        #     self.e_k = self.e
        # else:
        #     self.e_k = torch.zeros((self.parallel_sims, 1))

        self.integ_m = self.integrator_m.get_output(self.e)
        self.integ_k = self.integrator_k.get_output(self.e_k)

        self.eta = self.tap_skips(self.v_diff)

        if torch.abs(self.integ_m) > torch.ones(
            (self.parallel_sims, 1), dtype=torch.float64
        ):
            self.switching_m(self.dir * self.v_diff, self.eta)
            self.integrator_m.reset()
        if torch.abs(self.integ_k) > torch.ones(
            (self.parallel_sims, 1), dtype=torch.float64
        ):
            self.switching_k(self.dir * self.v_diff)
            self.integrator_k.reset()

        self.u_l = 1 + self.delta_k * (self.tap_pos_k + self.tap_pos_m * self.delta_m)

        return self.u_l

    def differential(self):
        """
        Return derivative vector for internal dynamic blocks.

        Returns:
            torch.Tensor: Concatenated derivatives for integrators and optional PT1.

        Raises:
            NotImplementedError: If an internal block does not provide a
                differential implementation.
        """
        try:
            if self.pt1 is not None:
                deriv_vec = torch.concatenate(
                    (
                        self.integrator_m.differential(),
                        self.integrator_k.differential(),
                        self.pt1.differential(),
                    ),
                    axis=1,
                )
            else:
                deriv_vec = torch.concatenate(
                    (
                        self.integrator_m.differential(),
                        self.integrator_k.differential(),
                    ),
                    axis=1,
                )
        except Exception:
            raise NotImplementedError("Differential not implemented for integrator.")

        return deriv_vec

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulation support for FSM_Discrete.

        Converts scalar parameters to tensors and enables batched operation for
        internal PT1 and integrator blocks.

        Args:
            parallel_sims (int): Number of parallel simulations.

        Returns:
            None
        """
        if self.pt1 is not None:
            self.pt1.enable_parallel_simulation(parallel_sims)
        self.integrator_k.enable_parallel_simulation(parallel_sims)
        self.integrator_m.enable_parallel_simulation(parallel_sims)
        self.deadband.enable_parallel_simulation(parallel_sims)

        self.db = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.db
        self.v_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_ref
        self.u_l = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.u_l

        self.delta_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.delta_m
        )
        self.m_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_max
        self.m_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m_min
        self.n_taps_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.n_taps_m
        )
        # self.taps_m = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.taps_m
        self.tap_pos_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.tap_pos_m
        )
        self.t_m = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_m

        self.delta_k = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.delta_k
        )
        self.k_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_max
        self.k_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_min
        self.n_taps_k = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.n_taps_k
        )
        # self.taps_k = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.taps_k
        self.tap_pos_k = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.tap_pos_k
        )
        self.t_k = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_k

        self.integ_m = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.integ_m
        )
        self.integ_k = (
            torch.ones((parallel_sims, 1), dtype=torch.float64) * self.integ_k
        )

        self.m = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.m
        self.k = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k

        self.gamma = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.gamma
        self.e = torch.ones((self.parallel_sims, 1), dtype=torch.float64) * self.e
        self.e_k = torch.ones((self.parallel_sims, 1), dtype=torch.float64) * self.e_k
        self.eta = torch.ones((self.parallel_sims, 1), dtype=torch.float64) * self.eta

    def initialize(self, voltage):
        """
        Initialize FSM_Discrete internal state using a reference voltage.

        Args:
            voltage (torch.Tensor or float): Initial reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        if self.pt1 is not None:
            self.pt1.initialize(voltage)
        self.integrator_m.initialize(
            torch.zeros((self.parallel_sims, 1), dtype=torch.float64)
        )
        self.integrator_k.initialize(
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

    def tap_skips(self, v_diff):
        """
        Compute the number of tap skips to perform based on voltage deviation.

        Args:
            v_diff (torch.Tensor or float): Voltage deviation used to compute skips.

        Returns:
            int: Number of tap skips limited by controller's gamma parameter.
        """
        eta = np.floor(torch.abs(v_diff) / (self.db * self.delta_m))
        if eta > self.gamma:
            return self.gamma
        else:
            return eta

    def switching_m(self, v_diff, skips):
        """
        Decide a switch on the m-branch tap position by `skips` if available.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation.
            skips (int): Number of tap steps to apply.

        Returns:
            None
        """
        if v_diff > 0:
            if (self.tap_pos_m - skips) in self.taps_m:
                self.tap_pos_m -= skips
            else:
                self.tap_pos_m = self.m_min
        elif v_diff < 0:
            if (self.tap_pos_m + skips) in self.taps_m:
                self.tap_pos_m += skips
            else:
                self.tap_pos_m = self.m_max
        return

    def switching_k(self, v_diff):
        """
        Perform a single-step k-branch tap change based on the sign of v_diff.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation.

        Returns:
            None
        """
        if (
            v_diff > 0
            and (
                self.tap_pos_k
                - torch.ones((self.parallel_sims, 1), dtype=torch.float64)
            )
            in self.taps_k
        ):
            self.tap_pos_k -= torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        elif (
            v_diff < 0
            and (
                self.tap_pos_k
                + torch.ones((self.parallel_sims, 1), dtype=torch.float64)
            )
            in self.taps_k
        ):
            self.tap_pos_k += torch.ones((self.parallel_sims, 1), dtype=torch.float64)
        return
