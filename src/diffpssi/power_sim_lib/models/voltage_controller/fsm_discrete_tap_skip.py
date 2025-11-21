"""Discrete FSM tap skipping controller implementation."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, PT1Limited
from diffpssi.power_sim_lib.models.voltage_controller.voltage_controller_interface import (
    Voltage_Controller,
)


class FSM_Discrete_Tap_Skip(Voltage_Controller):
    """
    FSM discrete controller with tap-skip behavior.

    This variant implements tap-skip logic similar to :class:`FSM_Discrete` but
    uses a different rule for computing skips and switching thresholds. It
    exposes the same public interface (state vector, enable_parallel_simulation,
    initialize, get_output, differential) so it can be swapped with other
    FSM controllers in the transformer models.

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
        Initialize the FSM_Discrete_2 controller with given parameters.

        Args:
            name (str, optional): Controller name.
            trafo (object, optional): Associated transformer instance.
            param_dict (dict, optional): Parameter dictionary. If omitted the
                provided defaults are used. Expected keys include timing,
                deadband and tap ranges for both FSM and OLTC modules.
            parallel_sims (int): Number of parallel simulations.
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
            self.pt1 = PT1Limited(t_pt1=self.pt1_const, gain_pt1=1)
        else:
            self.pt1 = None
        self.deadband = DeadBand(self.db)
        self.integrator_m = Integrator(k_i=1 / self.t_m, limiter=None)
        self.integrator_k = Integrator(k_i=1 / self.t_k, limiter=None)

        self.parallel_sims = parallel_sims

    def get_state_vector(self):
        """
        Return the internal state vector for FSM_Discrete_Tap_Skip.

        The returned vector concatenates the integrator states and the PT1
        state (when present) in a stable order used by :py:meth:`set_state_vector`.

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
        Restore internal integrator and PT1 states from an external vector.

        Args:
            x (torch.Tensor): State vector matching the layout returned by
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
        Compute the tap ratio output for the tap-skip FSM controller.

        The method filters the measurement (optional PT1), applies deadband
        logic, integrates the error on both branches and initiates tap changes
        including multi-step skips when thresholds are exceeded.

        Args:
            vbb (torch.Tensor or float): Measured bus voltage magnitude.

        Returns:
            torch.Tensor or float: Updated tap ratio ``u_l`` after switching.
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
        Return derivatives of internal dynamic blocks for tap-skip FSM.

        Returns:
            torch.Tensor: Concatenated derivative tensor for integrators and
                optional PT1 filter.

        Raises:
            NotImplementedError: If an internal block lacks differential support.
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
        Enable parallel simulation (batched) mode for the tap-skip FSM.

        Converts scalar fields into tensors sized by ``parallel_sims`` and
        enables batched operation in internal integrator/PT1 blocks.

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
        Initialize controller internal state using a given voltage.

        Args:
            voltage (torch.Tensor or float): Initial reference/measurement voltage.

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
        Update the reference voltage used by the tap-skip controller.

        Args:
            voltage (torch.Tensor or float): New reference voltage.

        Returns:
            None
        """
        self.v_ref = voltage
        return

    def tap_skips(self, v_diff):
        """
        Compute tap-skip count for this controller variant.

        Args:
            v_diff (torch.Tensor or float): Voltage deviation used to compute skips.

        Returns:
            int: Tap skip count limited by controller's gamma parameter.
        """
        eta = floor(torch.abs(v_diff) / (self.delta_k * self.delta_m))
        if eta > self.gamma:
            return self.gamma
        else:
            return eta

    def switching_m(self, v_diff, skips):
        """
        Apply m-branch tap changes with skipping for tap-skip controller.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation.
            skips (int): Number of tap steps to apply.

        Returns:
            None
        """
        # Switching logic of the FSM module
        if v_diff > 0:
            if (self.tap_pos_m - skips) in self.taps_m:
                self.tap_pos_m -= skips
                # if skips > 0:
                #     self.integrator_k.reset()
                # if skips == 0:
                #     self.tap_pos_m -= 1
            else:
                self.tap_pos_m = self.m_min
        elif v_diff < 0:
            if (self.tap_pos_m + skips) in self.taps_m:
                self.tap_pos_m += skips
                # if skips > 0:
                #     self.integrator_k.reset()
                # if skips == 0:
                #     self.tap_pos_m += 1
            else:
                self.tap_pos_m = self.m_max
        return

    def switching_k(self, v_diff):
        """
        Apply k-branch single-step switching for tap-skip controller.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation.

        Returns:
            None
        """
        # Switching logic of the OLTC module
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
