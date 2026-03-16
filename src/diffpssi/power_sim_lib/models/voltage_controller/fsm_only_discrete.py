"""Discrete FSM tap skipping controller implementation."""

import numpy as np

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import DeadBand, Integrator, PT1Limited
from diffpssi.power_sim_lib.models.voltage_controller.oltc_discrete import OLTC_Discrete


class FSM_Only_Discrete(OLTC_Discrete):
    """
    FSM discrete controller only.

    Provides deadband, PT1 filtering and m
    integrator branches.

    Args:
        N/A: See :py:meth:`__init__` for constructor arguments.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
        parallel_sims=1,
    ):
        """
        Initialize a discrete single-module OLTC controller.

        Args:
            name (str, optional): Controller name. Defaults to None.
            trafo (object, optional): Associated transformer instance. Defaults to None.
            param_dict (dict, optional): Parameter dictionary. If omitted the
                individual keyword arguments are used to build a default
                parameter set. Expected keys: ``t_1``, ``db``, ``delta_m``,
                ``m_max``, ``m_min``, ``v_ref``, ``pt_1``.
            parallel_sims (int): Number of parallel simulations (default: 1).

        Returns:
            None
        """
        super().__init__(
            name=name,
            trafo=trafo,
            param_dict=param_dict,
            parallel_sims=parallel_sims,
        )

        self.eta = 0
        self.gamma = param_dict.get("gamma", 4)

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
        super().enable_parallel_simulation(parallel_sims)
        self.eta = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.eta
        self.gamma = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.gamma

    def switching(self, v_diff, switch_mask):
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
        # determine possible skips of taps
        self.eta = self.tap_skips(v_diff)

        # determine switching direction
        a = torch.where(
            torch.logical_and(
                torch.logical_and(
                    v_diff < 0,
                    self.tap_pos_m + self.eta < self.taps[:, -1].reshape(-1, 1),
                ),
                switch_mask,
            ),
            self.eta,
            torch.where(
                torch.logical_and(
                    torch.logical_and(
                        v_diff > 0,
                        self.tap_pos_m - self.eta > self.taps[:, -1].reshape(-1, 1),
                    ),
                    switch_mask,
                ),
                torch.abs(self.tap_pos_m - self.taps[:, -1].reshape(-1, 1)),
                torch.zeros((1)),
            ),
        )
        b = torch.where(
            torch.logical_and(
                torch.logical_and(
                    v_diff > 0,
                    self.tap_pos_m - self.eta > self.taps[:, 0].reshape(-1, 1),
                ),
                switch_mask,
            ),
            -1 * self.eta,
            torch.where(
                torch.logical_and(
                    torch.logical_and(
                        v_diff > 0, self.tap_pos_m > self.taps[:, 0].reshape(-1, 1)
                    ),
                    switch_mask,
                ),
                -1 * torch.abs(self.tap_pos_m - self.taps[:, 0].reshape(-1, 1)),
                torch.zeros((1)),
            ),
        )
        m = torch.where(
            a != 0,
            a,
            torch.where(
                b != 0,
                b,
                torch.zeros((1)),
            ),
        )

        switched = torch.where(m != 0, True, False)

        self.tap_pos_m = self.tap_pos_m + m

        return switched

    def tap_skips(self, v_diff):
        """
        Determine the number of tap steps to skip based on voltage deviation.

        Args:
            v_diff (torch.Tensor or float): Signed voltage deviation from
                reference used to decide switching direction.

        Returns:
            torch.Tensor: Number of tap steps to skip (per parallel simulation).
        """
        # determine number of tap steps to skip
        skips = torch.floor((torch.abs(v_diff) - self.db) / self.delta_m)  # = 0

        # # wenn das trz. > 1, dann min. 1 skip
        # skips_upper_limit = (torch.abs(v_diff) / self.delta_m)
        # skips = torch.where(
        #     torch.logical_and(
        #         skips == 0,
        #         skips_upper_limit > 1,
        #     ),
        #     torch.floor(skips_upper_limit),
        #     skips
        # )
        # -> Does not work if v_diff = delta_m
        # -> skips = 1, but ratio too strong, resulting voltage is then
        # with changed sign, but within the deadband
        # -> potentially tap hunting?

        eta = torch.where(
            skips < self.gamma,
            skips,
            self.gamma,
        )

        return eta

    def reset(self):
        """
        Reset the controller to its initial state.

        Returns:
            None
        """
        super().reset()
        self.eta = torch.zeros_like(self.eta)
        self.gamma = torch.zeros_like(self.gamma)
