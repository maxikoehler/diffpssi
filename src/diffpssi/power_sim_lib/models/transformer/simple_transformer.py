"""Simple transformer model implementation."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.transformer.transformer_interface import Transformer


class Simple_Transformer(Transformer):
    """
    Simple transformer with a fixed phase-shifting representation.

    This lightweight transformer provides a direct phase-shift/tap ratio
    representation and delegates most behavior to the base :class:`Transformer`.
    """

    def __init__(
        self, s_n_sys, param_dict, parallel_sims, sim, trans_model="AM", name=None
    ):
        """
        Initialize a Simple_Transformer instance.

        Args:
            s_n_sys (float): System base apparent power.
            param_dict (dict): Transformer parameters (see base class).
            sim (object, optional): Simulation instance (unused by this class).
            trans_model (str): Transformer model id.
            parallel_sims (int): Number of parallel simulations.
        """
        super().__init__(s_n_sys, param_dict, parallel_sims, sim, trans_model)

    def update_ratio(self):
        """
        Update the transformer's tap/ratio.

        This method is a thin wrapper left for API compatibility and forwards to
        the superclass implementation if present.

        Returns:
            Depends on the superclass implementation.
        """
        return super().update_ratio()

    def calc_admittance(self, return_need):
        """
        Calculate the admittance matrix for the simple transformer.

        Args:
            return_need (bool): If True return the computed admittance matrix.

        Returns:
            torch.Tensor or None: See :py:meth:`Transformer.calc_admittance`.
        """
        self.u = self.u_l * torch.exp(1j * (self.theta / 360) * 2 * torch.pi)
        return super().calc_admittance(return_need)

    def calc_admittance_static(self, return_need):
        """
        Calculate the static admittance matrix for the simple transformer.

        Args:
            return_need (bool): If True return the computed admittance matrix.

        Returns:
            torch.Tensor or None: See :py:meth:`Transformer.calc_admittance`.
        """
        # should be the same as super-class
        self.u = self.u_l * torch.exp(1j * (self.theta / 180) * torch.pi)
        return super().calc_admittance_static(return_need)

    def calc_current_injections(self):
        """
        Calculate the absolute injected current from the shunt branch.

        Returns:
            torch.Tensor: Current injection as computed by
            :py:meth:`Transformer.calc_current_injections`.
        """
        return super().calc_current_injections()

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enable parallel simulation mode for this transformer.

        Args:
            parallel_sims (int): Number of parallel simulations to enable.
        """
        return super().enable_parallel_simulation(parallel_sims)

    def get_value(self, value, sim=None, angle_type="deg"):
        """
        Return derived quantities for the simple transformer.

        See :py:meth:`Transformer.get_value` for full parameter description.
        """
        return super().get_value(value, sim, angle_type)

    def initialize(self):
        """
        Perform initialization steps for the simple transformer.

        Delegates to the base class implementation.
        """
        return super().initialize()

    def differential(self):
        """
        Return the differential vector for the simple transformer.

        Delegates to the base class implementation.
        """
        return super().differential()
