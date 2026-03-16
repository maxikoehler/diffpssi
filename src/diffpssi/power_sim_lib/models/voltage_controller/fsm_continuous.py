"""Model a continuous FSM voltage controller."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.voltage_controller.oltc_continuous import (
    OLTC_Continuous_Milano,
)


class FSM_Continuous_Milano(OLTC_Continuous_Milano):
    """Finite State Machine (FSM) continuous voltage controller model.

    Inherits from OLTC_Continuous_Milano.

    Attributes:
        Inherits all attributes from OLTC_Continuous_Milano.
    """

    def __init__(
        self,
        name=None,
        trafo=None,
        param_dict=None,
    ):
        """Initialize FSM_Continuous_Milano with given parameters.

        Args:
            param_dict (dict): Dictionary of controller parameters.
        """
        super().__init__(
            name=name,
            trafo=trafo,
            param_dict=param_dict,
        )

        def def_k_i(delta_v):
            """Default k_i calculation based on delta_v."""
            return self.lag.ki

        def def_k_d(delta_v):
            """Default k_d calculation based on delta_v."""
            return self.lag.kd

        self.k_i_dependency = param_dict.get("k_i_dependency", def_k_i)
        self.k_d_dependency = param_dict.get("k_d_dependency", def_k_d)

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

        # update k_d and k_i based on v_diff
        self.lag.k_d = self.k_d_dependency(self.v_diff)
        self.lag.k_i = self.k_i_dependency(self.v_diff)

        u = self.lag.get_output(self.dir * self.v_diff)

        # discretizing
        # u = torch.floor_divide(u, 0.02) * 0.02

        if self.deadband != None:
            self.u_l = self.deadband.get_output(u)
        else:
            self.u_l = u

        return self.u_l
