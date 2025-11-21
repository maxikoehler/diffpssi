"""This module contains inverter models for power system simulations."""

import torch

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import *


class SimpleInverter(object):
    """Definition of a simple inverter model.

    The model consists of an ideal current source which is controlled by
    two PI controllers (P and Q controller).
    """

    def __init__(
        self,
        f_n_sys,
        s_n_sys,
        v_n_sys,
        bus,
        s_n,
        v_n,
        p_setp,
        q_setp,
        k_p_p,
        k_i_p,
        k_p_q,
        k_i_q,
    ):
        """Construct a SimpleInverter object.

        Args:
            bus: The bus to which the inverter is connected.
            s_n: The nominal power of the inverter.
            v_n: The nominal voltage of the inverter.
            p: The active power setpoint of the inverter.
            q: The reactive power setpoint of the inverter.
            k_p: The proportional gain of the PI controller.
            k_i: The integral gain of the PI controller.
            t_i: The integral time constant of the PI controller.
        """
        self.f_n_sys = f_n_sys
        self.s_n_sys = s_n_sys
        self.v_n_sys = v_n_sys

        self.bus = bus
        self.s_n = s_n
        self.v_n = v_n
        self.p_setp = p_setp
        self.q_setp = q_setp
        self.k_p_p = k_p_p
        self.k_i_p = k_i_p

        self.k_p_q = k_p_q
        self.k_i_q = k_i_q

        self.p = 0
        self.q = 0
        self.i_d = 0
        self.i_q = 0
        self.v_bb = 0

        self.pi_ctrl_p = PIController(k_p=self.k_p_p, k_i=self.k_i_p)
        self.pi_ctrl_q = PIController(k_p=self.k_p_q, k_i=self.k_i_q)

        self.parallel_sims = None

    def differential(self):
        """Calculate the differential equations of the inverter model.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        return torch.concatenate(
            [self.pi_ctrl_p.differential(), self.pi_ctrl_q.differential()], axis=1
        )

    def set_state_vector(self, x):
        """Set the state vector of the inverter model.

        Args:
            x: The state vector to set.
        """
        self.pi_ctrl_p.set_state_vector(x[:, 0:1])
        self.pi_ctrl_q.set_state_vector(x[:, 1:2])

    def get_state_vector(self):
        """Return the state vector of the inverter model.

        Returns:
            torch.Tensor: The state vector of the inverter model.
        """
        return torch.concatenate(
            [self.pi_ctrl_p.get_state_vector(), self.pi_ctrl_q.get_state_vector()],
            axis=1,
        )

    def calc_current_injections(self):
        """Calculate the current injections of the inverter model.

        Returns:
            torch.Tensor: A tensor containing the current injections of the inverter model.
        """
        # power shall always be in MW/MVAr not in p.u.
        self.p = abs(self.v_bb) * self.i_d * self.s_n
        self.q = abs(self.v_bb) * self.i_q * self.s_n

        self.i_d = self.pi_ctrl_p.get_output((self.p_setp - self.p) / self.s_n)
        self.i_q = self.pi_ctrl_q.get_output((self.q_setp - self.q) / self.s_n)

        return (self.i_d - 1j * self.i_q) * torch.exp(1j * torch.angle(self.v_bb))

    def update_internal_vars(self, v_bb):
        """Update the internal variables of the inverter model.

        Args:
            v_bb: The bus voltage of the inverter.
        """
        self.v_bb = v_bb

    def initialize(self, s_calc, v_bb):
        """Initialize the inverter model.

        Args:
            s_calc: The calculated power of the inverter.
            v_bb: The bus voltage of the inverter.
        """
        self.i_d = self.p_setp / self.s_n / (abs(v_bb))
        self.i_q = self.q_setp / self.s_n / (abs(v_bb))

        self.pi_ctrl_p.initialize(self.i_d)
        self.pi_ctrl_q.initialize(self.i_q)

        self.update_internal_vars(v_bb)

    def get_admittance(self, dyn):
        """Get the admittance of the inverter model.

        Args:
            dyn: Boolean indicating whether the simulation is dynamic or not.
        """
        return torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

    def get_lf_power(self):
        """Get the load flow power of the inverter model."""
        return (self.p_setp + 1j * self.q_setp) / self.s_n

    def enable_parallel_simulation(self, parallel_sims):
        """Enable parallel simulation for the inverter model."""
        self.parallel_sims = parallel_sims
        self.pi_ctrl_p.enable_parallel_simulation(parallel_sims)
        self.pi_ctrl_q.enable_parallel_simulation(parallel_sims)
