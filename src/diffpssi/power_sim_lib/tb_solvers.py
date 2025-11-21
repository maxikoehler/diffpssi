"""Solvers for usage in the test bench object.

These are differing on the reference of the models and how the states and differentials are called.
Currently only a simple Euler integrator is implemented.
"""

# from diffpssi.power_sim_lib.backend import *


class Euler:
    """Implement the Euler method for numerical integration in power system simulations.

    The Euler method is a first-order numerical procedure for solving ordinary differential
    equations (ODEs) with a given initial value. It is the most basic explicit method for
    numerical integration of ODEs and is the simplest Runge–Kutta method.

    Attributes:
        x_0_store (dict): A dictionary for storing the previous state vector of each model.
    """

    def __init__(self):
        """Initialize the Euler solver object."""
        self.x_0_store = {}

    def step(self, tb):
        """Execute one step of the Euler integration method for the power system simulation.

        Args:
            ps_sim (PowerSystemSimulation): The power system simulation object to be integrated.
        """
        for model in tb.diff_models:
            model_id = id(model)  # Unique identifier for each model
            dxdt_0 = model.differential()
            # Use previously stored x_1 if available, else use current state vector
            x_0 = model.get_state_vector()
            x_1 = x_0 + dxdt_0 * tb.time_step
            model.set_state_vector(x_1)
            # Store x_1 for next step
            self.x_0_store[model_id] = x_1

    def reset(self):
        """Reset the Euler solver object."""
        self.x_0_store = {}


tb_solver_dict = {
    "euler": Euler,
}
