"""Providing automated stability calculations based on standardized data."""

import typing

from diffpssi.power_sim_lib.simulator import PowerSystemSimulation

# pylint: disable=too-few-public-methods


class VoltageStability:
    """Voltage stability assessment class.

    The class encapsulates a workflow to perform voltage stability analysis on a
    given power system simulation. Typical steps are:
      - Create an assessment object with desired analysis parameters.
      - Initialize or attach a simulation object.
      - Extract necessary input data from the grid and add it to the recorder.
      - Run the simulation and compute desired indices over time.
      - Return indices and optionally store logs/results.

    Attributes:
        ps_sim (PowerSystemSimulation): Simulation object used for assessments.
        busses (list[str]): Bus names included in the assessment.
        grid (typing.Callable): Callable that provides or constructs grid data.
    """

    def __init__(
        self,
        ps_sim: PowerSystemSimulation,
        busses: list[str],
        grid: typing.Callable,
    ):
        """Initialize the VoltageStability assessment object.

        Args:
            ps_sim (PowerSystemSimulation): The simulation object to use for runs.
            busses (list[str]): List of bus names to include in the assessment.
            grid (typing.Callable): Callable used to obtain or construct grid data
                (e.g., a function that returns the grid configuration when called).
        """

    def run_voltage_stability(self):
        """Execute the voltage stability assessment workflow.

        This method should run the simulation, extract required recorder data,
        compute stability indices and return or store results.

        Returns:
            dict | None: Assessment results containing computed indices and
                optional metadata, or ``None`` if the method implementation
                does not return a value.
        """
