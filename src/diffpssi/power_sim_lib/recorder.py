"""Implements a recorder class to log simulation data over time."""


class Recorder(object):
    """
    Helper to manage recording configuration and execution.

    The Recorder wraps a user-provided recorder function and exposes helpers
    to produce a static description list and to record values during a run.

    Attributes:
        sim (PowerSystemSimulation or TestBench): Simulation/TestBench instance.
        recorder_dict (callable): Callable that defines what to record.
        description (list): List of recorded field descriptions.
    """

    def __init__(self, sim=None, recorder_dict=None):
        """
        Initialize the Recorder.

        Args:
            sim (PowerSystemSimulation or TestBench, optional): Simulation instance.
            recorder_dict (callable, optional): Recorder function. When provided,
                the Recorder will populate the description list by calling the
                function with ``call=False``.

        Returns:
            None
        """
        if sim is not None:
            self.sim = sim
        else:
            self.sim = None

        if recorder_dict is not None:
            self.recorder_dict = recorder_dict

        if recorder_dict is not None:
            self.description = []
            for entry in recorder_dict(self.sim, call=False):
                self.description.append(entry)

    def set_record_func(self, recorder_dict):
        """
        Set or replace the recorder function and rebuild the description list.

        Args:
            recorder_dict (callable): Callable taking (sim, call=False|True) and returning descriptions or values.

        Returns:
            None
        """
        self.recorder_dict = recorder_dict

        self.description = []
        for entry in recorder_dict(self.sim, call=False):
            self.description.append(entry)

    def record_fun(self, sim):
        """
        Execute the recorder function and return a list of recorded entries.

        Args:
            sim (PowerSystemSimulation or TestBench): Simulation instance passed to the recorder.

        Returns:
            list: List of recorded tensor/value entries.
        """
        rec = []
        for entry in self.recorder_dict(sim, call=True):
            rec.append(entry)
        return rec

    def record_list(self):
        """
        Return the list of recorded data field descriptions.

        Returns:
            list: Description strings previously collected from the recorder function.
        """
        return self.description
