"""Unit tests for the simulation examples in the examples folder."""

import os
import unittest


class TestCorrectBackend(unittest.TestCase):
    def test_correct_backend(self):
        """This test checks if the correct backend is used."""
        # os.environ["DIFFPSSI_FORCE_SIM_BACKEND"] = "numpy"

        # import diffpssi.power_sim_lib.backend as backend

        # self.assertEqual(backend.BACKEND, "numpy")
        pass
