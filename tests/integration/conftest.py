"""Configuration file for pytest environment."""

import os
import sys

extra_module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
if extra_module_path not in sys.path:
    sys.path.insert(0, extra_module_path)
