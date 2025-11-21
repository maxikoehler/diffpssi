"""Defines the backend for the simulation models.

Torch enables GPU acceleration and gradient tracking. Numpy is faster, but does not support GPU acceleration or gradient tracking.
Note: Even if torch is used, GPU is not always faster than CPU.
"""

import logging
import os

import numpy as np
import torch

_logger = logging.getLogger(__name__)

BACKEND = "torch"  # 'torch' or 'numpy'
DEFAULT_DEVICE = "cpu"  # 'cuda' or 'cpu'

if os.environ.get("DIFFPSSI_FORCE_SIM_BACKEND") is not None:
    BACKEND = os.environ.get("DIFFPSSI_FORCE_SIM_BACKEND")
    _logger.warning(
        "WARNING: FORCING THE USE OF {} AS BACKEND."
        "THIS SHOULD ONLY HAPPEN FOR UNITTESTS".format(
            os.environ.get("DIFFPSSI_FORCE_SIM_BACKEND")
        )
    )

if BACKEND == "torch":
    import torch

    _logger.info("Using torch as backend")
    if torch.cuda.is_available() and (
        DEFAULT_DEVICE == "cuda" or DEFAULT_DEVICE == "gpu"
    ):
        torch.set_default_device("cuda")
        _logger.info("Using torch with CUDA on GPU")
    else:
        _logger.info("Using torch on CPU")

elif BACKEND == "numpy":
    import numpy as torch

    torch.tensor = torch.array
    torch.unsqueeze = torch.expand_dims
    _logger.info("Using numpy as backend")
else:
    raise ValueError("No backend specified")


# class Backend:
#     """Backend class to handle tensor operations either via numpy or torch."""

#     @staticmethod
#     def backend():
#         """Returns the current backend as usable object."""
#         return backend_dict[BACKEND]

# backend_dict = {
#     'numpy': np,
#     'torch': torch
# }
