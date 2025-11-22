
# DiffPSSi — Differential Power System Simulation Toolkit

> Some badges will come soon.

Lightweight, modular Python library for simulation, optimization and stability analysis of electric power systems. This repository contains the library itself (`src/diffpssi`), accompanying examples, documentation and tests.

## Overview

DiffPSSi contains a framework designed for simulating and optimizing the dynamic behavior of power systems. The framework has two main benefits: 

1. Serve as a modular and relatively efficient dynamic power system simulation over which users have full control for research and educational purposes. 
1. Enable the use of automatic differentiation for dynamic power system simulations. 

Effectively, this allows for the calculation of gradients of all simulation parameters with respect to a desired output of the simulation. This is useful for parameter optimization or identification or integration of neural networks into the simulation.

It includes detailed models of various power system components such as synchronous machines, exciters, governors, and power system stabilizers. The toolkit is built in Python and leverages the power of libraries like numpy efficient computation, and torch for automatic differentiation.

The code is strongly based on this repository, but required a rewrite to enable the gradient calculation for optimization purposes. The code is still under development and will be extended in the future.

> Note: This repository is still under development and will be extended in the future. Use at your own risk.

## Features

- Inherently Parallel Implementation: A unique and important feature of this simulation framework, as it allows the execution of multiple simulations in parallel by using vectors of parameters for every element.
- Dynamic Simulation: Allows for detailed dynamic simulations of power systems, including interactions between various components.
- Extensible Model Library: Contains models of AVRs, governors, stabilizers, static models like lines, loads, transformers, and more.
- Backend Flexibility: Choose between torch and numpy as backend for computations.
- Solver Options: Includes Euler and Runge Kutta methods for numerical integration.

## Requirements and Installation

- Python 3.11+ (tested with Python 3.11 to 3.14)
- A standard Python package manager (pip)
- Optional: Jupyter for notebooks, MkDocs for documentation

Exact dependencies are listed in `pyproject.toml` and the documentation requirements.

For finding the suitable installation method, please see `INSTALLATION.md` and `CONTRIBUTING.md`.  

## Quickstart — minimal example

Simple example (Python):

```python
from diffpssi import some_entrypoint

# Replace `some_entrypoint` with the concrete script or API function
# you want to use from `src/diffpssi`.
res = some_entrypoint.run_example()
print(res.summary())
```

More concrete examples can be found in `examples/` (e.g. `examples/models/ibb_manual/ibb_manual_sim.py`).

For full CI or environment tests, `tox` is configured (`tox.ini`).

## Documentation

For an introductory explanation of how dynamic power system simulations work in principle, refer to the article: ["Watts Up with Dynamic Power System Simulations"](https://medium.com/@georg.kordowich/watts-up-with-dynamic-power-system-simulations-c0f16fc99769).

You can get the latest version of DiffPSSi via cloning this repository.

```bash
$ git clone git@github.com:maxikoehler/diffpssi.git
```

User and API documentation is located in the `docs/` folder and can be built locally with MkDocs.
After cloning and installing the package in the development version (see `INSTALLATION.md`), activate the virtual environment and serve the documentation via ``mkdocs``.

```bash
$ .venv/Source/activate  # for Windows
$ source .venv/bin/activate # for Linux / MacOS 

$ mkdocs serve
```

Then open http://127.0.0.1:8000 to view the documentation interactively.

## Structure of the Package

- `examples/`: runnable scripts with model data
- `src/diffpssi/`:
    - `grid_library/`: Contains a set of example network configurations.
    - `optimization_lib/`: Includes optimizers and tools for gradient computation.
    - `power_sim_lib/`: Core library with various submodules: 
        - `models`: Models for AVRs, governors, stabilizers, etc.
        - `load_flow`: Tools for load flow analysis.
        - `simulator`: The core simulation class.
        - `solvers`: Numerical solvers for integration.
    - `stability_lib/`: Methods and calculation indices for stability analysis of the system. 
- `tests/`: Contains tests for integration and ensuring validity against known PSS software (e.g. DIgSILENT PowerFactory)


## Contributing

Contributions are welcome. Please follow the guidelines in `CONTRIBUTING.md`. Short workflow:

1. Fork and create a branch
2. Add tests for your changes
3. Open a pull request against the `development` branch

## License and Citation

This project is licensed under the terms shown in the `LICENSE` file. The project may be used for academic purposes — please cite the following [work(s)](https://doi.org/10.30420/566464032):

```bibtex
@INPROCEEDINGS{10926572,
  author={Kordowich, Georg and Jaeger, Johann},
  booktitle={NEIS 2024; Conference on Sustainable Energy Supply and Energy Storage Systems}, 
  title={An Accessible PyTorch Implementation of Automatic Differentiation for Power System Model Parameter Identification and Optimization}, 
  year={2024},
  volume={},
  number={},
  pages={231-236},
  keywords={},
  doi={10.30420/566464032}
}
```

## Contact & Support

For questions, please open an issue or contact the maintainers via the repository.
If requested and needed, please feel to [contact](https://www.ees.tf.fau.de/faudir/georg-kordowich/).
