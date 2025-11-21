
# diffpssi — Differential Power System Simulation Toolkit

Lightweight, modular Python library for simulation, optimization and stability analysis of electric power systems. This repository contains the library itself (`src/diffpssi`), accompanying examples, validation notebooks, documentation and tests.

## Overview

`diffpssi` provides tools to build and analyze power system models (power flow, stability, parametric studies). The project includes:

- Reusable model components under `src/diffpssi`.
- Example scripts and model folders under `examples/` and `diffpssi.grid_library/`.
- Documentation built with MkDocs in the `docs/` folder.

The architecture is modular, allowing simulation, optimization and stability-analysis components to be used independently.

## Key features

- Power-flow and stability analyses
- Parametric studies (e.g. variation of load, transformer parameters)
- Example models (IEEE test systems, user-defined models)
- Tools for result visualization (notebooks & plotting scripts)

## Requirements

- Python 3.10+ (tested with Python 3.10 to 3.14)
- A standard Python package manager (pip)
- Optional: Jupyter for notebooks, MkDocs for documentation

Exact dependencies are listed in `pyproject.toml` and the documentation requirements.

## Installation

Recommended: create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate    # for zsh / bash on macOS/Linux
pip install --upgrade pip
pip install -e .
```

Additional packages may be required for documentation and notebooks (see `docs/` and `pyproject.toml`).
For development work on the project, it is recommended to follow the instructions in `CONTRIBUTING.md`.

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

User and API documentation is located in the `docs/` folder and can be built locally with MkDocs:

```bash
pip install mkdocs mkdocs-material  # optional
mkdocs serve
```

Then open http://127.0.0.1:8000 to view the documentation.

## Examples & notebooks

- `development_files/` — case studies and analysis notebooks
- `validation/` — validation notebooks and datasets
- `examples/` and `private_examples/` — runnable scripts with model data

## Contributing

Contributions are welcome. Please follow the guidelines in `CONTRIBUTING.md`. Short workflow:

1. Fork and create a branch
2. Add tests for your changes
3. Open a pull request against the `development` branch

## License and citation

This project is licensed under the terms shown in the `LICENSE` file. The project may be used for academic purposes — please cite the following work(s):

```bibtex

```

## Contact & support

For questions, please open an issue or contact the maintainers via the repository.
