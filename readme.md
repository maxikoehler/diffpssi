
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

Detailed usage instructions and examples can be found in the `examples` directory. For understanding the general workflow, the following shall demonstrate how to set up a simulation.

### Model Definition

Generally, there are two options to create a simulation. One option is to create the model as a dictionary and pass it to the simulation. This is the recommended way. For an example, check out the IBB model simulation example under `examples/models/ibb_model/ibb_sim.py` and the corresponding model 
in `grid_library/ibb_model.py`.

The other option is to create the model manually in the simulation file. This can be seen in the example under `examples/models/ibb_model/ibb_sim_manual.py`. For this option, first a simulation must be created and afterward, busses, generators, and lines can be added to the simulation. The following code snippet shows how to create a simulation and add busses, generators, and lines to it.

```python
sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=10,
        time_step=0.005,
        solver="heun",
    )

    sim.fn = 60
    sim.base_mva = 2200
    sim.base_voltage = 24

    sim.add_bus(Bus(name="Bus 0", v_n=24))
    sim.add_bus(Bus(name="Bus 1", v_n=24))

    sim.add_line(
        Line(
            name="L1",
            from_bus="Bus 0",
            to_bus="Bus 1",
            length=1,
            s_n=2200,
            v_n=24,
            unit="p.u.",
            r=0,
            x=0.65,
            b=0,
            s_n_sys=2200,
            v_n_sys=24,
        )
    )

    sim.add_generator(
        SynchMachine(
            name="IBB",
            bus="Bus 0",
            s_n=22000,
            v_n=24,
            p=-1998,
            v=0.995,
            h=3.5e7,
            d=0,
            x_d=1.81,
            x_q=1.76,
            x_d_t=0.3,
            x_q_t=0.65,
            x_d_st=0.23,
            x_q_st=0.23,
            t_d0_t=8.0,
            t_q0_t=1,
            t_d0_st=0.03,
            t_q0_st=0.07,
            f_n_sys=60,
            s_n_sys=2200,
            v_n_sys=24,
        )
    )
    sim.add_generator(
        SynchMachine(
            name="Gen 1",
            bus="Bus 1",
            s_n=2200,
            v_n=24,
            p=1998,
            v=1,
            h=3.5,
            d=0,
            x_d=1.81,
            x_q=1.76,
            x_d_t=0.3,
            x_q_t=0.65,
            x_d_st=0.23,
            x_q_st=0.23,
            t_d0_t=8.0,
            t_q0_t=1,
            t_d0_st=0.03,
            t_q0_st=0.07,
            f_n_sys=60,
            s_n_sys=2200,
            v_n_sys=24,
        )
    )

    sim.set_slack_bus("Bus 0")

```

### Running Simulations

Once the model is defined, you can run a simulation. One unique feature of this framework is that you can define the number of parallel simulations to run. This is useful for parameter optimization, where you can run multiple simulations in parallel to speed up the process. It is also possible to add events to the simulation, such as a short circuit event.

```python
sim = mdl.get_model(parallel_sims)
sim.add_sc_event(1, 1.05, 'Bus 1')
sim.set_record_function(record_desired_parameters)

# Run the simulation. Recorder format shall be [batch, timestep, value]
t, recorder = sim.run()
``` 

### Recording Simulation Parameters

To acquire and record data during the simulation, you can define a record function. This function is called at every time step and can be used to record any desired parameters. The function shall return a list of desired parameters, which will be recorded during the simulation. The recorder format shall be `[batch, timestep, value]`. By using this function during the simulation, parameters can be plotted afterward.

```python
# Simple form without connection to the parameter description
def record_desired_parameters(simulation):
    # Record the desired parameters
    record_list = [
        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_d_st.real,
    ]
    return record_list

sim
```

If one wants to look into more parameters for example, there is the possibility to connect the Parameter description directly with the recorded parameter. This improves the difficulty e.g. for plotting.
```python
# More sophisticated version with connected parameter description for plotting
def record_dict(simulation, call=False):
    record_dict = {
        r'Machine spped $\omega$':                        simulation.busses[1].models[0].omega.real,
        r'Machine excitation Voltage $E_\mathrm{q,st}$':  simulation.busses[1].models[0].e_q_st.real,
        r'Machine excitation Voltage $E_\mathrm{q,st}$':  simulation.busses[1].models[0].e_d_st.real,
    }
    if call:
        return record_dict.values()
    else:
        return record_dict

# The setting of the record function is then a bit alternated.
# Setting the recorder functions in the sim
rec_sim = Recorder(sim=sim, recorder_dict=record_dict)

# Set the sim recorder
sim.set_record_function(rec_sim.record_fun)

# Getting the descriptions of the Recorder 
record_list = rec_sim.record_list()
```

More concrete examples can be found in `examples/` (e.g. `examples/models/ibb_manual/ibb_manual_sim.py`).

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
