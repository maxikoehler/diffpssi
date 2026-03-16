"""
This example demonstrates how to use the PowerSystemOptimization class to optimize the parameters of a power system
model.
"""

import numpy as np
import torch
import os

os.environ["DIFFPSSI_SIM_BACKEND"] = "torch"

import examples.models.ieee_9bus.ieee_9bus_model as mdl
from src.diffpssi.optimization_lib.ps_optimization import PowerSystemOptimization
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

np.random.seed(0)


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    record_list = [
        simulation.busses[0].models[0].omega.real,
        simulation.busses[0].models[0].p_e,
        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].p_e,
        simulation.busses[2].models[0].omega.real,
        simulation.busses[2].models[0].p_e,
    ]
    return record_list


def main(parallel_sims=5, num_params=15):
    """
    This function runs a parameter identification for the IEEE 9 bus system.
    """
    parallel_sims = parallel_sims

    sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=1,
        time_step=0.01,
        solver="heun",
        grid_data=mdl.load(),
    )

    sim.add_sc_event(0.1, 0.15, "B8")

    sim.set_record_function(record_desired_parameters)

    param_names = [
        "G1 h",
        "G2 h",
        "G3 h",
        "G1 x_d",
        "G2 x_d",
        "G3 x_d",
        "G1 x_q",
        "G2 x_q",
        "G3 x_q",
        "G1 x_d_t",
        "G2 x_d_t",
        "G3 x_d_t",
        "G1 x_q_t",
        "G2 x_q_t",
        "G3 x_q_t",
    ]

    params_original = [
        9.55,  # G1 h
        3.92,  # G2 h
        2.766544,  # G3 h
        0.36135,  # G1 x_d
        1.719936,  # G2 x_d
        1.68,  # G3 x_d
        0.2398275,  # G1 x_q
        1.65984,  # G2 x_q
        1.609984,  # G3 x_q
        0.15048,  # G1 x_d_t
        0.230016,  # G2 x_d_t
        0.232064,  # G3 x_d_t
        0.15048,  # G1 x_q_t
        0.378048,  # G2 x_q_t
        0.32,  # G3 x_q_t
    ]

    if num_params > 0:
        sim.busses[0].models[0].h = torch.tensor(
            np.random.uniform(
                0.5 * params_original[0], 2.0 * params_original[0], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 1:
        sim.busses[1].models[0].h = torch.tensor(
            np.random.uniform(
                0.5 * params_original[1], 2.0 * params_original[1], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 2:
        sim.busses[2].models[0].h = torch.tensor(
            np.random.uniform(
                0.5 * params_original[2], 2.0 * params_original[2], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 3:
        sim.busses[0].models[0].x_d = torch.tensor(
            np.random.uniform(
                0.5 * params_original[3], 2.0 * params_original[3], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 4:
        sim.busses[1].models[0].x_d = torch.tensor(
            np.random.uniform(
                0.5 * params_original[4], 2.0 * params_original[4], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 5:
        sim.busses[2].models[0].x_d = torch.tensor(
            np.random.uniform(
                0.5 * params_original[5], 2.0 * params_original[5], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 6:
        sim.busses[0].models[0].x_q = torch.tensor(
            np.random.uniform(
                0.5 * params_original[6], 2.0 * params_original[6], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 7:
        sim.busses[1].models[0].x_q = torch.tensor(
            np.random.uniform(
                0.5 * params_original[7], 2.0 * params_original[7], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 8:
        sim.busses[2].models[0].x_q = torch.tensor(
            np.random.uniform(
                0.5 * params_original[8], 2.0 * params_original[8], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 9:
        sim.busses[0].models[0].x_d_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[9], 2.0 * params_original[9], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 10:
        sim.busses[1].models[0].x_d_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[10], 2.0 * params_original[10], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 11:
        sim.busses[2].models[0].x_d_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[11], 2.0 * params_original[11], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 12:
        sim.busses[0].models[0].x_q_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[12], 2.0 * params_original[12], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 13:
        sim.busses[1].models[0].x_q_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[13], 2.0 * params_original[13], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )
    if num_params > 14:
        sim.busses[2].models[0].x_q_t = torch.tensor(
            np.random.uniform(
                0.5 * params_original[14], 2.0 * params_original[14], (parallel_sims, 1)
            ),
            requires_grad=True,
            dtype=torch.complex128,
        )

    optimizable_parameters = [
        sim.busses[0].models[0].h,
        sim.busses[1].models[0].h,
        sim.busses[2].models[0].h,
        sim.busses[0].models[0].x_d,
        sim.busses[1].models[0].x_d,
        sim.busses[2].models[0].x_d,
        sim.busses[0].models[0].x_q,
        sim.busses[1].models[0].x_q,
        sim.busses[2].models[0].x_q,
        sim.busses[0].models[0].x_d_t,
        sim.busses[1].models[0].x_d_t,
        sim.busses[2].models[0].x_d_t,
        sim.busses[0].models[0].x_q_t,
        sim.busses[1].models[0].x_q_t,
        sim.busses[2].models[0].x_q_t,
    ]

    optimizable_parameters = optimizable_parameters[:num_params]
    params_original = params_original[:num_params]

    # mute sim because it does not generate added value
    sim.verbose = False

    original_data = np.load("private_examples/ieee_9bus/data/original_data.npy")
    orig_tensor = torch.tensor(original_data) * torch.ones(
        (parallel_sims,) + original_data.shape
    )

    opt = PowerSystemOptimization(
        sim,
        orig_tensor,
        params_optimizable=optimizable_parameters,
        params_original=params_original,
        param_names=param_names,
        enable_plots=True,
        normalize_loss=True,
    )

    opt.run(max_steps=10)


if __name__ == "__main__":
    for i in range(1, 16):
        print(f"Running with {i} parameters")
        main(parallel_sims=1, num_params=i)
