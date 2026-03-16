"""Optimization of controller parameters using PyTorch."""

import os

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

import diffpssi.grid_library.ibb_model as mdl
from diffpssi.optimization_lib.ps_optimization import PowerSystemOptimization
from diffpssi.power_sim_lib import TestBench
from diffpssi.power_sim_lib.models.voltage_controller import (
    FSM_Only_Discrete,
    OLTC_Continuous_Milano,
)
from diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

# np.random.seed(0)


def jump_input(t, a: float = 1, b: float = 1.1, threshold: float = 1):
    """Callable jump input function.
    Args:
        t (float or list): time instant(s)
        a (float, optional): value before threshold. Defaults to 1.
        b (float, optional): value after threshold. Defaults to 1.1.
        threshold (float, optional): time threshold. Defaults to 1.
    Returns:
        torch.tensor or list: jump value(s)
    """
    # Depending on input type, return single or list of jump values
    # Necessary for compatibility with Recorder class and simulator
    if isinstance(t, float):
        if t < threshold:
            jump = a * torch.ones((1, 1), dtype=torch.float64)
        else:
            jump = b * torch.ones((1, 1), dtype=torch.float64)
    else:  # isinstance(t, list):
        jump = []
        for i in t:
            if i < threshold:
                jump.append(a * torch.ones((1, 1), dtype=torch.float64))
            else:
                jump.append(b * torch.ones((1, 1), dtype=torch.float64))
        jump = torch.tensor(jump, dtype=torch.float64)
    return jump


def input_function(t, a: float = 0.1, b: float = 1.1):
    """Input function for the test bench simulation, respectively control block testing.

    Args:
        t (float or list): time instant(s)
        a (float, optional): exponential decay rate. Defaults to 0.1.
        b (float, optional): final value after decay. Defaults to 1.1.

    Returns:
        torch.tensor or list: input value(s)
    """
    # Depending on input type, return single or list of jump values
    # Necessary for compatibility with Recorder class and simulator
    if isinstance(t, float):
        a = torch.tensor(a, dtype=torch.float64)
        b = torch.tensor(b, dtype=torch.float64)
        input = (
            b
            - (b - 1) * torch.exp(-a * t)
            # * torch.ones((1, 1), dtype=torch.float64)
        )
    else:
        a = torch.tensor(a, dtype=torch.float64)
        b = torch.tensor(b, dtype=torch.float64)
        input = []
        for i in t:
            input.append(
                b
                - (b - 1) * torch.exp(-a * i)
                # * torch.ones((1, 1), dtype=torch.float64)
            )

        input = torch.tensor(input, dtype=torch.float64)

    return input


def duplet_input(
    t, a: float = 1, b: float = 0.3, t1: float = 1, t2: float = 5, td: float = 2
):
    """Callable jump input function.
    Args:
        t (float or list): time instant(s)
        a (float, optional): value before threshold. Defaults to 1.
        b (float, optional): additional value after threshold. Defaults to 0.3.
        t1 (float, optional): start time threshold 1. Defaults to 1.
        t2 (float, optional): start time threshold 2. Defaults to 5.
        td (float, optional): duration between jumps. Defaults to 2.

    Returns:
        torch.tensor or list: jump value(s)
    """
    # Depending on input type, return single or list of jump values
    # Necessary for compatibility with Recorder class and simulator
    if isinstance(t, float):
        if t > t1 and t < t1 + td:
            jump = (a + b) * torch.ones((1, 1), dtype=torch.float64)
        elif t > t2 and t < t2 + td:
            jump = (a - b) * torch.ones((1, 1), dtype=torch.float64)
        else:
            jump = a * torch.ones((1, 1), dtype=torch.float64)
    else:  # isinstance(t, list):
        jump = []
        for i in t:
            if i > t1 and i < t1 + td:
                jump.append((a + b) * torch.ones((1, 1), dtype=torch.float64))
            elif i > t2 and i < t2 + td:
                jump.append((a - b) * torch.ones((1, 1), dtype=torch.float64))
            else:
                jump.append(a * torch.ones((1, 1), dtype=torch.float64))
        jump = torch.tensor(jump, dtype=torch.float64)
    return jump


def record_dict(simulation, call=False):
    record_dict = {
        "Model output": simulation.diff_models[0].u_l,
    }
    if call:
        return record_dict.values()
    else:
        return record_dict


def get_models():
    analysis_models = [
        [
            FSM_Only_Discrete(
                param_dict={
                    "t_1": 0.02,
                    "db": 0.025,
                    "delta_m": 0.04,
                    "m_min": 0.84,
                    "m_max": 1.20,
                    "gamma": 3,
                }
            ),
            1,
        ],
        [
            OLTC_Continuous_Milano(
                param_dict={
                    "kd": 0.2,
                    "ki": 0.2,
                    "m_min": 0.84,
                    "m_max": 1.20,
                    "v_ref": 1.0,
                }
            ),
            1,
        ],
    ]

    return analysis_models


def optimize_param_set(
    n_opt: int = 10,
    max_step: float = 0.08,
    decay: float = 0.9,
    input_func: callable = duplet_input,
):
    """Optimize controller parameters using PyTorch autograd functionality.

    Args:
        n_opt (int, optional): Number of optimization steps. Defaults to 10.
        max_step (float, optional): Maximum step size for parameter update. Defaults to 0.08.
        decay (float, optional): Decay factor for step size. Defaults to 0.9.
        input_func (callable, optional): Input function for the test bench simulation. Defaults to duplet_input.
    """
    parallel_sims = 10

    tb = TestBench(
        time_step=0.005,
        sim_time=20,
        inspection_models=get_models(),
        input_func=input_func,
        record_func=record_dict,
        default_runner="run_oltc_control",
        parallel_sims=parallel_sims,
    )
    tb.verbose = False

    # tb.diff_models[0].dir = -1

    t, benchmark_result = tb.run_oltc_control()

    benchmark_transformed = (
        benchmark_result[0, :, 0]
        * torch.ones((parallel_sims, 1, 1), requires_grad=True)
    ).reshape(parallel_sims, -1, 1)

    benchmark_transformed = benchmark_transformed.detach().clone()

    # plt.plot(t, benchmark_transformed[0, :, 0].detach().numpy())
    # plt.show()

    tb.diff_models.pop(0)

    # For model Milano
    tb.diff_models[0].lag.kd = torch.tensor(
        np.random.uniform(0.01, 0.5, (parallel_sims, 1)),
        requires_grad=True,
        dtype=torch.complex128,
    )
    tb.diff_models[0].lag.ki = torch.tensor(
        np.random.uniform(0.01, 0.5, (parallel_sims, 1)),
        requires_grad=True,
        dtype=torch.complex128,
    )

    optimizable_parameters = [
        tb.diff_models[0].lag.kd,
        tb.diff_models[0].lag.ki,
    ]

    param_names = [
        "k_d",  # Geändert: korrekter Name
        "k_i",  # Geändert: korrekter Name
    ]

    params_original = [
        0.8,  # Original k_d Wert
        0.8,  # Original k_i Wert
    ]

    opt = PowerSystemOptimization(
        sim=tb,
        original_data=benchmark_transformed,
        params_optimizable=optimizable_parameters,
        params_original=params_original,
        param_names=param_names,
        max_step=max_step,
        decay=decay,
        verbose=True,
        # enable_plots=True,
    )

    torch.autograd.set_detect_anomaly(True)  # Bessere Fehlermeldungen

    result, opt_params, losses = opt.run(max_steps=n_opt)

    del opt
    del tb

    return result, opt_params, losses


if __name__ == "__main__":
    v_ref = 0.9
    input_mock = lambda t: jump_input(t, b=v_ref)

    # generate output directory and file
    output_dir = os.path.join(
        os.getcwd(),
        "data",
        # "optimization_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "optimization_data.yaml")

    result, opt_params, losses = optimize_param_set(
        n_opt=10, max_step=0.08, decay=0.9, input_func=input_mock
    )

    # open output_file
    with open(output_file, "a") as f:
        # save result, opt_params, min_loss to file in yaml format
        data = {
            f"iteration {v_ref}": {
                "v_ref": float(v_ref),
                "optimized_parameters": opt_params,
                "minimum_loss": losses,
                # "optimization_result": result
            }
        }
        # save outputs to file
        yaml.dump(data, f)
