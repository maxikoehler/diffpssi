"""Run optimization study for FSM continuous controller parameters."""

import os
import sys

import numpy as np
import torch
import yaml
from opt_param_set import *
from tqdm import tqdm


def main():
    """Main function to run the optimization study."""
    n_opt = 10  # Number of optimization steps
    max_step = 0.1  # Maximum step size for parameter updates
    decay = 0.9  # Decay factor for step size

    start_step = -0.3  # Starting step size for input function
    end_step = 0.3  # Ending step size for input function
    step_size = 0.1  # Step size for input function

    # Generate list of input function parameters
    steps = np.arange(start_step, end_step + step_size, step_size)
    print(f"Steps for input function: {steps}")

    # generate output directory and file
    output_dir = os.path.join(
        os.getcwd(),
        "data",
        # "optimization_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "optimization_data.yaml")

    # Run optimization(s)
    for i, step in enumerate(steps):
        print("------------------------------------------------")
        print(f"Running optimization for v_ref: {np.round(step, 3)}")
        if np.round(step, 3) == 0:
            # catch errors from nan problems when step is 0
            continue

        input_mock = lambda t: jump_input(t, b=(1 + np.round(step, 3)))
        # input_mock = lambda t: duplet_input(t, b=step)
        # input_mock = lambda t: duplet_input(t, b=step)

        result, opt_params, losses = optimize_param_set(
            n_opt=n_opt, max_step=max_step, decay=decay, input_func=input_mock
        )

        # open output_file
        with open(output_file, "a") as f:
            # save result, opt_params, min_loss to file in json format
            import json

            data = {
                f"step {i}": {
                    "v_ref": float(step),
                    "optimized_parameters": opt_params,
                    "minimum_loss": losses,
                    # "optimization_result": result
                }
            }
            # save outputs to file
            yaml.dump(data, f)

        del input_mock

        # close file


if __name__ == "__main__":
    main()
