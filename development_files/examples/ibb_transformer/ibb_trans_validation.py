"""
Validation of the simple transformer model using the IBB model and compare it to the result of PowerFactory.
"""

import numpy as np
from ibb_trans_model import load
from tools.colors import *
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

import matplotlib.pyplot as plt
import matplotlib as mpl


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    # Record the desired parameters
    record_list = [
        # simulation.busses[0].voltage.real,
        # simulation.busses[1].voltage.real,
        # simulation.busses[2].voltage.real,
        # simulation.busses[0].voltage.imag,
        # simulation.busses[1].voltage.imag,
        # simulation.busses[2].voltage.imag,
        simulation.busses[0].get_value("voltage_mag"),
        simulation.busses[1].get_value("voltage_mag"),
        simulation.busses[2].get_value("voltage_mag"),
    ]
    return record_list


def do_simulation():
    """
    This function simulates the IBB transformer model.
    """
    parallel_sims = 1

    sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=5,
        time_step=0.005,
        solver="rk4",
        grid_data=load(),
    )

    sim.add_sc_event(1, 1.05, "Bus 0")

    sim.set_record_function(record_desired_parameters)
    t, recorder = sim.run()
    return t, recorder


# External data for comparison (example data)
external_file_path = (
    "./examples/master_thesis/ibb_transformer/data/ibb_sim_simple-transformer_valid.csv"
)
external_data = np.genfromtxt(external_file_path, delimiter=";", skip_header=1)
external_data = external_data[:, 1:4]

# Perform simulation
time, model_voltages = do_simulation()

# Calculate mean error
mean_errors = np.zeros((3, len(time)))
for i in range(3):
    ext_voltages = external_data[:, i]
    model_voltage = model_voltages[0, :, i]
    errors = [
        abs(ext_v - model_v) for model_v, ext_v in zip(model_voltage, ext_voltages)
    ]
    mean_errors[i] = errors

# Set the matplotlib settings
plt.rcParams["font.family"] = "Charter"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.autolayout"] = True
plt.rcParams["text.usetex"] = True
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[ees_blue, ees_yellow, ees_green, ees_red, ees_lightblue]
)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

for i in range(3):
    row, col = divmod(i, 2)
    axs[row, col].plot(time, external_data[:, i], label="PowerFactory Data")
    axs[row, col].plot(
        time, model_voltages[0, :, i], linestyle="-", label="diffpssi Data"
    )
    axs[row, col].set_title(f"Voltage Comparison for Bus {i}")
    axs[row, col].set_xlabel("Time (s)")
    axs[row, col].set_ylabel("Voltage in p.u.")
    axs[row, col].legend()
    axs[row, col].grid(True)

# Add mean error plot
axs[1, 1].plot(time, mean_errors[0], label="Error for Bus 0")
axs[1, 1].plot(time, mean_errors[1], label="Error for Bus 1")
axs[1, 1].plot(time, mean_errors[2], label="Error for Bus 2")
axs[1, 1].set_ylabel("Error in p.u.")
axs[1, 1].set_title("Absolute Error Comparison")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# plt.tight_layout()
# plt.savefig('./examples/master_thesis/ibb_transformer/data/comp_simple_pi.pdf')
plt.show()

# Print mean errors
# for bus, error in mean_errors.items():
#     print(f'Mean error for {bus}: {error*100:.2f} %')
