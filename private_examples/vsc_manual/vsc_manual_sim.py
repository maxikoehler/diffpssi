"""
This example shows how to manually create and simulate the IBB model.
"""

import os
import sys

parent = os.path.abspath("/Users/maxikoehler/Documents/GitHub/diffpssi-ma-kohler/")
sys.path.insert(1, parent)
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

from src.diffpssi.power_sim_lib.models.machines import SynchMachine
from src.diffpssi.power_sim_lib.models.inverters import SimpleInverter
from src.diffpssi.power_sim_lib.models.static_models import *


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    # Record the desired parameters
    record_list = [
        simulation.busses[1].models[0].p,
        simulation.busses[1].models[0].q,
        abs(simulation.busses[1].models[0].i_d),
        abs(simulation.busses[1].models[0].i_q),
        abs(simulation.busses[1].models[0].v_bb),
    ]
    return record_list


def main():
    """
    This function simulates the IBB model.
    """
    parallel_sims = 1

    sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=2,
        time_step=0.001,
        trans_model="old",
        solver="rk4",
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
    sim.add_inverter(
        SimpleInverter(
            f_n_sys=60,
            s_n_sys=2200,
            v_n_sys=24,
            bus="Bus 1",
            s_n=2200,
            v_n=24,
            p_setp=500,
            q_setp=-100,
            k_p_p=1,
            k_i_p=1,
            k_p_q=1,
            k_i_q=1,
        )
    )

    sim.set_slack_bus("Bus 0")

    # sim.add_sc_event(1, 1.05, 'Bus 1')
    sim.add_param_event(1, sim.busses[1].models[0], "p_setp", 100)

    sim.set_record_function(record_desired_parameters)

    # Run the simulation. Recorder format shall be [batch, timestep, value]
    t, recorder = sim.run()

    param_names = ["P", "Q", "I_d", "I_q", "V_bb"]

    # Plot the results
    plt.figure(figsize=(6, 10))
    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel(param_names[i])
        plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

    if os.environ.get("DIFFPSSI_TESTING") == "True":
        np.save("./data/original_data.npy", recorder[0].real)


if __name__ == "__main__":
    main()
