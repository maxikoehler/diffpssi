"""File contains an example of how to simulate the K2A model."""

import examples.models.k2a.k2a_model as mdl
import matplotlib.pyplot as plt
import numpy as np

from diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss


def record_desired_parameters(simulation):
    """Record desired parameters from the simulation.

    Args:
        simulation: The simulation object.

    Returns:
        list: Recorded parameters.
    """
    # Record the desired parameters
    record_list = [
        simulation.busses[0].models[0].delta.real,
        simulation.busses[1].models[0].delta.real,
        simulation.busses[2].models[0].delta.real,
        simulation.busses[3].models[0].delta.real,
        simulation.busses[0].models[0].omega.real,
        simulation.busses[1].models[0].omega.real,
        simulation.busses[2].models[0].omega.real,
        simulation.busses[3].models[0].omega.real,
        simulation.busses[0].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[2].models[0].e_q_st.real,
        simulation.busses[3].models[0].e_q_st.real,
        simulation.busses[0].models[0].e_d_st.real,
        simulation.busses[1].models[0].e_d_st.real,
        simulation.busses[2].models[0].e_d_st.real,
        simulation.busses[3].models[0].e_d_st.real,
    ]
    return record_list


def main():
    """Simuliert das K2A-Modell und speichert die Ergebnisse."""
    parallel_sims = 1

    sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=10,
        time_step=0.005,
        solver="heun",
        grid_data=mdl.load(),
    )

    sim.add_sc_event(1, 1.1, "B1")

    sim.set_record_function(record_desired_parameters)
    t, recorder = sim.run()

    # Format shall be [batch, timestep, value]
    # create a new subplot for each parameter
    plt.figure()

    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel("Parameter {}".format(i))
        plt.xlabel("Time [s]")

    plt.savefig("data/plots/original.png".format())
    plt.show()

    # add time to the first column
    saver = np.zeros((len(t), len(recorder[0, 0, :]) + 1))
    saver[:, 0] = t
    saver[:, 1:] = recorder[0, :, :].real

    np.save("data/original_data_t.npy", saver)
    np.save("data/original_data.npy", recorder[0].real)


if __name__ == "__main__":
    main()
