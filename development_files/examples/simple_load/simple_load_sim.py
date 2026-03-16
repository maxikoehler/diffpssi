"""
This example shows a stability assessment for an simple example grid.
"""

import numpy as np
from src.diffpssi.power_sim_lib.backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from tools.colors import *

# Set the matplotlib settings
plt.rcParams.update(
    {
        "font.family": "Charter",
        "font.size": 12,
        "figure.autolayout": True,
        "text.usetex": True,
    }
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[ees_blue, ees_yellow, ees_green, ees_red, ees_lightblue]
)

import development_files.examples.simple_load.simple_load_mod as mdl
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss
from src.diffpssi.power_sim_lib.simulator import Recorder
from tools.calcs import *


def record_dict(simulation, call=False):
    record_dict = {
        "Bus 0: voltage magnitude": simulation.busses[0].get_value("voltage_mag"),
        "Bus 1: voltage magnitude": simulation.busses[1].get_value("voltage_mag"),
        # OLTC details section
        r"Transformer $u_\mathrm{l}$": simulation.trafos[0].oltc.u_l,
        r"Voltage difference $v_\mathrm{diff}$": simulation.trafos[0].oltc.v_diff,
        # r'Transformer $m$':                         simulation.trafos[0].oltc.tap_pos_m,
        # r'Integrator $m$':                          simulation.trafos[0].oltc.integrator_m.state_1,
        # Additional FSM details section
        # r'Transformer FSM ratio $k$':                           simulation.trafos[0].oltc.tap_pos_k,
        # r'Integrator $k$':                                      simulation.trafos[0].oltc.integrator_k.state_1,
    }
    if call:
        return record_dict.values()
    else:
        return record_dict


def main():
    """
    This function simulates the IBB transformer model.
    """
    parallel_sims = 1

    param_dict_fsm = {
        "t_m": 0.02,
        "t_k": 5,
        "db": 0.025,
        "delta_m": 2,
        "delta_k": 0.02,
        "m_max": 4,
        "m_min": -4,
        "k_max": 10,
        "k_min": -10,
        "v_ref": 1,
        "gamma_max": 8,
        "pt_1": 0.01,
    }

    param_dict_oltc = {
        "t_1": 5,
        "db": 0.05,
        "delta_m": 0.02,
        "m_max": 1.1,
        "m_min": 0.9,
        "v_ref": 1,
        # 'tap_pos_m':    0,
    }

    sim = Pss(
        parallel_sims=parallel_sims,
        sim_time=120,
        time_step=0.005,
        solver="heun",
        grid_data=mdl.load(
            tap_side="lv",
            measure_side="hv",
            trans_type="oltc",
            trans_control="oltc",
            param_dict_oltc=param_dict_oltc,
        ),
    )

    sim.trafos[0].from_bus = "lv"

    # sim.add_sc_event(1, 1.05, 'Bus 0')
    # sim.add_param_event(1, sim.busses[1].models[0], 'p_soll_mw', 3*sim.busses[1].models[0].p_soll_mw)

    rec = Recorder(sim=sim, recorder_dict=record_dict)

    sim.set_record_function(rec.record_fun)
    record_list = rec.record_list()

    t, recorder = sim.run()

    # # Very simple plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(t, recorder[0, :, 0], label=record_list[0])
    # plt.plot(t, recorder[0, :, 1], label=record_list[1])
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Time in [s]')
    # plt.ylabel('Voltage Magnitude in [p.u.]')
    # plt.show()

    # Voltages plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for i in range(0, 2):
        axs[0].plot(t, recorder[0, :, i], label=record_list[i])
    axs[0].grid()
    axs[0].legend()
    # axs[0].set_ylim([0, 2])

    for i in range(2, 4):
        axs[1].plot(t, recorder[0, :, i], label=record_list[i])
    axs[1].grid()
    axs[1].legend()

    fig.supylabel("Voltage Magnitude in [p.u.]")
    fig.supxlabel("Time in [s]")
    plt.show()

    # # OLTC plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(t, recorder[0, :, 4], label=record_list[4])
    # plt.plot(t, recorder[0, :, 5], label=record_list[5])
    # # Additional FSM details
    # # plt.plot(t, recorder[0, :, 6], label=record_list[4])
    # # plt.plot(t, recorder[0, :, 7], label=record_list[5])

    # plt.grid()
    # plt.legend()
    # plt.xlabel('Time in [s]')
    # plt.show()


if __name__ == "__main__":
    main()
