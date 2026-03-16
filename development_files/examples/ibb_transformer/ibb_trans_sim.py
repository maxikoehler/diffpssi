"""
This example shows how to simulate the IBB transformer model.
"""

import numpy as np
import sys, os
import pandas as pd
from src.diffpssi.power_sim_lib.backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl

parent = os.path.abspath("..")
sys.path.insert(1, parent)
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

import development_files.examples.ibb_transformer.ibb_trans_model as mdl
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss
from src.diffpssi.power_sim_lib.simulator import Recorder
from tools.calcs import *


def record_dict(simulation, call=False):
    record_dict = {
        "Bus 0: voltage magnitude": simulation.busses[0].get_value("voltage_mag"),
        "Bus 1: voltage magnitude": simulation.busses[1].get_value("voltage_mag"),
        "Bus 2: voltage magnitude": simulation.busses[2].get_value("voltage_mag"),
        # 'Determinant of Jacobi-M':      simulation.jacobian_matrix,
        # 'Transformer m':                simulation.trafos[0].oltc.m,
        # 'P ibb machine':                    simulation.busses[0].models[0].p_e,
        # 'P small machine':                    simulation.busses[2].models[0].p_e,
        # r'Transformer $u_\mathrm{l}$':  simulation.trafos[0].u_l,
        # 'Real power at Load 1':         simulation.busses[0].models[0].get_value('P_atm', sim=simulation),
        # simulation.trafos[0].oltc.integrator.state_1 / simulation.trafos[0].oltc.t_1,
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
            tap_side="hv",
            measure_side="hv",
            trans_type="oltc",
            trans_control="fsm",
            param_dict_oltc=param_dict_fsm,
        ),
        #   jacobi_calculation=True,
        #   verbose=False,
    )

    start = 1
    end = 1.05
    # sim.add_sc_event(start, end, 'Bus 0')
    sim.add_param_event(
        1, sim.busses[1].models[0], "p_soll_mw", 6 * sim.busses[1].models[0].p_soll_mw
    )

    rec = Recorder(sim=sim, recorder_dict=record_dict)

    sim.set_record_function(rec.record_fun)
    record_list = rec.record_list()

    t, recorder = sim.run()

    # Format shall be [batch, timestep, value]

    plt.figure(figsize=(12, 8))
    for i in range(len(record_list)):
        plt.plot(t, recorder[0, :, i], label=record_list[i])

    plt.grid()
    plt.legend()
    # plt.ylim(0.6, 1.4)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage magnitude [p.u.]")

    plt.show()

    # fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # for i in range(len(record_list)-2):
    #     axs[0].plot(t, recorder[0, :, i], label=record_list[i])

    # axs[0].grid()
    # axs[0].set_ylim(0.6, 1.4)
    # axs[0].legend()

    # axs[1].plot(t, recorder[0, :, -1], label=record_list[-1])
    # axs[1].plot(t, recorder[0, :, -2], label=record_list[-2])
    # # axs[1].axvline(start, color='red', linestyle='--', label='SC event start')
    # # axs[1].axvline(end, color='red', linestyle='--', label='SC event end')
    # axs[1].grid()
    # axs[1].legend()

    # plt.show()

    # data = pd.DataFrame(recorder[0, :, :])
    # data.columns = record_list
    # data['t'] = t
    # data.set_index('t', inplace=True)

    # data.to_csv('./examples/master_thesis/ibb_transformer/data/ibb_transformer.csv')


if __name__ == "__main__":
    main()
