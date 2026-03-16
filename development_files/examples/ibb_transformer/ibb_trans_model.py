"""
File contains the data for the IBB transformer model.
"""


def load(
    tap_side="hv",
    measure_side="hv",
    trans_type="simple",
    trans_control="oltc",
    param_dict_oltc=None,
    controllers=False,
):
    """
    Loads the grid data of the IBB transformer model.
    Returns: The grid data in form of a dictionary.

    """
    return {
        "base_mva": 2200,
        "f": 60,
        "slack_bus": "B0",
        "base_voltage": 100,
        "busses": [
            ["name", "V_n"],
            ["B0", 100],
            ["B1", 100],
            ["B2", 10],
        ],
        "lines": [
            ["name", "from_bus", "to_bus", "length", "unit", "R", "X", "B"],
            ["Line 1", "B0", "B1", 1, "p.u.", 0, 0.0484, 0],
        ],
        "transformers": [
            [
                "type",
                "control",
                "name",
                "from_bus",
                "to_bus",
                "S_n",
                "V_n_from",
                "V_n_to",
                "tap_side",
                "measure_side",
                "R",
                "X",
                "theta",
                "param_dict_oltc",
            ],
            [
                trans_type,
                trans_control,
                "T1",
                "B1",
                "B2",
                2200,
                100,
                10,
                tap_side,
                measure_side,
                0,
                0.15,
                0,
                param_dict_oltc,
            ],
        ],
        "generators": {
            "GEN": [
                [
                    "name",
                    "bus",
                    "S_n",
                    "V_n",
                    "P",
                    "V",
                    "H",
                    "D",
                    "X_d",
                    "X_q",
                    "X_d_t",
                    "X_q_t",
                    "X_d_st",
                    "X_q_st",
                    "T_d0_t",
                    "T_q0_t",
                    "T_d0_st",
                    "T_q0_st",
                ],
                [
                    "G1",
                    "B0",
                    5000,
                    100,
                    -1998,
                    0.995,
                    3.5e7,
                    0,
                    1.81,
                    1.76,
                    0.3,
                    0.65,
                    0.23,
                    0.23,
                    8.0,
                    1,
                    0.03,
                    0.07,
                ],
                [
                    "G2",
                    "B2",
                    2200,
                    10,
                    1998,
                    1,
                    3.5,
                    0,
                    1.81,
                    1.76,
                    0.3,
                    0.65,
                    0.23,
                    0.23,
                    8.0,
                    1,
                    0.03,
                    0.07,
                ],
            ],
            # 'SOURCE': [
            #     [], # description dict
            #     [], # value dict
            # ],
        },
        # 'loads': {
        #     'ZIP': [
        #         ['name', 'bus', 'P', 'Q', 'model'],
        #         ['L1', 'B1', 100, 0, 'Z'],
        #     ],
        #     # 'shunts': [
        #     #     ['name', 'bus', 'V_n', 'Q', 'model'],
        #     #     ['S1', 'Bus 2', 100, 1, 'Z'],
        #     # ],
        #     # 'InductionMachine': [ # -> Values which do make sense are included in Cutsem und Vournas 1998 S.101ff.
        #     #     ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'R_s', 'R_r', 'X_s', 'X_r', 'X_m', 'T_r0'],
        #     #     ['IM1', 'Bus 1', 2000, 100, -1998, 1, 3.5e7, 0, 0.024, 0.024, 75e-6, 75e-6, 754e-6, 0.2],
        #     # ],
        #     # 'SynchronousMachine': [
        #     #     ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st',
        #     #      'T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st'],
        #     #     ['G1', 'Bus 0', 11000, 100, -1998, 0.995, 3.5e7, 0, 1.81, 1.76, 0.3, 0.65, 0.23, 0.23, 8.0, 1, 0.03,
        #     #      0.07],
        #     # ],
        # },
    }
