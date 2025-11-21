"""Initialize all relevant stuff for the voltage controller models."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.voltage_controller.fsm_discrete import FSM_Discrete
from diffpssi.power_sim_lib.models.voltage_controller.fsm_discrete_2 import (
    FSM_Discrete_2,
)
from diffpssi.power_sim_lib.models.voltage_controller.fsm_discrete_tap_skip import (
    FSM_Discrete_Tap_Skip,
)
from diffpssi.power_sim_lib.models.voltage_controller.oltc_continuous import (
    OLTC_Continuous,
)
from diffpssi.power_sim_lib.models.voltage_controller.oltc_discrete import OLTC_Discrete
from diffpssi.power_sim_lib.models.voltage_controller.voltage_controller_interface import (
    Voltage_Controller,
)

voltage_control_dict = {
    # Dictionary mapping controller type keys to controller classes.
    "oltc": OLTC_Discrete,
    "fsm": FSM_Discrete,
    "fsm_2": FSM_Discrete_2,
    "fsm_tap": FSM_Discrete_Tap_Skip,
    "oltc_c": OLTC_Continuous,
}
