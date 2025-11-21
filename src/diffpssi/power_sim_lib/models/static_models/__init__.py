"""Initialize package with static models."""

from diffpssi.power_sim_lib.models.static_models.bus import Bus
from diffpssi.power_sim_lib.models.static_models.line import Line
from diffpssi.power_sim_lib.models.static_models.load import Load
from diffpssi.power_sim_lib.models.static_models.param_event import (
    ParamDependencyTime,
    ParamEvent,
)
from diffpssi.power_sim_lib.models.static_models.sc_event import ScEvent
from diffpssi.power_sim_lib.models.static_models.shunt import Shunt
from diffpssi.power_sim_lib.models.static_models.static_model_interface import (
    StaticModelInterface,
)
from diffpssi.power_sim_lib.models.static_models.transformer_old import Transformer_Old
