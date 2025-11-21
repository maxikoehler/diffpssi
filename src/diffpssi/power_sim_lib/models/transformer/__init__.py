"""Initialize all relevant stuff for the transformer models."""

from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.static_models import Transformer_Old
from diffpssi.power_sim_lib.models.transformer.oltc_transformer import OLTC_Transformer
from diffpssi.power_sim_lib.models.transformer.simple_transformer import (
    Simple_Transformer,
)
from diffpssi.power_sim_lib.models.transformer.transformer_interface import Transformer

transformer_type_dict = {
    "old": Transformer_Old,  # from models.static_models
    "simple": Simple_Transformer,
    "oltc": OLTC_Transformer,
}
