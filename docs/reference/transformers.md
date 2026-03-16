The transformer is generally connecting two nodes, therefore it is similar to lines, a two port object. Further, there are different ways to model it.
First is assuming a reactance, but constant and ideal transforming ration of 1 p.u. The second incorporates branches, to model for varying (complex) ratios, which can also be modelled as a on-load tap-changer, actively changing its ratio based on a controller.

Therefore, the following models are implemented in the package:

1. Idealized transformer model (Old Transformer Model)
2. Simple Pi Model
3. OLTC Transformer Model

## Old Transformer Model 

::: diffpssi.power_sim_lib.models.transformer.transformer_old.Transformer_Old
    options: 
        show_root_heading: false
        heading_level: 3
        members: 
            - __init__
            - calc_admittance
            - get_value
        show_source: false

## Variable Pi-Transformer Model 

::: diffpssi.power_sim_lib.models.transformer.transformer_interface.Transformer
    options: 
        show_root_heading: false
        heading_level: 3
        members: 
            - __init__
            - calc_admittance
            - get_value
        show_source: false

::: diffpssi.power_sim_lib.models.transformer.simple_transformer.Simple_Transformer
    options: 
        show_root_heading: false
        heading_level: 3
        members: 
            - __init__
            - calc_admittance
            - get_value
        show_source: false

::: diffpssi.power_sim_lib.models.transformer.oltc_transformer.OLTC_Transformer
    options: 
        show_root_heading: false
        heading_level: 3
        members: 
            - __init__
            - calc_admittance
            - get_value
        show_source: false
