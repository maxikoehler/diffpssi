"""Matplotlib settings for DiffPSSI plots."""

import matplotlib as mpl
import matplotlib.pyplot as plt

from diffpssi.tools.colors import (
    ees_black,
    ees_blue,
    ees_green,
    ees_lightblue,
    ees_red,
    ees_yellow,
)


# Set the matplotlib settings
def set_matplot_settings(
    font_family="Charter",
    font_size=14,
    **kwargs,
):
    """Set custom matplotlib settings, standard for the diffpssi package.

    Args:
        font_family (str): Font family to use in plots.
        font_size (int): Font size to use in plots.
    """
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "figure.autolayout": True,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            **kwargs,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[ees_blue, ees_yellow, ees_green, ees_red, ees_lightblue]
    )
