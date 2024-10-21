import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

from functions.utils import _get_input_y_n


def select_sample(signal: np.ndarray, sf: int, color1: str, color2: str):
    """
    This function allows the user to select a sample from a plot representing
    the given signal with the sampling frequency provided.
    The user can zoom in and out, and the last click before answering
    y will be the selected sample.

    Inputs:
    signal: np.ndarray, the signal to plot
    sf: int, the sampling frequency of the plotted signal
    color1: str, the color to plot the signal as a line
    color2: str, the color to plot the signal scattered

    Returns:
    closest_value: float, the manually selected sample
    """

    signal_timescale_s = np.arange(0, (len(signal) / sf), (1 / sf))
    selected_x = interaction(
        data=signal, timescale=signal_timescale_s, color1=color1, color2=color2
    )

    # Find the index of the closest value
    closest_index = np.argmin(np.abs(signal_timescale_s - selected_x))

    # Get the closest value
    closest_value = signal_timescale_s[closest_index]

    return closest_value


def interaction(data: np.ndarray, timescale: np.ndarray, color1: str, color2: str):
    """
    This function draws an interactive plot representing the given data with
    the timescale provided. The user can zoom in and out.
    """

    # collecting the clicked x and y values
    pos = []

    fig, ax = plt.subplots()
    ax.plot(timescale, data, c=color1, zorder=1)
    ax.scatter(timescale, data, s=8, c=color2, zorder=2)
    ax.set_title(
        "Click on the plot to select the sample \n"
        "where the artifact starts. You can use the zoom, \n"
        'as long as the black "+" is placed on the correct sample \n'
        'before answering "y" in the terminal'
    )

    (plus_symbol,) = ax.plot([], [], "k+", markersize=10)

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            pos.append([event.xdata, event.ydata])

            # Update the position of the black "+" symbol
            closest_index_x = np.argmin(np.abs(timescale - event.xdata))
            closest_value_x = timescale[closest_index_x]
            closest_value_y = data[closest_index_x]
            plus_symbol.set_data(closest_value_x, closest_value_y)
            plt.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    # plt.show(block=False)
    plt.show()
    condition_met = False

    input_y_or_n = _get_input_y_n("Artifact found?")

    while not condition_met:
        if input_y_or_n == "y":
            condition_met = True
        else:
            input_y_or_n = _get_input_y_n("Artifact found?")

    artifact_x = [x_list[0] for x_list in pos]  # list of all clicked x values

    return artifact_x[-1]
