'''
import json
import matplotlib.pyplot as plt
from os.path import join

from functions.interactive import select_sample
from functions.utils import _update_and_save_multiple_params, _detrend_data
'''
import numpy as np
import scipy
from matplotlib.backend_bases import MouseButton


def compute_timeshift(self):
    timeshift = (self.dataset_extra.last_artifact - self.dataset_intra.last_artifact)*1000
    self.label_timeshift.setText(f"Timeshift: {timeshift} ms")


def select_first_artifact_intra_eff_sf_correction(self):
    # Check if we're already in selection mode of the last and prevent interference
    if hasattr(self, 'cid_intra_last') and self.cid_intra_last is not None:
        self.canvas_intra_sf.mpl_disconnect(self.cid_intra_last)
        #self.cid_intra_last = None

    pos = []

    self.ax_intra_sf.set_title(
        'Right click on the plot to select the start of the first artifact (shown by the black "+")'
    )
    #self.canvas_intra_sf.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_intra_first'):
        self.plus_symbol_intra_first, = self.ax_intra_sf.plot([], [], "k+", markersize=10)


    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(self.dataset_intra.times - event.xdata))
                closest_value_x = self.dataset_intra.times[closest_index_x]
                closest_value_y = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index][closest_index_x]
                self.plus_symbol_intra_first.set_data([closest_value_x], [closest_value_y])
                self.canvas_intra_sf.draw()

                self.dataset_intra.first_art_start_time = closest_value_x
                #self.dataset_intra.first_art_start_idx = closest_index_x
                self.dataset_intra.first_art_start_idx = int(np.round(closest_value_x * self.dataset_intra.sf))
                self.label_time_select_first_intra.setText(f"Selected Artifact start: {np.round(closest_value_x, decimals=3)} s")
                self.label_sample_select_first_intra.setText(f"Sample n# {int(self.dataset_intra.first_art_start_idx)}")
                self.update_compute_eff_sf_button_state()

    self.cid_intra_first = self.canvas_intra_sf.mpl_connect("button_press_event", onclick)


def select_last_artifact_intra_eff_sf_correction(self):
    # Check if we're already in external selection mode and prevent interference
    if hasattr(self, 'cid_intra_first') and self.cid_intra_first is not None:
        self.canvas_intra_sf.mpl_disconnect(self.cid_intra_first)
        #self.cid_intra_first = None

    pos = []

    self.ax_intra_sf.set_title(
        'Right click on the plot to select the start of the last artifact (shown by the red "+")'
    )
    #self.canvas_intra_sf.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_intra_last'):
        self.plus_symbol_intra_last, = self.ax_intra_sf.plot([], [], "r+", markersize=10)

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(self.dataset_intra.times - event.xdata))
                closest_value_x = self.dataset_intra.times[closest_index_x]
                closest_value_y = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index][closest_index_x]
                self.plus_symbol_intra_last.set_data([closest_value_x], [closest_value_y])
                self.canvas_intra_sf.draw()
                self.dataset_intra.last_art_start_time = closest_value_x
                self.dataset_intra.last_art_start_idx = int(np.round(closest_value_x * self.dataset_intra.sf))
                self.label_time_select_last_intra.setText(f"Selected Artifact start: {np.round(closest_value_x, decimals=3)} s")
                self.label_sample_select_last_intra.setText(f"Sample n# {int(self.dataset_intra.last_art_start_idx)}")
                self.update_compute_eff_sf_button_state()

    self.cid_intra_last = self.canvas_intra_sf.mpl_connect("button_press_event", onclick)   



def select_first_artifact_extra_eff_sf_correction(self):
    # Check if we're already in selection mode of the last and prevent interference
    if hasattr(self, 'cid_extra_last') and self.cid_extra_last is not None:
        self.canvas_extra_sf.mpl_disconnect(self.cid_extra_last)


    pos = []

    self.ax_extra_sf.set_title(
        'Right click on the plot to select the start of the first artifact (shown by the black "+")'
    )
    #self.canvas_extra_sf.draw()

    # Create or update the extracranial "+" symbol
    if not hasattr(self, 'plus_symbol_extra_first'):
        self.plus_symbol_extra_first, = self.ax_extra_sf.plot([], [], "k+", markersize=10)
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data = scipy.signal.filtfilt(b, a, self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index])

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(self.dataset_extra.times - event.xdata))
                closest_value_x = self.dataset_extra.times[closest_index_x]
                closest_value_y = data[closest_index_x]
                self.plus_symbol_extra_first.set_data([closest_value_x], [closest_value_y])
                self.canvas_extra_sf.draw()

                self.dataset_extra.first_art_start_time = closest_value_x
                self.dataset_extra.first_art_start_idx = closest_index_x 
                self.label_time_select_first_extra.setText(f"Selected Artifact start: {np.round(closest_value_x, decimals=8)} s")
                self.label_sample_select_first_extra.setText(f"Sample n# {int(self.dataset_extra.first_art_start_idx)}")
                self.update_compute_eff_sf_button_state()

    self.cid_extra_first = self.canvas_extra_sf.mpl_connect("button_press_event", onclick)


def select_last_artifact_extra_eff_sf_correction(self):
    # Check if we're already in external selection mode and prevent interference
    if hasattr(self, 'cid_extra_first') and self.cid_extra_first is not None:
        self.canvas_extra_sf.mpl_disconnect(self.cid_extra_first)
        #self.cid_intra_first = None

    pos = []

    self.ax_extra_sf.set_title(
        'Right click on the plot to select the start of the last artifact (shown by the red "+")'
    )
    #self.canvas_extra_sf.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_extra_last'):
        self.plus_symbol_extra_last, = self.ax_extra_sf.plot([], [], "r+", markersize=10)
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data = scipy.signal.filtfilt(b, a, self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index])
    
    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(self.dataset_extra.times - event.xdata))
                closest_value_x = self.dataset_extra.times[closest_index_x]
                closest_value_y = data[closest_index_x]
                self.plus_symbol_extra_last.set_data([closest_value_x], [closest_value_y])
                self.canvas_extra_sf.draw()
                self.dataset_extra.last_art_start_time = closest_value_x
                self.dataset_extra.last_art_start_idx = closest_index_x
                self.label_time_select_last_extra.setText(f"Selected Artifact start: {np.round(closest_value_x, decimals=8)} s")
                self.label_sample_select_last_extra.setText(f"Sample n# {int(self.dataset_extra.last_art_start_idx)}")
                self.update_compute_eff_sf_button_state()

    self.cid_extra_last = self.canvas_extra_sf.mpl_connect("button_press_event", onclick)    




def compute_eff_sf(self):
    time_interval = self.dataset_extra.last_art_start_time - self.dataset_extra.first_art_start_time
    print(f"time interval: {time_interval}")
    sample_interval = self.dataset_intra.last_art_start_idx - self.dataset_intra.first_art_start_idx
    print(f"sample interval: {sample_interval}")
    self.dataset_intra.eff_sf = sample_interval/time_interval
    self.dataset_intra.sf = self.dataset_intra.eff_sf
    self.dataset_intra.times = np.linspace(0, self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.raw_data.get_data().shape[1])
    print(f"eff_sf: {self.dataset_intra.eff_sf}")
    self.label_eff_sf.setText(f"The effective sampling frequency of the intracranial recording is actually {self.dataset_intra.eff_sf} and will be used for synchronization.")


#################### not validated for GUI ####################################
'''
def check_timeshift(
    session_ID: str,
    LFP_synchronized: np.ndarray,
    sf_LFP: int,
    external_synchronized: np.ndarray,
    sf_external: int,
    saving_path: str,
):
    """
    Check the timeshift between the intracranial and external recordings after
    synchronization. As the two recording systems are different, it may happen
    that the internal clocks are not completely identical. This function allows
    to check this and to warn in case of a large timeshift.
    To do so, the function plots the intracranial recording and the external one.
    On each plot, the user is asked to select the sample corresponding to the
    last artifact in the recording. The function then computes the time difference
    between the two times. If the difference is large, it may indicate a problem
    in the recording, such as a packet loss in the intracranial recording.

    Inputs:
        - session_ID: str, the subject ID
        - LFP_synchronized: np.ndarray, the intracranial recording containing all
        recorded channels
        - sf_LFP: int, sampling frequency of intracranial recording
        - external_synchronized: np.ndarray, the external recording containing all
        recorded channels
        - sf_external: int, sampling frequency of external recording
        - saving_path: str, path to the folder where the parameters.json file is
        saved

    """

    # import settings
    json_filename = saving_path + "\\parameters_" + str(session_ID) + ".json"
    with open(json_filename, "r") as f:
        loaded_dict = json.load(f)

    LFP_channel_offset = LFP_synchronized[loaded_dict["CH_IDX_LFP"], :]
    print(LFP_channel_offset.shape)
    print(type(LFP_channel_offset))
    BIP_channel_offset = external_synchronized[loaded_dict["CH_IDX_EXTERNAL"], :]
    print(BIP_channel_offset.shape)
    print(type(BIP_channel_offset))

    # Generate new timescales:
    LFP_timescale_offset_s = np.arange(
        start=0, stop=len(LFP_channel_offset) / sf_LFP, step=1 / sf_LFP
    )
    external_timescale_offset_s = np.arange(
        start=0, stop=len(BIP_channel_offset) / sf_external, step=1 / sf_external
    )

    # detrend external recording with high-pass filter before processing:
    filtered_external_offset = _detrend_data(BIP_channel_offset)

    print("Select the first sample of the last artifact in the intracranial recording")
    last_artifact_lfp_x = select_sample(
        signal=LFP_channel_offset, sf=sf_LFP, color1="peachpuff", color2="darkorange"
    )
    print("Select the first sample of the last artifact in the external recording")
    last_artifact_external_x = select_sample(
        signal=filtered_external_offset,
        sf=sf_external,
        color1="paleturquoise",
        color2="darkcyan",
    )

    timeshift_ms = (last_artifact_external_x - last_artifact_lfp_x) * 1000
    
    dictionary = {"LAST_ART_IN_EXT": last_artifact_external_x, "LAST_ART_IN_LFP": last_artifact_lfp_x, "TIMESHIFT": timeshift_ms, "REC DURATION FOR TIMESHIFT": last_artifact_external_x}
    _update_and_save_multiple_params(dictionary, session_ID, saving_path)

    if abs(timeshift_ms) > 100:
        print(
            "WARNING: the timeshift is unusually high,"
            "consider checking for packet loss in LFP data."
        )

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(str(session_ID))
    fig.set_figheight(12)
    fig.set_figwidth(6)
    ax1.axes.xaxis.set_ticklabels([])
    ax2.set_xlabel("Time (s)")
    ax1.set_ylabel("Intracranial LFP channel (µV)")
    ax2.set_ylabel("External bipolar channel (mV)")
    ax1.set_xlim(last_artifact_external_x - 0.1, last_artifact_external_x + 0.1)
    ax2.set_xlim(last_artifact_external_x - 0.1, last_artifact_external_x + 0.1)
    ax1.plot(LFP_timescale_offset_s, LFP_channel_offset, color="peachpuff", zorder=1)
    ax1.scatter(
        LFP_timescale_offset_s, LFP_channel_offset, color="darkorange", s=4, zorder=2
    )
    ax1.axvline(
        x=last_artifact_lfp_x,
        ymin=min(LFP_channel_offset),
        ymax=max(LFP_channel_offset),
        color="black",
        linestyle="dashed",
        alpha=0.3,
    )
    ax2.plot(
        external_timescale_offset_s,
        filtered_external_offset,
        color="paleturquoise",
        zorder=1,
    )
    ax2.scatter(
        external_timescale_offset_s,
        filtered_external_offset,
        color="darkcyan",
        s=4,
        zorder=2,
    )
    ax2.axvline(
        x=last_artifact_external_x, color="black", linestyle="dashed", alpha=0.3
    )
    ax1.text(
        0.05,
        0.85,
        s="delay intra/exter: " + str(round(timeshift_ms, 2)) + "ms",
        fontsize=14,
        transform=ax1.transAxes,
    )

    plt.gcf()
    plt.show(block=True)
    fig.savefig(
        join(
            saving_path,
            "FigA-Timeshift - Intracranial and external recordings aligned - last artifact.png",
        ),
        bbox_inches="tight",
        dpi=1200,
    )
'''