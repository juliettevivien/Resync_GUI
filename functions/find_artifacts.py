# import librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from itertools import compress
import scipy

from matplotlib.backend_bases import MouseButton

# import custom-made functions
from functions.plotting import (
    plot_scatter_channel_intra, 
    plot_scatter_channel_external
    )
#from functions.interactive import select_sample
from functions.utils import _detrend_data, _find_closest_index



def detect_artifacts_external(self):
    """Detect artifacts in .xdf data."""
    channel_data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    chan_data_detrend = scipy.signal.filtfilt(b, a, channel_data)        
    self.dataset_extra.art_start = find_external_sync_artifact(data=chan_data_detrend, sf_external=self.dataset_extra.sf, times = self.dataset_extra.times, start_index=0)
    print(f"Artifact detected in BIP data at time: {self.dataset_extra.art_start} s")
    self.label_automatic_artifact_time_xdf.setText(f"Artifact start detected at: {self.dataset_extra.art_start} s")
    self.label_manual_artifact_time_xdf.setText("No artifact manually selected")

    # Plot the channel with artifact
    plot_scatter_channel_external(self, art_start_BIP=self.dataset_extra.art_start)
    self.update_synchronize_button_state()  # Check if we can enable the button


def manual_selection_external(self):
    self.toolbar_xdf.setEnabled(True)
    self.canvas_xdf.setEnabled(True)
    self.ax_xdf.clear()
    data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    channel_data_to_plot = scipy.signal.filtfilt(b, a, data)
    timescale = self.dataset_extra.times

    pos = []

    self.ax_xdf.scatter(timescale, channel_data_to_plot, s=8)
    self.canvas_xdf.draw()
    self.ax_xdf.set_title(
        'Right click on the plot to select the start of the artifact (shown by the black "+")'
    )

    (plus_symbol,) = self.ax_xdf.plot([], [], "k+", markersize=10)

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(timescale - event.xdata))
                closest_value_x = timescale[closest_index_x]
                closest_value_y = channel_data_to_plot[closest_index_x]
                plus_symbol.set_data([closest_value_x], [closest_value_y])
                self.canvas_xdf.draw()
                self.dataset_extra.art_start = closest_value_x
                self.label_manual_artifact_time_xdf.setText(f"Selected Artifact start: {closest_value_x} s")
                self.label_automatic_artifact_time_xdf.setText("No artifact automatically detected")
                self.update_synchronize_button_state()

    self.canvas_xdf.mpl_connect("button_press_event", onclick)

# Detection of artifacts in external
def find_external_sync_artifact(data: np.ndarray, sf_external, times, start_index=0):
    """
    Function that finds artifacts caused by increasing/reducing
    stimulation from 0 to 1mA without ramp.
    For correct functioning, the external data recording should
    start in stim-off, and typically short pulses are given
    (without ramping). The first 2 seconds are used for threshold calculation
    and should therefore be free of any artifact.
    The signal are pre-processed previously with a high-pass
    Butterworth filter (1Hz) to ensure removal of slow drifts
    and offset around 0 (using _detrend_data function in utils.py).

    Inputs:
        - data: np.ndarray, single external channel (from bipolar electrode)
        - sf_external: int, sampling frequency of external recording
        - times: np.ndarray, timescale of the signal
        - start_index: default is 0, which means that the function will start
            looking for artifacts from the beginning of the signal. If the
            function is called with a start_index different from 0, the function
            will start looking for artifacts from that index. This is useful when
            the recording was started in Stimulation ON or when there is another
            kind of artifact at the beginning of the recording
    Returns:
        - art_time_BIP: the timestamp where the artifact starts in external recording (in seconds)

    """

    """
    To be properly detected in external channel, artifacts have to look 
    like a downward deflection (they are more negative than positive). 
    If for some reason the data recorder picks up the artifact as an upward 
    deflection instead, then the signal has to be inverted before detecting 
    artifacts.
    """
    # check polarity of artifacts before detection:
    if abs(max(data[:-1000])) > abs(min(data[:-1000])):
        print("external signal is reversed")
        data = data * -1
        print("invertion undone")

    # find indexes of artifacts
    # the external sync artifact is a sharp downward deflexion repeated at a high
    # frequency (stimulation frequency). Therefore, the artifact is detected when
    # the signal is below the threshold, and when the signal is lower than the
    # previous and next sample (first peak of the artifact).
    Skipping = False
    if start_index != 0:
        Skipping = True

    art_time_BIP = None
    while art_time_BIP == None:
        # define thresh_BIP as 1.5 times the difference between the max and min
        if Skipping:
            thresh_BIP = -1.5 * (np.ptp(data[0:int(sf_external * 2)]))
        else:
            thresh_BIP = -1.5 * (np.ptp(data[start_index:(start_index + int(sf_external * 2))]))
        
        for q in range(start_index, len(data) - 2):
            if (
                (data[q] <= thresh_BIP)
                and (data[q] < data[q + 1])
                and (data[q] < data[q - 1])
            ):
                art_time_BIP = times[q]
                break
        start_index += 1*sf_external
        print(art_time_BIP)
        

    return art_time_BIP



def detect_artifacts_intra(self):
    thres_window = round(self.dataset_intra.sf * 2)
    data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
    thres = np.ptp(data[:thres_window])
    # Compute absolute value to be invariant to the polarity of the signal
    abs_data = np.abs(data)
    # Check where the data exceeds the threshold
    over_thres = np.where(abs_data[0:] > thres)[0][0]
    # Take last sample that lies within the value distribution of the thres_window 
    # before the threshold passing, and add 4 samples
    # The percentile is something that can be varied
    stim_idx = (np.where(
        abs_data[:over_thres] <= np.percentile(abs_data[:over_thres], 95)
    )[0][-1]) + 4
    self.dataset_intra.art_start = stim_idx / self.dataset_intra.sf
    print(f"Artifact detected in LFP data at time: {self.dataset_intra.art_start} s")
    plot_scatter_channel_intra(self, art_start_intra = self.dataset_intra.art_start)
    self.update_synchronize_button_state()  # Check if we can enable the button
    self.label_automatic_artifact_time_intra.setText(f"Artifact start: {self.dataset_intra.art_start} s")
    self.label_manual_artifact_time_intra.setText("No artifact manually selected")


def manual_selection_intra(self):
    self.toolbar_intra.setEnabled(True)
    self.canvas_intra.setEnabled(True)
    self.ax_intra.clear()
    data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
    timescale = self.dataset_intra.times

    pos = []

    self.ax_intra.scatter(timescale, data, s=8)
    self.canvas_intra.draw()
    self.ax_intra.set_title(
        'Right click on the plot to select the start of the artifact (shown by the black "+")'
    )

    (plus_symbol,) = self.ax_intra.plot([], [], "k+", markersize=10)

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(timescale - event.xdata))
                closest_value_x = timescale[closest_index_x]
                closest_value_y = data[closest_index_x]
                plus_symbol.set_data([closest_value_x], [closest_value_y])
                self.canvas_intra.draw()
                self.dataset_intra.art_start = closest_value_x
                self.label_automatic_artifact_time_intra.setText("No artifact automatically detected")
                self.label_manual_artifact_time_intra.setText(f"Selected Artifact start: {closest_value_x} s")
                self.update_synchronize_button_state()

    self.canvas_intra.mpl_connect("button_press_event", onclick)









###############################################################################
# not yet validated for the GUI
'''



# Detection of artifacts in LFP
def find_LFP_sync_artifact(data: np.ndarray, sf_LFP: int, use_method: str, start_index):
    """
    Function that finds artifacts caused by
    augmenting-reducing stimulation from 0 to 1mA without ramp.
    For correct functioning, the LFP data should
    start in stim-off, and typically short pulses
    are given (without ramping).
    The function can use two different methods:
    'thresh': a threshold computed on the baseline is used to detect the stim-artifact.
    '1' & '2' are 2 versions of a method using kernels. A kernel mimics the 
    stimulation artifact shape. This kernel is multiplied with time-series
    snippets of the same length. If the time-serie is
    similar to the kernel, the dot-product is high, and this
    indicates a stim-artifact.

    Input:
        - data: np.ndarray, single data channel of intracranial recording, containing
            the stimulation artifact
        - sf_LFP: int, sampling frequency of intracranial recording
        - use_method: str, '1' or '2' or 'thresh'. The kernel/method
            used to detect the stim-artifact.
                '1' is a simple kernel that only detects the steep decrease
            of the stim-artifact.
                '2' is a more complex kernel that takes into account the
            steep decrease and slow recover of the stim-artifact.
                'thresh' is a method that uses a threshold to detect
            the stim-artifact.

    Returns:
        - art_time_LFP: the timestamp where the artifact starts in the
        intracranial recording.
    """

    # checks correct input for use_kernel variable
    assert use_method in ["1", "2", "thresh"], "use_method incorrect. Should be '1', '2' or 'thresh'"

    if use_method == "thresh":
        thres_window = round(sf_LFP * 2)
        thres = np.ptp(data[:thres_window])
        # Compute absolute value to be invariant to the polarity of the signal
        abs_data = np.abs(data)
        # Check where the data exceeds the threshold
        rounded_start_index = round(start_index)
        over_thres = np.where(abs_data[rounded_start_index:] > thres)[0][0]
        if start_index != 0:
            over_thres += rounded_start_index
        # Take last sample that lies within the value distribution of the thres_window before the threshold passing
        # The percentile is something that can be varied
        stim_idx = np.where(
            abs_data[:over_thres] <= np.percentile(abs_data[:over_thres], 95)
        )[0][-1]



    else:
        signal_inverted = False  # defaults false

        kernels = {
            "1": np.array([1, -1]),
            "2": np.array([1, 0, -1] + list(np.linspace(-1, 0, 20))),
        }
        ker = kernels[use_method]

        
        # get dot-products between kernel and time-serie snippets
        res = []  # store results of dot-products
        for i in np.arange(0, len(data) - len(ker)):
            res.append(ker @ data[i : i + len(ker)])
            # calculate dot-product of vectors
            # the dot-product result is high when the timeseries snippet
            # is very similar to the kernel
        res = np.array(res)  # convert list to array

        # normalise dot product results
        res = res / max(res)

        # calculate a ratio between std dev and maximum during
        # the first seconds to check whether an stim-artifact was present
        ratio_max_sd = np.max(res[: round(sf_LFP) * 30] / np.std(res[: round(sf_LFP) * 5]))

        # find peak of kernel dot products
        # pos_idx contains all the positive peaks, neg_idx all the negative peaks
        pos_idx = find_peaks(x=res, height=0.3 * max(res), distance=sf_LFP)[0]
        neg_idx = find_peaks(x=-res, height=-0.3 * min(res), distance=sf_LFP)[0]

        # check warn if NO STIM artifacts are suspected
        if (len(neg_idx) > 20 and ratio_max_sd < 8) or (len(pos_idx) > 20 and ratio_max_sd < 8):
            print(
                "WARNING: probably the LFP signal did NOT"
                " contain any artifacts. Many incorrect timings"
                " could be returned"
            )

        # check whether signal is inverted
        if neg_idx[0] < pos_idx[0]:
            # the first peak should be POSITIVE (this is for the dot-product results)
            # actual signal is first peak negative
            # if NEG peak before POS then signal is inverted
            print("intracranial signal is inverted")
            signal_inverted = True
            # re-check inverted for difficult cases with small pos-lfp peak
            # before negative stim-artifact
            if (
                pos_idx[0] - neg_idx[0]
            ) < 50:  # if first positive and negative are very close
                width_pos = 0
                r_i = pos_idx[0]
                while res[r_i] > (max(res) * 0.3):
                    r_i += 1
                    width_pos += 1
                width_neg = 0
                r_i = neg_idx[0]
                while res[r_i] < (min(res) * 0.3):
                    r_i += 1
                    width_neg += 1
                # undo invertion if negative dot-product (pos lfp peak) is very narrow
                if width_pos > (2 * width_neg):
                    signal_inverted = False
                    print("invertion undone")

        # return either POS or NEG peak-indices based on normal or inverted signal
        if not signal_inverted:
            stim_idx = pos_idx  # this is for 'normal' signal

        elif signal_inverted:
            stim_idx = neg_idx

        # filter out inconsistencies in peak heights (assuming sync-stim-artifacts are stable)
        abs_heights = [max(abs(data[i - 5 : i + 5])) for i in stim_idx]

        # check polarity of peak
        if not signal_inverted:
            sel_idx = np.array([min(data[i - 5 : i + 5]) for i in stim_idx]) < (
                np.median(abs_heights) * -0.5
            )
        elif signal_inverted:
            sel_idx = np.array([max(data[i - 5 : i + 5]) for i in stim_idx]) > (
                np.median(abs_heights) * 0.5
            )
        stim_idx_all = list(compress(stim_idx, sel_idx))
        stim_idx = stim_idx_all[0]

    art_time_LFP = stim_idx / sf_LFP

    return art_time_LFP




def detect_artifacts_in_external_recording(
    session_ID: str,
    BIP_channel: np.ndarray,
    sf_external,
    saving_path: str,
    start_index: int = 0,
):
    """
    This function detects the artifacts in the external recording and plots it
    for verification.

    Inputs:
        - session_ID: str, session identifier
        - BIP_channel: np.ndarray, the channel of the external recording to be
        used for synchronization (the one containing deep brain stimulation
        artifacts = the channel recorded with the bipolar electrode)
        - sf_external: sampling frequency of external recording
        - saving_path: str, path to the folder where the figures will be saved
        - start_index: default is 0 when recording is properly started in StimOff,
        but it can be changed when this is not the case (back-up option)


    Output:
        - art_start_BIP: the timestamp when the artifact starts in external recording
    """

    # Generate timescale:
    external_timescale_s = np.arange(
        start=0, stop=(len(BIP_channel) / sf_external), step=(1 / sf_external)
    )

    # apply a highpass filter at 1Hz to the external bipolar channel (detrending)
    filtered_external = _detrend_data(BIP_channel)

    # PLOT 1 :
    # plot the signal of the external channel used for artifact detection:
    plot_channel(
        session_ID=session_ID,
        timescale=external_timescale_s,
        data=filtered_external,
        color="darkcyan",
        ylabel="External bipolar channel - voltage (mV)",
        title="Fig1-External bipolar channel raw plot.png",
        saving_path=saving_path,
        vertical_line=False,
        art_time=None,
        scatter=False
    )
    plt.close()

    ### DETECT ARTIFACTS ###

    # find artifacts in external bipolar channel:
    art_start_BIP = find_external_sync_artifact(
        data=filtered_external, sf_external=sf_external, start_index=start_index
    )

    # PLOT 2 : plot the external channel with the first artifact detected:
    plot_channel(
        session_ID=session_ID,
        timescale=external_timescale_s,
        data=filtered_external,
        color="darkcyan",
        ylabel="Artifact channel BIP (mV)", 
        title="Fig2-External bipolar channel with artifact detected.png", 
        saving_path=saving_path,
        vertical_line=True,
        art_time=art_start_BIP,
        scatter=False,
    )
    plt.show(block=False)

    # PLOT 3 :
    # plot the first artifact detected in external channel (verification of sample choice):
    #idx_start = np.where(external_timescale_s == (art_start_BIP - (60/sf_external)))[0][0]
    #idx_end = np.where(external_timescale_s == (art_start_BIP + (60/sf_external)))[0][0]
    try:
        idx_start = _find_closest_index(external_timescale_s, art_start_BIP - (60/sf_external))
        idx_end = _find_closest_index(external_timescale_s, art_start_BIP + (60/sf_external))
        
        # Ensure there are enough samples
        if idx_start < 0 or idx_end >= len(external_timescale_s):
            raise IndexError("Insufficient samples before or after the specified point.")
            
        print(f"Start index: {idx_start}, End index: {idx_end}")
    except ValueError as e:
        print(e)
    except IndexError as e:
        print(e)


    plot_channel(
        session_ID=session_ID,
        timescale=external_timescale_s[idx_start:idx_end],
        data=filtered_external[idx_start:idx_end],
        color="darkcyan",
        ylabel="Artifact channel BIP - Voltage (mV)",
        title="Fig3-External bipolar channel - first artifact detected.png",
        saving_path=saving_path,
        vertical_line=True,
        art_time=art_start_BIP,
        scatter=True,
    )
    plt.show() #block=False
    

    return art_start_BIP














def detect_artifacts_in_intracranial_recording(
    session_ID: str, lfp_sig: np.ndarray, sf_LFP: int, saving_path: str,  method: str, start_index = 0
):
    """
    This function detects the first artifact in the intracranial recording and plots it.

    Inputs:
        - session_ID: str, session identifier
        - lfp_sig: np.ndarray, the channel of the intracranial recording to be
        used for synchronization (the one containing deep brain stimulation artifacts)
        - sf_LFP: int, sampling frequency of intracranial recording
        - saving_path: str, path to the folder where the figures will be saved
        - method: str, method used for artifact detection in intracranial recording
        (1, 2, thresh, manual)


    Returns:
        - art_start_LFP: the timestamp when the artifact starts in intracranial recording

    """

    # Generate timescale:
    LFP_timescale_s = np.arange(
        start=0, stop=(len(lfp_sig) / sf_LFP), step=(1 / sf_LFP)
    )

    # PLOT 4 :
    # raw signal of the intracranial channel used for artifact detection:
    plot_channel(
        session_ID=session_ID,
        timescale=LFP_timescale_s,
        data=lfp_sig,
        color="darkorange",
        ylabel="Intracerebral LFP channel (µV)",
        title="Fig4-Intracranial channel raw plot.png",
        saving_path=saving_path,
        vertical_line=False,
        art_time=None,
        scatter=False
    )
    plt.close()

    ### DETECT ARTIFACTS ###
    if method in ["1", "2", "thresh"]:
        art_start_LFP = find_LFP_sync_artifact(
            data=lfp_sig, sf_LFP=sf_LFP, use_method=method, start_index=start_index
        )

        # PLOT 5 :
        # plot the intracranial channel with its artifacts detected:
        plot_channel(
            session_ID=session_ID,
            timescale=LFP_timescale_s,
            data=lfp_sig,
            color="darkorange",
            ylabel="Intracranial LFP channel (µV)",
            title="Fig5-Intracranial channel with artifact detected - method " + str(method) + ".png",
            saving_path=saving_path,
            vertical_line=True,
            art_time=art_start_LFP,
            scatter=False,
        )
        #plt.gcf()
        plt.show() #block=False

        # PLOT 6 :
        # plot the first artifact detected in intracranial channel (verification of sample choice):
        #idx_start = round(np.where(LFP_timescale_s == (art_start_LFP))[0][0] - (0.1*sf_LFP))
        #idx_end = round(np.where(LFP_timescale_s == (art_start_LFP))[0][0] + (0.3*sf_LFP))

        try:
            idx_start = _find_closest_index(LFP_timescale_s, art_start_LFP - (25/sf_LFP))
            print(f"idx_start: {idx_start}")
            idx_end = _find_closest_index(LFP_timescale_s, art_start_LFP + (45/sf_LFP))
            print(f"idx_end: {idx_end}")
            # Ensure there are enough samples
            if idx_start < 0 or idx_end >= len(LFP_timescale_s):
                raise IndexError("Insufficient samples before or after the specified point.")
                
            print(f"Start index: {idx_start}, End index: {idx_end}")
        except ValueError as e:
            print(e)
        except IndexError as e:
            print(e)


        plot_channel(
            session_ID=session_ID,
            timescale=LFP_timescale_s[idx_start:idx_end],
            data=lfp_sig[idx_start:idx_end],
            color="darkorange",
            ylabel="Intracranial LFP channel (µV)",
            title="Fig6-Intracranial channel - first artifact detected - method " + str(method) + ".png",
            saving_path=saving_path,
            vertical_line=True,
            art_time=art_start_LFP,
            scatter=True,
        )
        plt.show() #block=False

    if method == "manual":
        print(
            f"Automatic detection of intracranial artifacts failed, using manual method. \n"
            f"In the pop up window, zoom on the first artifact until you can select properly  "
            f"the last sample before the deflection, click on it and close the window."
        )
        art_start_LFP = select_sample(
            signal=lfp_sig, sf=sf_LFP, color1="peachpuff", color2="darkorange"
        )
        #[0:10*round(sf_LFP)]
        # PLOT 7 : plot the artifact adjusted by user in the intracranial channel:
        # idx_start = round(np.where(LFP_timescale_s == (art_start_LFP))[0][0] - (0.1*sf_LFP))
        # idx_end = round(np.where(LFP_timescale_s == (art_start_LFP))[0][0] + (0.3*sf_LFP))

        try:
            idx_start = _find_closest_index(LFP_timescale_s, art_start_LFP - (25/sf_LFP))
            print(f"idx_start: {idx_start}")
            idx_end = _find_closest_index(LFP_timescale_s, art_start_LFP + (45/sf_LFP))
            print(f"idx_end: {idx_end}")
            # Ensure there are enough samples
            if idx_start < 0 or idx_end >= len(LFP_timescale_s):
                raise IndexError("Insufficient samples before or after the specified point.")
                
            print(f"Start index: {idx_start}, End index: {idx_end}")
        except ValueError as e:
            print(e)
        except IndexError as e:
            print(e)
        
        plot_channel(
            session_ID=session_ID,
            timescale=LFP_timescale_s[idx_start:idx_end],
            data=lfp_sig[idx_start:idx_end],
            color="darkorange",
            ylabel="Intracranial LFP channel (µV)",
            title="Fig7-Intracranial channel - first artifact corrected by user.png",  
            saving_path=saving_path,
            vertical_line=True,
            art_time=art_start_LFP,
            scatter=True,
        )
        #plt.gcf()
        #plt.show(block=False)
        plt.show()

    return art_start_LFP

'''