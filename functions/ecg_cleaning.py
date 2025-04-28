    #######################################################################
    #                         ECG CLEANING FUNCTIONS                      #
    #######################################################################


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton,
    QWidget, QDialog, QLabel, QLineEdit, QRadioButton,
    QButtonGroup, QFormLayout, QDialogButtonBox, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import Qt
import numpy as np
import scipy
import matplotlib.pyplot as plt
import mne



class CleaningParameterDialog(QDialog):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cleaning Parameters")
        self.callback = callback  # store the callback function

        layout = QVBoxLayout()

        # Time input fields
        form = QFormLayout()
        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        self.threshold_input = QComboBox()
        self.threshold_input.addItems(['95', '96', '97', '98', '99'])

        self.start_input.setPlaceholderText("0")
        self.end_input.setPlaceholderText("0")

        form.addRow("Start time (s):", self.start_input)
        form.addRow("End time (s):", self.end_input)
        form.addRow("Threshold:", self.threshold_input)

        layout.addLayout(form)

        # Polarity radio buttons
        self.radio_up = QRadioButton("Up")
        self.radio_down = QRadioButton("Down")
        self.radio_up.setChecked(True)

        self.polarity_group = QButtonGroup()
        self.polarity_group.addButton(self.radio_up)
        self.polarity_group.addButton(self.radio_down)

        layout.addWidget(QLabel("Artifact polarity:"))
        layout.addWidget(self.radio_up)
        layout.addWidget(self.radio_down)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_ok)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def on_ok(self):
        # Read values and call the callback
        start = int(self.start_input.text() or 0)
        end = int(self.end_input.text() or 0)
        threshold = int(self.threshold_input.currentText() or 95)
        polarity = "Up" if self.radio_up.isChecked() else "Down"
        self.callback(start, end, threshold, polarity)
        self.close()


    #######################################################################
    #########                 FINDING THE R-PEAKS                 #########
    #######################################################################   



def find_r_peaks_based_on_ext_ecg(self, full_data, times, window_artifact):
    #### FIRST STEP: PREDETERMINE R-PEAKS TIMESTAMPS USING ECG CHANNEL ####
    """ The externally recorded ECG signal was used to predetermine
    the timestamps of the R-peaks. The ECG signal was z-scored ((x-l)/r) 
    over the entire recording and the function findpeaks was used to search 
    for R-peaks with a specific height (at least two and-a-half times 
    the standard deviation of the entire time series) and at a 
    specific inter-peak distance (minimally 500 ms). 
    The algorithm accounts for a negative QRS complex
    by repeating this procedure after multiplying the signal with
    1. For both orientations of the LFP signal, the values of the
    peaks were averaged and the peaks with the highest mean
    determined the orientation of the QRS complexes, provided
    that at least the same amount of peaks were detected.
    """

    data_extra = self.dataset_extra.synced_data.get_data()[self.dataset_extra.selected_channel_index_ecg]

    # Apply 0.1 Hz-100Hz band-pass filter to ECG data
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    detrended_data = scipy.signal.filtfilt(b, a, data_extra)
    low_cutoff = 100.0  # Hz
    b2, a2 = scipy.signal.butter(
        N=4,  # Filter order
        Wn=low_cutoff,
        btype="lowpass",
        fs=self.dataset_extra.sf 
    )
    ecg_data = scipy.signal.filtfilt(b2, a2, detrended_data)
    timescale_extra = np.linspace(0, self.dataset_extra.synced_data.get_data().shape[1]/self.dataset_extra.sf, self.dataset_extra.synced_data.get_data().shape[1])

    # 1. Z-score the ECG signal
    ecg_z = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)

    # 2. Define peak detection params
    threshold = 2.5  # 2.5 * std (signal is already z-scored)
    min_distance_samples = int(0.5 * self.dataset_extra.sf)  # 500 ms in samples

    # 3. Detect peaks in original signal
    peaks_pos, props_pos = scipy.signal.find_peaks(
        ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # 4. Detect peaks in inverted signal
    peaks_neg, props_neg = scipy.signal.find_peaks(
        -ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # 5. Compare polarity by mean peak amplitude
    mean_pos = np.mean(props_pos['peak_heights']) if len(peaks_pos) > 0 else 0
    mean_neg = np.mean(props_neg['peak_heights']) if len(peaks_neg) > 0 else 0

    # 6. Select better polarity (also check similar number of peaks)
    if len(peaks_pos) >= len(peaks_neg):
        chosen_peaks = peaks_pos
    else:
        chosen_peaks = peaks_neg     

    #### Plot the detected peaks ####
    self.canvas_detected_peaks.setEnabled(True)
    self.toolbar_detected_peaks.setEnabled(True)
    self.ax_detected_peaks.clear()
    self.ax_detected_peaks.set_title('Detected Peaks')
    self.ax_detected_peaks.plot(timescale_extra, ecg_data, label='Raw ECG', alpha=0.1)
    self.ax_detected_peaks.plot(timescale_extra[chosen_peaks], ecg_data[chosen_peaks], 'ro', label='Detected Peaks', alpha=0.1)
    self.canvas_detected_peaks.draw()

    """
    For each ECG R-peak, search for LFP peaks (min and max) in ±20 LFP samples.
    Returns:
        selected_peaks: list of peak values (either all max or all min)
        polarity: 'positive' or 'negative' QRS orientation in LFP
    """

    # Convert R-peaks from ECG samples to seconds
    r_peak_times_sec = chosen_peaks / self.dataset_extra.sf

    # Convert times to LFP sample indices
    r_peaks_lfp_idx = np.round(r_peak_times_sec * self.dataset_intra.sf).astype(int)
    print(r_peaks_lfp_idx)

    window = 20  # ±20 LFP samples
    max_peaks = []
    min_peaks = []

    for idx in r_peaks_lfp_idx:
        start = max(idx - window, 0)
        end = min(idx + window + 1, len(full_data))
        segment = full_data[start:end]

        if len(segment) > 0:
            max_peaks.append(np.max(segment))
            min_peaks.append(np.min(segment))

    # Calculate mean absolute values
    mean_abs_max = np.nanmean(np.abs(max_peaks)) if max_peaks else 0
    mean_abs_min = np.nanmean(np.abs(min_peaks)) if min_peaks else 0

    # Choose the orientation with the higher mean absolute amplitude
    lfp_peak_indices = []
    polarity = None
    if mean_abs_max >= mean_abs_min:
        polarity = 'Up'
        for idx in r_peaks_lfp_idx:
            start = max(idx - window, 0)
            end = min(idx + window + 1, len(full_data))
            segment = full_data[start:end]
            local_max_idx = np.argmax(segment)
            peak_global_idx = start + local_max_idx
            lfp_peak_indices.append(peak_global_idx)
    else:
        polarity = 'Down'
        for idx in r_peaks_lfp_idx:
            start = max(idx - window, 0)
            end = min(idx + window + 1, len(full_data))
            segment = full_data[start:end]
            local_min_idx = np.argmin(segment)
            peak_global_idx = start + local_min_idx
            lfp_peak_indices.append(peak_global_idx)

    #### Plot the detected peaks ####
    self.ax_detected_peaks.plot(times, full_data, label='Raw LFP', color='black')
    self.ax_detected_peaks.plot(np.array(times)[lfp_peak_indices], np.array(full_data)[lfp_peak_indices], 'ro', label='LFP Peaks')
    self.ax_detected_peaks.legend()
    self.canvas_detected_peaks.draw()

    # Estimate HR
    peak_intervals = np.diff(lfp_peak_indices) / self.dataset_intra.sf  # Convert to seconds
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    #ecg['stats']['hr'] = hr
    #ecg['stats']['pctartefact'] = (1 - len(final_peaks) / ns) * 100
    self.label_heart_rate_lfp.setText(f'Heart rate: {hr} bpm')

    # Define epoch window
    sf_lfp = self.dataset_intra.sf
    pre_samples = int(abs(window_artifact[0]) * sf_lfp)
    post_samples = int(window_artifact[1] * sf_lfp)
    epoch_length = pre_samples + post_samples  # Total length of each epoch
    time = np.linspace(window_artifact[0], window_artifact[1], epoch_length)  # Time in seconds

    # avoid beginning and end of the recording to compute the average QRS template (because of the sync pulses):
    # remove first and last minute just to be sure:
    after_pulse = 60 * self.dataset_intra.sf
    before_last_pulse = (times[-1] - 60) * self.dataset_intra.sf

    epochs = []  # Store extracted heartbeats

    for peak in lfp_peak_indices:
        start = peak - pre_samples
        end = peak + post_samples

        #if start >= 0 and end < len(full_data):  # Ensure we don't go out of bounds
        if start >= after_pulse and end < before_last_pulse:
            epochs.append(full_data[start:end])
    
        #if start >= 0 and end < len(full_data):  # Ensure we don't go out of bounds
        #    epochs.append(full_data[start:end])

    epochs = np.array(epochs)

    # Compute average heartbeat template
    mean_epoch = np.nanmean(epochs, axis=0)

    # Plot the detected ECG epochs
    self.canvas_ecg_artifact.setEnabled(True)
    self.toolbar_ecg_artifact.setEnabled(True)
    self.ax_ecg_artifact.clear()
    self.ax_ecg_artifact.set_title("Detected ECG epochs")

    for epoch in epochs:
        self.ax_ecg_artifact.plot(time, epoch, color='gray', alpha=0.3)

    self.ax_ecg_artifact.plot(time, mean_epoch, color='black', linewidth=2, label='Average ECG Template')
    self.ax_ecg_artifact.set_xlabel("Time (s)")
    self.ax_ecg_artifact.set_ylabel("Amplitude")
    self.ax_ecg_artifact.legend()
    self.canvas_ecg_artifact.draw()


    return lfp_peak_indices, polarity, mean_epoch




def find_r_peaks_in_lfp_channel(
        self, 
        data_crop, 
        polarity, 
        detection_threshold, 
        window = [-0.5, 0.5]
        ):
    #### FIRST STEP: DETERMINE R-PEAKS TIMESTAMPS USING ONLY LFP CHANNEL ####    
    if polarity == "Down":
        REVERSED = True # true if cardiac artifacts are going downwards
        cropped_data = - data_crop
    else:
        REVERSED = False
        cropped_data = data_crop

    ecg = {'proc': {}, 'stats': {}, 'cleandata': None, 'detected': False}
    ns = len(cropped_data)
    
    # Segment the signal into overlapping windows
    sf_lfp = round(self.dataset_intra.sf)
    dwindow = int(round(sf_lfp))  # 1s window
    dmove = sf_lfp  # 1s step
    n_segments = (ns - dwindow) // dmove + 1
    
    x = np.array([cropped_data[i * dmove: i * dmove + dwindow] for i in range(n_segments) if i * dmove + dwindow <= ns])
    
    detected_peaks = []  # Store peak indices in the original timescale
    
    # Loop through each segment and find peaks
    for i in range(n_segments):
        segment = x[i]
        # 6. Skip segment if it contains any NaNs
        if np.isnan(segment).any():
            continue
        peaks, _ = scipy.signal.find_peaks(segment, height=np.percentile(segment, 90), distance=sf_lfp//2)  # Adjust threshold & min distance
        real_peaks = peaks + (i * dmove)  # Convert to original timescale
        detected_peaks.extend(real_peaks)

    detected_peaks = np.array(detected_peaks)
    print(detected_peaks)
    
    # Define epoch window (-0.5s to +0.5s)
    pre_samples = int(abs(window[0]) * sf_lfp)
    post_samples = int(window[1] * sf_lfp)
    epoch_length = pre_samples + post_samples  # Total length of each epoch
    time = np.linspace(window[0], window[1], epoch_length)  # Time in seconds

    epochs = []  # Store extracted heartbeats

    for peak in detected_peaks:
        start = peak - pre_samples
        end = peak + post_samples
        
        if start >= 0 and end < ns:  # Ensure we don't go out of bounds
            epoch = cropped_data[start:end]
            if np.isnan(epoch).any():
                continue
            else:
                epochs.append(epoch)

    epochs = np.array(epochs)
    print(len(epochs))

    # Compute average heartbeat template
    mean_epoch = np.nanmean(epochs, axis=0)
    print(mean_epoch)
    ecg['proc']['template1'] = mean_epoch  # First ECG template

    # Plot the detected ECG epochs
    self.canvas_ecg_artifact.setEnabled(True)
    self.toolbar_ecg_artifact.setEnabled(True)
    self.ax_ecg_artifact.clear()
    self.ax_ecg_artifact.set_title("Detected ECG epochs")

    for epoch in epochs:
        if REVERSED:
            epoch = - epoch
        self.ax_ecg_artifact.plot(time, epoch, color='gray', alpha=0.3)
    
    if REVERSED:
            self.ax_ecg_artifact.plot(time, - mean_epoch, color='black', linewidth=2, label='Average ECG Template')
    else:
        self.ax_ecg_artifact.plot(time, mean_epoch, color='black', linewidth=2, label='Average ECG Template')
    
    self.ax_ecg_artifact.set_xlabel("Time (s)")
    self.ax_ecg_artifact.set_ylabel("Amplitude")
    self.ax_ecg_artifact.legend()
    self.canvas_ecg_artifact.draw()

    # Temporal correlation for ECG detection
    # adapt in case NaNs are present:
    if np.isnan(cropped_data).any():
        cropped_data_clean = np.nan_to_num(cropped_data, nan=0.0)
        r = np.correlate(cropped_data_clean, mean_epoch, mode='same')
    else:
        r = np.correlate(cropped_data, mean_epoch, mode='same')
    threshold = np.percentile(r, 95)
    detected_peaks, _ = scipy.signal.find_peaks(r, height=threshold, distance=sf_lfp//2)


    # Second pass for refining detection
    refined_template = np.nanmean([cropped_data[p - dwindow//2 : p + dwindow//2] for p in detected_peaks if p - dwindow//2 > 0 and p + dwindow//2 < ns], axis=0)
    print(f'refined template: {refined_template}')
    ecg['proc']['template2'] = refined_template
    if np.isnan(cropped_data).any():
        cropped_data_clean = np.nan_to_num(cropped_data, nan=0.0)
        r2 = np.correlate(cropped_data_clean, refined_template, mode='same')  
    else:  
        r2 = np.correlate(cropped_data, refined_template, mode='same')
    threshold2 = np.percentile(r2, detection_threshold)
    final_peaks, _ = scipy.signal.find_peaks(r2, height=threshold2, distance=sf_lfp//2)


    # plot the detected peaks
    self.canvas_detected_peaks.setEnabled(True)
    self.toolbar_detected_peaks.setEnabled(True)
    self.ax_detected_peaks.clear()
    self.ax_detected_peaks.set_title('Detected Peaks')

    if REVERSED:
        self.ax_detected_peaks.plot(- cropped_data, label='Raw ECG')
        self.ax_detected_peaks.plot(detected_peaks, - cropped_data[detected_peaks], 'ro', label='Detected Peaks')
        self.canvas_detected_peaks.draw()
    else:
        self.ax_detected_peaks.plot(cropped_data, label='Raw ECG')
        self.ax_detected_peaks.plot(detected_peaks, cropped_data[detected_peaks], 'ro', label='Detected Peaks')
        self.canvas_detected_peaks.draw()            


    # Estimate HR
    peak_intervals = np.diff(final_peaks) / sf_lfp  # Convert to seconds
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    #ecg['stats']['hr'] = hr
    #ecg['stats']['pctartefact'] = (1 - len(final_peaks) / ns) * 100
    self.label_heart_rate_lfp.setText(f'Heart rate: {hr} bpm')

    if REVERSED:
        mean_epoch = - mean_epoch

    return final_peaks, mean_epoch



    #######################################################################
    #########                  INTERPOLATION METHOD               #########
    #######################################################################        

def start_ecg_cleaning_interpolation(self):
    """Start the ECG cleaning process using the interpolation method from Perceive toolbox."""
    if self.dataset_intra.synced_data is not None and self.dataset_intra.selected_channel_index_ecg is not None:
        try:
            clean_ecg_interpolation(self)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")



def clean_ecg_interpolation(self):
    full_data = self.dataset_intra.synced_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
    times = np.linspace(0, self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.synced_data.get_data().shape[1])
    window = [-0.5, 0.5]

    if self.dataset_extra.selected_channel_name_ecg is not None:
        final_peaks, polarity, mean_epoch = find_r_peaks_based_on_ext_ecg(self, full_data, times, window)
        clean_data = np.copy(full_data)
        ns = len(full_data)

        
    ############################################################################
    elif self.dataset_extra.selected_channel_name_ecg is None:
        # Add a flag to control after-dialog processing
        self.dialog_completed = False

        def after_dialog(start, end, threshold, polarity, full_data, times):
            print("Received from dialog:")
            print(f"Start: {start}, End: {end}, Polarity: {polarity}, Threshold: {threshold}")

            sf_lfp = round(self.dataset_intra.sf)

            # Crop data to remove sync pulses (amplitude is too big)
            data_crop = full_data[int(start * sf_lfp):int(end * sf_lfp)]
            beginning_part = full_data[:int(start * sf_lfp)]
            end_part = full_data[int(end * sf_lfp):]

            final_peaks, mean_epoch = find_r_peaks_in_lfp_channel(self, data_crop, polarity, threshold, window)
            clean_data = np.copy(data_crop)
            ns = len(data_crop)

            #print("Done cleaning.")

            self.beginning_part = beginning_part
            self.end_part = end_part
            self.final_peaks = final_peaks
            self.clean_data = clean_data
            self.ns = ns
            self.mean_epoch = mean_epoch

            # Set the flag to True after the dialog completes
            self.dialog_completed = True

        # Use lambda to pass full_data and times into callback
        self.dialog = CleaningParameterDialog(
            callback=lambda start, end, threshold, polarity: after_dialog(
                start, end, threshold, polarity, full_data, times
            )
        )
        self.dialog.show()  # Non-modal, allows interaction with main window
        #print('showing dialog')

        # Wait for dialog to finish before proceeding
        while not self.dialog_completed:
            QApplication.processEvents()  # Process the UI events (keeps UI responsive)
        

    if self.dataset_extra.selected_channel_name_ecg is None:
        beginning_part = self.beginning_part
        end_part = self.end_part
        final_peaks = self.final_peaks
        clean_data = self.clean_data
        ns = self.ns
        mean_epoch = self.mean_epoch


    #### SECOND STEP: INTERPOLATE DATA AT EACH R-PEAK FOUND ####
    # Remove artifacts (simple interpolation)
    for p in final_peaks:
        clean_data[max(0, p - 5): min(ns, p + 5)] = np.nan  # NaN out artifacts
    clean_data = np.interp(np.arange(ns), np.arange(ns)[~np.isnan(clean_data)], clean_data[~np.isnan(clean_data)])

    if self.dataset_extra.selected_channel_name_ecg is not None:
        clean_data_full = clean_data
    else:
        clean_data_full = np.concatenate([beginning_part, clean_data, end_part])


    if self.dataset_intra.selected_channel_index_ecg == 0:
        self.dataset_intra.cleaned_ecg_left = clean_data_full
        print("Left channel cleaned")

    elif self.dataset_intra.selected_channel_index_ecg == 1:
        self.dataset_intra.cleaned_ecg_right = clean_data_full
        print("Right channel cleaned")

    # plot an overlap of the raw and cleaned data
    self.canvas_ecg_clean.setEnabled(True)
    self.toolbar_ecg_clean.setEnabled(True)
    self.ax_ecg_clean.clear()
    self.ax_ecg_clean.set_title("Cleaned ECG Signal")
    self.ax_ecg_clean.plot(times,full_data, label='Raw data')
    self.ax_ecg_clean.plot(times,clean_data_full, label='Cleaned data')
    self.ax_ecg_clean.set_xlabel("Time (s)")
    self.ax_ecg_clean.set_ylabel("Amplitude")
    self.ax_ecg_clean.legend()
    self.canvas_ecg_clean.draw()

    # Plot an overlap of the power spectrum using welch's method:
    n_fft = int(round(self.dataset_intra.sf))
    n_overlap=int(round(self.dataset_intra.sf)/2)

    psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(
        full_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    
    psd_clean, freqs_clean = mne.time_frequency.psd_array_welch(
        clean_data_full,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)

    self.canvas_psd.setEnabled(True)
    self.toolbar_psd.setEnabled(True)
    self.ax_psd.clear()
    self.ax_psd.plot(freqs_raw, np.log(psd_raw), color='blue', label='PSD raw channel')
    self.ax_psd.plot(freqs_clean, np.log(psd_clean), color = 'orange', label='PSD cleaned channel')
    self.ax_psd.legend()
    self.canvas_psd.draw()

    self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning



    #######################################################################
    #########              TEMPLATE SUBSTRACTION METHOD           #########
    #######################################################################      


def start_ecg_cleaning_template_sub(self):
    """Start the ECG cleaning process using the template substraction method."""
    if self.dataset_intra.synced_data is not None and self.dataset_intra.selected_channel_index_ecg is not None:
        # Perform the ECG cleaning process here
        try:
            clean_ecg_template_sub(self)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")

def clean_ecg_template_sub(self):
    full_data = self.dataset_intra.synced_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
    times = np.linspace(0, self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.synced_data.get_data().shape[1])
    window = [-0.2, 0.2]
    
    if self.dataset_extra.selected_channel_name_ecg is not None:
        final_peaks, polarity, mean_epoch = find_r_peaks_based_on_ext_ecg(self, full_data, times, window) 
        clean_data = np.copy(full_data)
        #ns = len(full_data)

        """
        # Create a QRS template #
        # Define epoch window (-0.2s to +0.2s): only keep QRS complex (based on Stam et al., 2023)
        pre_samples = int(abs(window[0]) * self.dataset_intra.sf)
        post_samples = int(window[1] * self.dataset_intra.sf)
        epoch_length = pre_samples + post_samples  # Total length of each epoch
        time = np.linspace(window[0], window[1], epoch_length)  # Time in seconds

        epochs = []  # Store extracted heartbeats        

        # avoid beginning and end of the recording to compute the average QRS template (because of the sync pulses):
        # remove first and last minute just to be sure:
        after_pulse = 60 * self.dataset_intra.sf
        before_last_pulse = (times[-1] - 60) * self.dataset_intra.sf

        for peak in final_peaks:
            start = peak - pre_samples
            end = peak + post_samples
            
            #if start >= 0 and end < len(full_data):  # Ensure we don't go out of bounds
            if start >= after_pulse and end < before_last_pulse:
                epochs.append(full_data[start:end])

        epochs = np.array(epochs)

        # Compute average QRS template
        mean_epoch = np.nanmean(epochs, axis=0)
        self.canvas_ecg_artifact.setEnabled(True)
        self.toolbar_ecg_artifact.setEnabled(True)
        self.ax_ecg_artifact.clear()
        self.ax_ecg_artifact.set_title("Detected QRS epochs")

        for epoch in epochs:
            self.ax_ecg_artifact.plot(time, epoch, color='gray', alpha=0.3)
        
        self.ax_ecg_artifact.plot(time, mean_epoch, color='black', linewidth=2, label='Average QRS Template')
        self.ax_ecg_artifact.set_xlabel("Time (s)")
        self.ax_ecg_artifact.set_ylabel("Amplitude")
        self.ax_ecg_artifact.legend()
        self.canvas_ecg_artifact.draw()
        """
    ############################################################################
    elif self.dataset_extra.selected_channel_name_ecg is None:

        # Add a flag to control after-dialog processing
        self.dialog_completed = False

        def after_dialog(start, end, threshold, polarity, full_data, times):
            print("Received from dialog:")
            print(f"Start: {start}, End: {end}, Polarity: {polarity}, Threshold: {threshold}")

            sf_lfp = round(self.dataset_intra.sf)

            # Crop data to remove sync pulses (amplitude is too big)
            data_crop = full_data[int(start * sf_lfp):int(end * sf_lfp)]
            beginning_part = full_data[:int(start * sf_lfp)]
            end_part = full_data[int(end * sf_lfp):]

            final_peaks, mean_epoch = find_r_peaks_in_lfp_channel(self, data_crop, polarity, threshold, window)
            clean_data = np.copy(data_crop)
            ns = len(data_crop)

            # Now you can use or return them, or update UI/state/etc.
            print("Done cleaning.")

            self.beginning_part = beginning_part
            self.end_part = end_part
            self.final_peaks = final_peaks
            self.clean_data = clean_data
            self.ns = ns
            self.mean_epoch = mean_epoch
            self.polarity_sig = polarity

            # Set the flag to True after the dialog completes
            self.dialog_completed = True

        # Use lambda to pass full_data and times into callback
        self.dialog = CleaningParameterDialog(
            callback=lambda start, end, threshold, polarity: after_dialog(
                start, end, threshold, polarity, full_data, times
            )
        )
        self.dialog.show()  # Non-modal, allows interaction with main window
        print('showing dialog')

        # Wait for dialog to finish before proceeding
        while not self.dialog_completed:
            QApplication.processEvents()  # Process the UI events (keeps UI responsive)
        

    if self.dataset_extra.selected_channel_name_ecg is None:
        beginning_part = self.beginning_part
        end_part = self.end_part
        final_peaks = self.final_peaks
        clean_data = self.clean_data
        ns = self.ns
        mean_epoch = self.mean_epoch
        polarity = self.polarity_sig
    
    
    #### SECOND STEP: SUBSTRACT TEMPLATE DATA AT EACH R-PEAK FOUND ####
    ####################################################################
    # create a short QRS template to be substracted from the signal:
    # Assuming the R-peak is at the center of the mean_epoch
    center_idx = len(mean_epoch) // 2  
    # Detect Q-peak (local minimum before R)
    Q_range = mean_epoch[center_idx-25:center_idx]  # Left side of the R-peak (100ms before)

    if polarity == 'Down': 
        reversed_idx = np.nanargmax(Q_range[::-1])
        Q_idx = center_idx - 1 - reversed_idx
    elif polarity == 'Up':
        reversed_idx = np.nanargmin(Q_range[::-1])   # Q-peak index (minimum value before R)
        Q_idx = center_idx - 1 - reversed_idx  

    # Detect S-peak (local minimum after R)
    S_range = mean_epoch[center_idx: center_idx+25]  # Right side of the R-peak (100ms after)

    if polarity == 'Down':
        S_idx = np.nanargmax(S_range) + center_idx
    elif polarity == 'Up':
        S_idx = np.nanargmin(S_range) + center_idx  # Adjust index relative to full epoch
    
    max_offset = 30  # Maximum samples to check for minimal difference

    # Define the left tail: from beginning of epoch up# to Q-peak (max 30 samples)
    left_tail_end = min(Q_idx, max_offset)
    Q_window = mean_epoch[:left_tail_end]

    # Define the right tail: from S-peak to end of epoch (max 30 samples)
    right_tail_start = max(S_idx + 1, len(mean_epoch) - max_offset)
    S_window = mean_epoch[right_tail_start:]

    # Find pair (q, s) with smallest absolute difference
    min_diff = float("inf")
    start_idx, end_idx = None, None

    for i, q_val in enumerate(Q_window):
        for j, s_val in enumerate(S_window):
            diff = abs(q_val - s_val)
            if diff < min_diff:
                min_diff = diff
                start_idx = i
                end_idx = j + right_tail_start  # Adjust index relative to full epoch


    if mean_epoch[start_idx] != mean_epoch[end_idx]:
        higher_value = max(mean_epoch[start_idx], mean_epoch[end_idx])
        mean_epoch[start_idx] = mean_epoch[end_idx] = higher_value

    complex_qrs_template = mean_epoch[start_idx:end_idx]  # Extract the QRS complex template  

    # Define original epoch length
    original_length = len(mean_epoch)

    # Compute missing samples on both sides
    missing_left = start_idx  # Samples removed before start_idx
    missing_right = original_length - end_idx  # Samples removed after end_idx

    # Get common value to extend (the value at start_idx and end_idx are equal)
    common_value = mean_epoch[start_idx]

    # Extend the refined template with straight tails
    extended_template = np.concatenate([
        np.full(missing_left, common_value),  # Left tail
        mean_epoch[start_idx:end_idx],                     # Main refined template
        np.full(missing_right, common_value)   # Right tail
    ])


    # Ensure the length is correct
    assert len(extended_template) == original_length, "Length mismatch!"

    time = np.linspace(window[0], window[1], original_length)  # Time in seconds
    # overlap the average QRS template with equal tails onto the original QRS average:
    self.ax_ecg_artifact.plot(time, extended_template, color='purple', linewidth=2, label='Average with equal tails')
    self.ax_ecg_artifact.legend()
    self.canvas_ecg_artifact.draw()

    template = complex_qrs_template
    template_len = len(template)

    # 1. Find R-peak index in template (highest value for peaks going up, lower value for peaks going down)
    if polarity == 'Down':
        template_r_idx = np.argmin(template)
    elif polarity == 'Up':
        template_r_idx = np.argmax(template)

    # 2. Prepare design matrix for linear fit (scale + offset)
    X_template = np.vstack([template, np.ones_like(template)]).T  # Shape: (template_len, 2)

    for peak_idx in final_peaks:
        # 3. Align R-peak in signal with R-peak in template
        start = peak_idx - template_r_idx
        end = start + template_len

        # 4. Check signal boundaries
        if start < 0 or end > len(clean_data):
            continue

        # 5. Extract corresponding signal segment
        segment = clean_data[start:end]  

        # 6. Skip segment if it contains any NaNs
        if np.isnan(segment).any():
            continue

        # 7. Solve for optimal scale (a) and offset (b) using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_template, segment, rcond=None)
        a, b = coeffs

        # 8. Build fitted template and subtract
        fitted_template = a * template + b

        clean_data[start:end] -= fitted_template

    if self.dataset_extra.selected_channel_name_ecg is not None:
        clean_data_full = clean_data
    else:
        clean_data_full = np.concatenate([beginning_part, clean_data, end_part])


    if self.dataset_intra.selected_channel_index_ecg == 0:
        self.dataset_intra.cleaned_ecg_left = clean_data_full
        print("Left channel cleaned")

    elif self.dataset_intra.selected_channel_index_ecg == 1:
        self.dataset_intra.cleaned_ecg_right = clean_data_full
        print("Right channel cleaned")

    # plot an overlap of the raw and cleaned data
    self.canvas_ecg_clean.setEnabled(True)
    self.toolbar_ecg_clean.setEnabled(True)
    self.ax_ecg_clean.clear()
    self.ax_ecg_clean.set_title("Cleaned ECG Signal")
    self.ax_ecg_clean.plot(times, full_data, label='Raw data')
    self.ax_ecg_clean.plot(times, clean_data_full, label='Cleaned data')
    self.ax_ecg_clean.set_xlabel("Time (s)")
    self.ax_ecg_clean.set_ylabel("Amplitude")
    self.ax_ecg_clean.legend()
    self.canvas_ecg_clean.draw()

    # Plot an overlap of the power spectrum using welch's method:
    n_fft = int(round(self.dataset_intra.sf))
    n_overlap=int(round(self.dataset_intra.sf)/2)

    psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(
        full_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    
    psd_clean, freqs_clean = mne.time_frequency.psd_array_welch(
        clean_data_full,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)

    self.canvas_psd.setEnabled(True)
    self.toolbar_psd.setEnabled(True)
    self.ax_psd.clear()
    self.ax_psd.plot(freqs_raw, np.log(psd_raw), color='blue', label='PSD raw channel')
    self.ax_psd.plot(freqs_clean, np.log(psd_clean), color = 'orange', label='PSD cleaned channel')
    self.ax_psd.legend()
    self.canvas_psd.draw()

    self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning

      


    #######################################################################
    #########          SINGULAR VALUE DECOMPOSITION METHOD        #########
    #######################################################################      

def start_ecg_cleaning_svd(self):
    """Start the ECG cleaning process using Singular Value Decomposition method."""
    if self.dataset_intra.raw_data is not None and self.dataset_intra.selected_channel_index_ecg is not None:
        # Perform the ECG cleaning process here
        try:
            clean_ecg_svd(self)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")


def clean_ecg_svd(self):
    data_extra = self.dataset_extra.synced_data.get_data()[self.dataset_extra.selected_channel_index_ecg]
    #data_extra_scaled = data_extra * y_max_factor
    data_extra_scaled = data_extra

    # Apply 0.1 Hz-100Hz band-pass filter to ECG data
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    detrended_data = scipy.signal.filtfilt(b, a, data_extra_scaled)
    low_cutoff = 100.0  # Hz
    b2, a2 = scipy.signal.butter(
        N=4,  # Filter order
        Wn=low_cutoff,
        btype="lowpass",
        fs=self.dataset_extra.sf 
    )
    ecg_data = scipy.signal.filtfilt(b2, a2, detrended_data)
    timescale_extra = np.linspace(0, self.dataset_extra.synced_data.get_data().shape[1]/self.dataset_extra.sf, self.dataset_extra.synced_data.get_data().shape[1])

    # 1. Z-score the ECG signal
    ecg_z = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)

    # 2. Define peak detection params
    threshold = 2.5  # 2.5 * std (signal is already z-scored)
    min_distance_samples = int(0.5 * self.dataset_extra.sf)  # 500 ms in samples

    # 3. Detect peaks in original signal
    peaks_pos, props_pos = scipy.signal.find_peaks(
        ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # 4. Detect peaks in inverted signal
    peaks_neg, props_neg = scipy.signal.find_peaks(
        -ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # 5. Compare polarity by mean peak amplitude
    mean_pos = np.mean(props_pos['peak_heights']) if len(peaks_pos) > 0 else 0
    mean_neg = np.mean(props_neg['peak_heights']) if len(peaks_neg) > 0 else 0

    # 6. Select better polarity (also check similar number of peaks)
    #if len(peaks_pos) >= len(peaks_neg):
    if mean_pos >= mean_neg:
        chosen_peaks = peaks_pos
        orientation = 'positive'
    else:
        chosen_peaks = peaks_neg
        orientation = 'negative'        
    print(f"peaks orientation in the ecg channel: {orientation}")
    #### Plot the detected peaks ####
    self.canvas_detected_peaks_with_ext.setEnabled(True)
    self.toolbar_detected_peaks_with_ext.setEnabled(True)
    self.ax_detected_peaks_with_ext.clear()
    self.ax_detected_peaks_with_ext.set_title('Detected Peaks')
    self.ax_detected_peaks_with_ext.plot(timescale_extra, ecg_data, label='Raw ECG', alpha=0.1)
    self.ax_detected_peaks_with_ext.plot(timescale_extra[chosen_peaks], ecg_data[chosen_peaks], 'ro', label='Detected Peaks', alpha=0.1)
    self.canvas_detected_peaks_with_ext.draw()

    #### SECOND STEP: FIND CORRESPONDING R-PEAKS IN THE LFP SIGNAL ####
    """ Subsequently, the LFP signal was searched for peaks using
    numpy functions min and max in a window of 20 samples
    prior and after the R-peaks found in the ECG signal. This
    window was adopted in order to account for inaccuracies
    in the synchronization of the LFP- and ECG signals. The absolute 
    values of the peaks found with min and max were averaged and the peaks
    with the highest absolute mean determined the orientation of the QRS 
    complexes.
    """

    full_data = self.dataset_intra.synced_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
    times = np.linspace(0, self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.synced_data.get_data().shape[1])
    
    """
    For each ECG R-peak, search for LFP peaks (min and max) in ±20 LFP samples.
    """
    # Convert R-peaks from ECG samples to seconds
    r_peak_times_sec = chosen_peaks / self.dataset_extra.sf

    # Convert times to LFP sample indices
    r_peaks_lfp_idx = np.round(r_peak_times_sec * self.dataset_intra.sf).astype(int)
    print(len(r_peaks_lfp_idx)) # sub017 DBS ON Left = 1685

    window = 20  # ±20 LFP samples
    max_peaks = []
    min_peaks = []

    for idx in r_peaks_lfp_idx:
        start = max(idx - window, 0)
        end = min(idx + window + 1, len(full_data) -1 )
        segment = full_data[start:end]

        #if len(segment) > 0:  # will need to be refined for robustness to NaN values
        if segment.size:
            max_peaks.append(np.nanmax(segment))
            min_peaks.append(np.nanmin(segment))

    # Calculate mean absolute values
    mean_abs_max = np.nanmean(np.abs(max_peaks)) if max_peaks else 0
    print(mean_abs_max)
    mean_abs_min = np.nanmean(np.abs(min_peaks)) if min_peaks else 0
    print(mean_abs_min)

    # Choose the orientation with the higher mean absolute amplitude
    lfp_peak_indices = []
    polarity = None
    if mean_abs_max >= mean_abs_min:
        polarity = 'positive'
        print(f"peaks polarity in the LFP channel: {polarity}")
        for idx in r_peaks_lfp_idx:
            start = max(idx - window, 0)
            end = min(idx + window + 1, len(full_data)-1)
            segment = full_data[start:end]
            #if len(segment) > 0:
            if segment.size:
                local_max_idx = np.nanargmax(segment) 
                peak_global_idx = start + local_max_idx
                lfp_peak_indices.append(peak_global_idx)
    else:
        polarity = 'negative'
        print(f"peaks polarity in the LFP channel: {polarity}")
        for idx in r_peaks_lfp_idx:
            start = max(idx - window, 0)
            end = min(idx + window + 1, len(full_data)-1)
            segment = full_data[start:end]
            #if len(segment) > 0:
            if segment.size:
                local_min_idx = np.nanargmin(segment) 
                peak_global_idx = start + local_min_idx
                lfp_peak_indices.append(peak_global_idx)

    self.ax_detected_peaks_with_ext.plot(times, full_data, label='Raw LFP', color='black')
    self.ax_detected_peaks_with_ext.plot(np.array(times)[lfp_peak_indices], np.array(full_data)[lfp_peak_indices], 'ro', label='LFP Peaks')
    self.ax_detected_peaks_with_ext.legend()
    self.canvas_detected_peaks_with_ext.draw()        

    # Estimate HR
    peak_intervals = np.diff(lfp_peak_indices) / self.dataset_intra.sf  # Convert to seconds
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    #ecg['stats']['hr'] = hr
    #ecg['stats']['pctartefact'] = (1 - len(final_peaks) / ns) * 100
    self.label_heart_rate_lfp_with_ext.setText(f'Heart rate: {hr} bpm')

    # Create a QRS template #
    # Define epoch window (-0.2s to +0.2s): only keep QRS complex (based on Stam et al., 2023)
    pre_samples = int(0.2 * self.dataset_intra.sf)
    post_samples = int(0.2 * self.dataset_intra.sf)
    epoch_length = pre_samples + post_samples  # Total length of each epoch

    epochs = []  # Store extracted heartbeats

    # avoid beginning and end of the recording to compute the average QRS template (because of the sync pulses):
    # remove first and last minute just to be sure:
    after_pulse = 60 * self.dataset_intra.sf
    before_last_pulse = (times[-1] - 60) * self.dataset_intra.sf

    lfp_peak_indices_filtered = []

    for peak in lfp_peak_indices:
        start = peak - pre_samples
        end = peak + post_samples
        
        #if start >= 0 and end < len(full_data):  # Ensure we don't go out of bounds
        if start >= after_pulse and end < before_last_pulse:
            epochs.append(full_data[start:end])
            lfp_peak_indices_filtered.append(peak)

    epochs = np.array(epochs)    # shape: (n timepoints, n epochs)

    ######### SINGULAR VALUE DECOMPOSITION ################
    X = epochs.T                        # shape: (n epochs, n timepoints)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # only use SVD1 (therefore index 0): CAN BE MODIFIED LATER
    reconstructed = np.outer(U[:, 0], S[0] * Vh[0, :]).T   # shape (n epochs, n timepoints)

    # using different SVD: example with SVD2:
    #SVD_components = {}
    #SVD_level = 2
    #comps = [1,2,3,4]
    #for i in range(len(comps)):
    #    SVD = U[:, :comps[i]] @ np.diag(S[:comps[i]]) @ Vh[:comps[i], :]
    #    SVD_components[i] = SVD

    #reconstructed = SVD_components[1].T

    all_complex_svd_templates = []
    all_complex_offset_svd_templates = []

    max_sample = 30 # tail with 30 samples max
    self.ax_ecg_artifact_with_ext.clear()
    for r in reconstructed:
        left_tail = r[0:max_sample]
        right_tail_start = (len(r)-1) - max_sample
        right_tail = r[right_tail_start:]
        min_diff = float("inf")
        start_idx, end_idx = None, None

        for i, q_val in enumerate(left_tail):
            for j, s_val in enumerate(right_tail):
                diff = abs(q_val - s_val)
                if diff < min_diff:
                    min_diff = diff
                    start_idx = i
                    end_idx = j + right_tail_start
        
        if r[start_idx] != r[end_idx]:
            higher_value = max(r[start_idx], r[end_idx])
            r[start_idx] = r[end_idx] = higher_value

        svd_template = r[start_idx:end_idx]
        all_complex_svd_templates.append(svd_template)
        offset_svd_template = svd_template - r[start_idx]
        all_complex_offset_svd_templates.append(offset_svd_template)
        timescale_template = np.linspace(0, len(all_complex_svd_templates[0])/self.dataset_intra.sf, len(all_complex_svd_templates[0]))
        self.ax_ecg_artifact_with_ext.plot(timescale_template, offset_svd_template, color='lightgrey')
    
    mean_svd_template = np.mean(np.array(all_complex_offset_svd_templates), axis=0)
    self.ax_ecg_artifact_with_ext.plot(timescale_template, mean_svd_template, color = 'black', label = 'average SVD across epochs')
    self.ax_ecg_artifact_with_ext.set_xlabel("Time (s)")
    self.ax_ecg_artifact_with_ext.set_ylabel("Amplitude")
    self.ax_ecg_artifact_with_ext.legend()
    self.canvas_ecg_artifact_with_ext.draw()


    clean_data = np.copy(full_data)

    for i, peak_idx in enumerate(lfp_peak_indices_filtered):
        # load the template corresponding to that specific peak:
        template = all_complex_svd_templates[i]
        svd_template_len = len(template)
        
        # 3. Align R-peak in signal with R-peak in template
        svd_template_r_idx = np.argmax(template) if polarity == 'positive' else np.argmin(template)
        start = peak_idx - svd_template_r_idx
        end = start + svd_template_len

        # 2. Prepare design matrix for linear fit (scale + offset)
        X_template = np.vstack([template, np.ones_like(template)]).T  # Shape: (template_len, 2)

        # 4. Check signal boundaries
        if start < 0 or end > len(full_data):
            continue

        # 5. Extract corresponding signal segment
        segment = full_data[start:end]

        # 6. Solve for optimal scale (a) and offset (b) using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_template, segment, rcond=None)
        a, b = coeffs

        # 7. Build fitted template and subtract
        #fitted_template = a * template + b
        fitted_template = template + b # ONLY CHANGES THE OFFSET, LEAVES THE SCALING INTACT !!
        if i == 1:
            plt.plot(segment, label = 'original segment')
            plt.plot(fitted_template, label = 'fitted_template')
        
        clean_data[start:end] -= fitted_template

    if self.dataset_intra.selected_channel_index_ecg == 0:
        self.dataset_intra.cleaned_ecg_left = clean_data
        self.dataset_intra.detected_peaks_left = lfp_peak_indices
        #self.dataset_intra.epochs_left = epochs
        print("Left channel cleaned")

    elif self.dataset_intra.selected_channel_index_ecg == 1:
        self.dataset_intra.cleaned_ecg_right = clean_data
        self.dataset_intra.detected_peaks_right = lfp_peak_indices
        #self.dataset_intra.epochs_right = epochs
        print("Right channel cleaned")

    
    # plot an overlap of the raw and cleaned data
    self.canvas_ecg_clean_with_ext.setEnabled(True)
    self.toolbar_ecg_clean_with_ext.setEnabled(True)
    self.ax_ecg_clean_with_ext.clear()
    self.ax_ecg_clean_with_ext.set_title("Cleaned ECG Signal")
    self.ax_ecg_clean_with_ext.plot(times,full_data, label='Raw data')
    self.ax_ecg_clean_with_ext.plot(times,clean_data, label='Cleaned data')
    self.ax_ecg_clean_with_ext.set_xlabel("Time (s)")
    self.ax_ecg_clean_with_ext.set_ylabel("Amplitude")
    self.ax_ecg_clean_with_ext.legend()
    self.canvas_ecg_clean_with_ext.draw()
    

    self.btn_confirm_cleaning_with_ext.setEnabled(True)  # Enable the button after cleaning            

