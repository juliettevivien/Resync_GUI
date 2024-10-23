"""
TO DO: 
    - re-assess external sampling frequency, is not 2048Hz exactly !!!
    - Modify  synchronize_datasets_as_pickles function to work properly !!! 
    Use the other button for now instead: all as one .pickle
"""


import sys
import matplotlib
matplotlib.use('Qt5Agg')
import PyQt5
from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QListWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QInputDialog, QMessageBox, QStackedWidget
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_fieldtrip  # Import necessary functions
from mnelab.io.readers import read_raw
from os.path import join, basename, dirname
from pyxdf import resolve_streams
import scipy
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle

# import modules
from pyxdftools.xdfdata import XdfData

from functions.io import write_set
from functions.find_artifacts import (
    find_external_sync_artifact
)



class DataSet:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data
        self.file_path = None
        self.file_name = None
        self.selected_channel_index = None
        self.selected_channel_name = None
        self.ch_names = None
        self.sf = None  
        self.art_start= None 
        self.times = None
        self.last_artifact = None
        self.reset_timescale = None
        self.reset_data = None



class Button(QPushButton):
    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        
        # Apply a common style (rounded corners and light grey border)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};  /* Color provided dynamically */
                color: black;
                font-size: 14px;
                border-radius: 10px;  /* Rounded corners */
                border: 1px solid lightgrey;  /* Light grey border */
                padding: 5px 10px;  /* Padding for a nicer look */
            }}
            QPushButton:hover {{
                background-color: lightgray;  /* Hover effect */
            }}
            QPushButton:disabled {{
                background-color: lightgray; 
                color: gray;}}
        """)
        



class SyncGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.folder_path = None

        # Set up the main window
        self.setWindowTitle("ReSync GUI")
        self.setGeometry(100, 100, 1000, 600)

        # Create a stacked widget to hold multiple pages
        self.stacked_widget = QStackedWidget()

        # First page with your current layout
        self.first_page = self.create_first_page()
        self.stacked_widget.addWidget(self.first_page)

        # Second page with a different layout
        self.second_page = self.create_second_page()
        self.stacked_widget.addWidget(self.second_page)

        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)        

        # Central widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initialize datasets objects
        self.dataset_intra = DataSet()  # Dataset for the intracranial recording (STN recordings from Percept). Should be .mat file
        self.dataset_extra = DataSet()  # Dataset for the extracranial recording (EEG for example) Should be .xdf file


    def create_first_page(self):
        # Main vertical layout for the first page
        main_layout = QVBoxLayout()

        # Horizontal layout for .mat and .xdf panels
        panel_layout = QHBoxLayout()

        # Create a button to select the folder where to save the results
        self.btn_select_folder = Button("Select folder to save results", "lightyellow")
        self.btn_select_folder.clicked.connect(self.select_folder)
        main_layout.addWidget(self.btn_select_folder)

        self.label_saving_folder = QLabel("No saving folder selected")
        self.label_saving_folder.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.label_saving_folder)

        # Left panel for .mat file
        self.mat_panel = self.create_mat_panel()
        panel_layout.addLayout(self.mat_panel)

        # Right panel for .xdf file
        self.xdf_panel = self.create_xdf_panel()
        panel_layout.addLayout(self.xdf_panel)

        # Add the horizontal panel layout to the main layout
        main_layout.addLayout(panel_layout)

        # Synchronize and save buttons
        self.label_sync_and_save = QLabel('Synchronize and save as:')
        self.label_sync_and_save.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.label_sync_and_save)

        saving_layout = QHBoxLayout()

        self.btn_sync_as_set = Button("separately as .SET files", "lightyellow")
        self.btn_sync_as_set.setEnabled(False)
        self.btn_sync_as_set.clicked.connect(self.synchronize_datasets_as_set)
        saving_layout.addWidget(self.btn_sync_as_set)

        self.btn_sync_as_pickle = Button("separately as .pkl files", "lightyellow")
        self.btn_sync_as_pickle.setEnabled(False)
        self.btn_sync_as_pickle.clicked.connect(self.synchronize_datasets_as_pickles)
        saving_layout.addWidget(self.btn_sync_as_pickle)

        self.btn_all_as_pickle = Button("all as one .pkl", "lightyellow")
        self.btn_all_as_pickle.setEnabled(False)
        self.btn_all_as_pickle.clicked.connect(self.synchronize_datasets_as_one_pickle)
        saving_layout.addWidget(self.btn_all_as_pickle)

        main_layout.addLayout(saving_layout)

        # Add button to switch to the second page
        self.switch_to_second_page_button = Button("Go to timeshift analysis","#FFDAB9")
        self.switch_to_second_page_button.clicked.connect(self.show_second_page)
        self.switch_to_second_page_button.setEnabled(False)
        main_layout.addWidget(self.switch_to_second_page_button)

        # Create the first page widget and set the layout
        first_page_widget = QWidget()
        first_page_widget.setLayout(main_layout)
        return first_page_widget

    def create_second_page(self):
        # Second page layout
        layout_second_page = QVBoxLayout()

        self.btn_plot_synced_channels = Button("Plot synchronized channels", "lightyellow")
        self.btn_plot_synced_channels.clicked.connect(self.plot_synced_channels)
        layout_second_page.addWidget(self.btn_plot_synced_channels)

        # Set up canvas for matplotlib for .mat
        self.figure_synced, self.ax_synced = plt.subplots()
        self.canvas_synced = FigureCanvas(self.figure_synced)
        self.canvas_synced.setEnabled(False)  # Initially hidden

        # Set up the interactive toolbar to plot the synchronized signals together and check for timeshift
        self.toolbar_synced = NavigationToolbar(self.canvas_synced, self)
        self.toolbar_synced.setEnabled(False)
        layout_second_page.addWidget(self.toolbar_synced)
        layout_second_page.addWidget(self.canvas_synced)

        layout_second_page_selection = QHBoxLayout()
        layout_second_page_selection_mat = QVBoxLayout()
        layout_second_page_selection_xdf = QVBoxLayout()

        self.btn_select_last_art_mat = Button("Select last artifact in intracranial recording", "lightblue")
        self.btn_select_last_art_mat.clicked.connect(self.select_last_artifact_mat)
        layout_second_page_selection_mat.addWidget(self.btn_select_last_art_mat)

        self.label_select_last_art_mat = QLabel("No artifact selected")
        self.label_select_last_art_mat.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_second_page_selection_mat.addWidget(self.label_select_last_art_mat)


        self.btn_select_last_art_xdf = Button("Select last artifact in extracranial recording", "lightgreen")
        self.btn_select_last_art_xdf.clicked.connect(self.select_last_artifact_ext)
        layout_second_page_selection_xdf.addWidget(self.btn_select_last_art_xdf)

        self.label_select_last_art_xdf = QLabel("No artifact selected")
        self.label_select_last_art_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_second_page_selection_xdf.addWidget(self.label_select_last_art_xdf)

        layout_second_page_selection.addLayout(layout_second_page_selection_mat)
        layout_second_page_selection.addLayout(layout_second_page_selection_xdf)
        layout_second_page.addLayout(layout_second_page_selection)

        layout_timeshift = QHBoxLayout()
        self.btn_compute_timeshift = Button("Compute timeshift", "lightyellow")
        self.btn_compute_timeshift.clicked.connect(self.compute_timeshift)
        layout_timeshift.addWidget(self.btn_compute_timeshift)

        self.label_timeshift = QLabel("No timeshift computed")
        layout_timeshift.addWidget(self.label_timeshift)
        layout_second_page.addLayout(layout_timeshift)

        # Button to go back to the first page
        self.back_button = Button("Back to main window","#FFDAB9")
        self.back_button.clicked.connect(self.show_first_page)
        layout_second_page.addWidget(self.back_button)

        # Create the second page widget
        second_page_widget = QWidget()
        second_page_widget.setLayout(layout_second_page)
        return second_page_widget

    def show_first_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def show_second_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def create_mat_panel(self):
        """Create the left panel for .mat file processing."""
        layout = QVBoxLayout()

        # File selection button for .mat
        self.btn_load_file_mat = Button("Load .mat File", "lightblue")
        self.btn_load_file_mat.clicked.connect(self.load_mat_file)
        layout.addWidget(self.btn_load_file_mat)

        # Create a label to display the selected file name
        self.file_label_mat = QLabel("No file selected")
        self.file_label_mat.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.addWidget(self.file_label_mat)

        # Set up canvas for matplotlib for .mat
        self.figure_mat, self.ax_mat = plt.subplots()
        self.canvas_mat = FigureCanvas(self.figure_mat)
        self.canvas_mat.setEnabled(False)  # Initially hidden


        # Create a navigation toolbar and add it to the layout
        self.toolbar_mat = NavigationToolbar(self.canvas_mat, self)
        self.toolbar_mat.setEnabled(False) 
        layout.addWidget(self.toolbar_mat)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_mat)    # Add the canvas to the layout

        # Button layout for .mat channel selection and plotting
        self.channel_layout_mat = QVBoxLayout()
        self.channel_selection_layout_mat = QHBoxLayout()

        # Channel selection button for .mat (Initially hidden)
        self.btn_select_channel_mat = Button("Select Channel", "lightblue")
        self.btn_select_channel_mat.setEnabled(False)  # Initially inactive
        self.btn_select_channel_mat.clicked.connect(self.prompt_channel_name_mat)
        self.channel_selection_layout_mat.addWidget(self.btn_select_channel_mat)

        # Create a label to display the selected channel name
        self.channel_label_mat = QLabel("No channel selected")
        self.channel_label_mat.setEnabled(False) # Initially inactive
        self.channel_selection_layout_mat.addWidget(self.channel_label_mat)  
        self.channel_layout_mat.addLayout(self.channel_selection_layout_mat)      

        # Plot channel button for .mat (Initially hidden)
        self.btn_plot_channel_mat = Button("Plot Selected Channel", "lightblue")
        self.btn_plot_channel_mat.setEnabled(False)  # Initially inactive
        self.btn_plot_channel_mat.clicked.connect(self.plot_channel_mat)
        self.channel_layout_mat.addWidget(self.btn_plot_channel_mat)


        self.artifact_layout_mat = QHBoxLayout()
        self.automatic_artifact_layout_mat = QVBoxLayout()
        self.manual_artifact_layout_mat = QVBoxLayout()


        # Plot artifact detection button for .mat (Initially hidden)
        self.btn_artifact_detect_mat = Button("Automatic detection synchronization artifact", "lightblue")
        self.btn_artifact_detect_mat.setEnabled(False)  # Initially hidden
        self.btn_artifact_detect_mat.clicked.connect(self.detect_artifacts_mat)
        self.automatic_artifact_layout_mat.addWidget(self.btn_artifact_detect_mat)
        self.label_automatic_artifact_time_mat = QLabel("No artifact automatically detected")
        self.label_automatic_artifact_time_mat.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_automatic_artifact_time_mat.setVisible(False)  # Initially hidden
        self.automatic_artifact_layout_mat.addWidget(self.label_automatic_artifact_time_mat)

        self.btn_manual_select_artifact_mat = Button("Manual detection synchronization artifact", "lightblue") 
        self.btn_manual_select_artifact_mat.setEnabled(False)
        self.btn_manual_select_artifact_mat.clicked.connect(self.manual_selection_mat)
        self.manual_artifact_layout_mat.addWidget(self.btn_manual_select_artifact_mat)
        self.label_manual_artifact_time_mat = QLabel("No artifact manually selected")
        self.label_manual_artifact_time_mat.setVisible(False)
        self.label_manual_artifact_time_mat.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.manual_artifact_layout_mat.addWidget(self.label_manual_artifact_time_mat)        
        
        self.artifact_layout_mat.addLayout(self.automatic_artifact_layout_mat)
        self.artifact_layout_mat.addLayout(self.manual_artifact_layout_mat)

        # Add channel layout to main layout for .mat
        layout.addLayout(self.channel_layout_mat)
        layout.addLayout(self.artifact_layout_mat)


        return layout



    def load_mat_file(self):
        """Load .mat file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select MAT File", "", "MAT Files (*.mat);;All Files (*)")
        
        if file_name:
            try:
                # Load the .mat file using mne's read_raw_fieldtrip
                raw_data = read_raw_fieldtrip(file_name, info={}, data_name="data")
                self.dataset_intra.raw_data = raw_data  # Assign to dataset
                self.dataset_intra.sf = raw_data.info["sfreq"]  # Assign sampling frequency
                self.dataset_intra.ch_names = raw_data.ch_names  # Assign channel names
                self.dataset_intra.times = raw_data.times # Assign timescale
                self.file_label_mat.setText(f"Selected File: {basename(file_name)}")
                self.dataset_intra.file_name = basename(file_name)
                self.dataset_intra.file_path = dirname(file_name)
                
                # Show success message
                #QMessageBox.information(self, "Success", f"MAT file loaded successfully: {file_name}")

                # Show channel selection and plot buttons for .mat
                self.btn_select_channel_mat.setEnabled(True)
                self.channel_label_mat.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load .mat file: {e}")

    def prompt_channel_name_mat(self):
        """Prompt for channel name selection for .mat file."""
        if self.dataset_intra.raw_data:
            try:
                channel_names = self.dataset_intra.ch_names  # List of channel names
                channel_name, ok = QInputDialog.getItem(self, "Channel Selection", "Select a channel:", channel_names, 0, False)

                if ok and channel_name:  # Check if a channel was selected
                    self.dataset_intra.selected_channel_name = channel_name
                    self.dataset_intra.selected_channel_index = channel_names.index(channel_name)  # Get the index of the selected channel
                    self.channel_label_mat.setText(f"Selected Channel: {channel_name}")
                    # Enable the plot button since a channel has been selected
                    self.btn_plot_channel_mat.setEnabled(True)
                    self.btn_artifact_detect_mat.setEnabled(True)    
                    self.label_automatic_artifact_time_mat.setVisible(True)   
                    self.btn_manual_select_artifact_mat.setEnabled(True)
                    self.label_manual_artifact_time_mat.setVisible(True)
                    

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")


    def plot_channel_mat(self):
        """Plot the selected channel data from the .mat file."""
        if self.dataset_intra.raw_data and self.dataset_intra.selected_channel_index is not None:
            self.canvas_mat.setEnabled(True)
            self.toolbar_mat.setEnabled(True)
            self.ax_mat.clear()
            channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
            times = self.dataset_intra.times
            self.ax_mat.plot(times, channel_data)
            self.ax_mat.set_title(f"Channel {self.dataset_intra.selected_channel_index} data - {self.dataset_intra.selected_channel_name}")
            self.ax_mat.set_xlabel("Time (s)")
            self.ax_mat.set_ylabel("Amplitude")
            self.canvas_mat.draw()


    def detect_artifacts_mat(self):
        thres_window = round(self.dataset_intra.sf * 2)
        data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        thres = np.ptp(data[:thres_window])
        # Compute absolute value to be invariant to the polarity of the signal
        abs_data = np.abs(data)
        # Check where the data exceeds the threshold
        over_thres = np.where(abs_data[0:] > thres)[0][0]
        # Take last sample that lies within the value distribution of the thres_window before the threshold passing
        # The percentile is something that can be varied
        stim_idx = np.where(
            abs_data[:over_thres] <= np.percentile(abs_data[:over_thres], 95)
        )[0][-1]
        self.dataset_intra.art_start = stim_idx / self.dataset_intra.sf
        print(f"Artifact detected in LFP data at time: {self.dataset_intra.art_start} s")
        self.plot_scatter_channel_mat(art_start_mat = self.dataset_intra.art_start)
        self.update_synchronize_button_state()  # Check if we can enable the button
        self.label_automatic_artifact_time_mat.setText(f"Artifact start: {self.dataset_intra.art_start} s")


    def plot_scatter_channel_mat(self, art_start_mat=None):
        """Plot scatter plot of the selected channel data."""
        
        self.toolbar_mat.setEnabled(True)
        self.canvas_mat.setEnabled(True)
        self.ax_mat.clear()
        
        # Plot the channel data
        channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        times = self.dataset_intra.raw_data.times  # Time vector corresponding to the data points
        
        # Plot scatter points
        start = int(round(art_start_mat * self.dataset_intra.sf)-round(self.dataset_intra.sf/10))
        end = int(round(art_start_mat * self.dataset_intra.sf)+round(self.dataset_intra.sf/10))
        times_array = np.array(times)
        channel_data_array = np.array(channel_data)
        self.ax_mat.scatter(times_array[start:end], channel_data_array[start:end], s=5)


        # Highlight artifact start points if available
        if art_start_mat is not None:
                self.ax_mat.axvline(x=art_start_mat, color='red', linestyle='--', label='Artifact Start')

        self.ax_mat.legend()
        
        # Allow interactive features like zoom and pan
        self.canvas_mat.draw()



    def manual_selection_mat(self):
        self.toolbar_mat.setEnabled(True)
        self.canvas_mat.setEnabled(True)
        self.ax_mat.clear()
        data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        timescale = self.dataset_intra.times

        pos = []

        self.ax_mat.scatter(timescale, data, s=8)
        self.canvas_mat.draw()
        self.ax_mat.set_title(
            'Right click on the plot to select the start of the artifact (shown by the black "+")'
        )

        (plus_symbol,) = self.ax_mat.plot([], [], "k+", markersize=10)

        def onclick(event):
            if event.inaxes is not None:  # Check if the click is inside the axes
                if event.button == MouseButton.RIGHT:
                    pos.append([event.xdata, event.ydata])

                    # Update the position of the black "+" symbol
                    closest_index_x = np.argmin(np.abs(timescale - event.xdata))
                    closest_value_x = timescale[closest_index_x]
                    closest_value_y = data[closest_index_x]
                    plus_symbol.set_data([closest_value_x], [closest_value_y])
                    self.canvas_xdf.draw()
                    self.dataset_intra.art_start = closest_value_x
                    self.label_manual_artifact_time_mat.setText(f"Selected Artifact start: {closest_value_x} s")

        self.canvas_mat.mpl_connect("button_press_event", onclick)
        self.update_synchronize_button_state()
    





    def create_xdf_panel(self):
        """Create the right panel for .xdf file processing."""
        layout = QVBoxLayout()

        # File selection button for .xdf
        self.btn_load_file_xdf = Button("Load .xdf File", "lightgreen")
        self.btn_load_file_xdf.clicked.connect(self.load_xdf_file)
        layout.addWidget(self.btn_load_file_xdf)

        # Create a label to display the selected file name
        self.file_label_xdf = QLabel("No file selected")
        self.file_label_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.addWidget(self.file_label_xdf)

        # Set up canvas for matplotlib for .xdf
        self.figure_xdf, self.ax_xdf = plt.subplots()
        self.canvas_xdf = FigureCanvas(self.figure_xdf)
        self.canvas_xdf.setEnabled(False)

        # Create a navigation toolbar and add it to the layout
        self.toolbar_xdf = NavigationToolbar(self.canvas_xdf, self)
        self.toolbar_xdf.setEnabled(False) 
        layout.addWidget(self.toolbar_xdf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_xdf)    # Add the canvas to the layout


        # Button layout for .xdf channel selection and plotting
        self.channel_layout_xdf = QVBoxLayout()
        self.channel_selection_layout_xdf = QHBoxLayout()

        # Channel selection button for .xdf (Initially hidden)
        self.btn_select_channel_xdf = Button("Select Channel", "lightgreen")
        self.btn_select_channel_xdf.setEnabled(False)  # Initially inactive
        self.btn_select_channel_xdf.clicked.connect(self.select_channel_xdf)
        self.channel_selection_layout_xdf.addWidget(self.btn_select_channel_xdf)

        # Create a label to display the selected channel name
        self.channel_label_xdf = QLabel("No channel selected")
        self.channel_label_xdf.setEnabled(False) # Initially inactive
        self.channel_selection_layout_xdf.addWidget(self.channel_label_xdf)  
        self.channel_layout_xdf.addLayout(self.channel_selection_layout_xdf)  

        # Plot channel button for .xdf (Initially hidden)
        self.btn_plot_channel_xdf = Button("Plot Selected Channel", "lightgreen")
        self.btn_plot_channel_xdf.setEnabled(False)  # Initially hidden
        self.btn_plot_channel_xdf.clicked.connect(self.plot_channel_xdf)
        self.channel_layout_xdf.addWidget(self.btn_plot_channel_xdf)


        self.artifact_layout_xdf = QHBoxLayout()
        self.automatic_artifact_layout_xdf = QVBoxLayout()
        self.manual_artifact_layout_xdf = QVBoxLayout()

        # Plot artifact detection button for .xdf (Initially hidden)
        self.btn_artifact_detect_xdf = Button("Automatic detection synchronization artifact", "lightgreen")
        self.btn_artifact_detect_xdf.setEnabled(False)  # Initially hidden
        self.btn_artifact_detect_xdf.clicked.connect(self.detect_artifacts_xdf)
        self.automatic_artifact_layout_xdf.addWidget(self.btn_artifact_detect_xdf)
        self.label_automatic_artifact_time_xdf = QLabel("No artifact automatically detected")
        self.label_automatic_artifact_time_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_automatic_artifact_time_xdf.setVisible(False)  # Initially hidden
        self.automatic_artifact_layout_xdf.addWidget(self.label_automatic_artifact_time_xdf)

        self.btn_manual_select_artifact_xdf = Button("Manual detection synchronization artifact", "lightgreen")    
        self.btn_manual_select_artifact_xdf.setEnabled(False)
        self.btn_manual_select_artifact_xdf.clicked.connect(self.manual_selection_xdf)
        self.manual_artifact_layout_xdf.addWidget(self.btn_manual_select_artifact_xdf)
        self.label_manual_artifact_time_xdf = QLabel("No artifact manually selected")
        self.label_manual_artifact_time_xdf.setVisible(False)
        self.label_manual_artifact_time_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.manual_artifact_layout_xdf.addWidget(self.label_manual_artifact_time_xdf)

        self.artifact_layout_xdf.addLayout(self.automatic_artifact_layout_xdf)
        self.artifact_layout_xdf.addLayout(self.manual_artifact_layout_xdf)


        # Add channel layout to main layout for .xdf
        layout.addLayout(self.channel_layout_xdf)
        layout.addLayout(self.artifact_layout_xdf)


        return layout


    def load_xdf_file(self):
        """Load .xdf file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select XDF File", "", "XDF Files (*.xdf);;All Files (*)")
        self.file_label_xdf.setText(f"Selected File: {basename(file_name)}")
        self.dataset_extra.file_name = basename(file_name)
        self.dataset_extra.file_path = dirname(file_name)
        
        if file_name:
            try:
                # Load the .xdf file using the read_raw function
                stream_id = self.find_EEG_stream(file_name, stream_name='SAGA')
                raw_data = read_raw(file_name, stream_ids=[stream_id], preload=True)
                self.dataset_extra.raw_data = raw_data
                self.dataset_extra.sf = round(raw_data.info["sfreq"])  # Get the sampling frequency
                self.dataset_extra.ch_names = raw_data.ch_names  # Get the channel names
                self.dataset_extra.times = raw_data.times # Get the timescale
                #QMessageBox.information(self, "Success", f"XDF file loaded successfully: {file_name}")

                # Show channel selection and plot buttons for .xdf
                self.channel_label_xdf.setEnabled(True)
                self.btn_select_channel_xdf.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load .xdf file: {e}")


    def select_channel_xdf(self):
        """Open a dialog to select a channel by name from the .xdf file."""
        if self.dataset_extra.raw_data is not None:
            channel_names = self.dataset_extra.ch_names  # Get the channel names
            dialog = QWidget()
            dialog.setWindowTitle("Select Channel")
            layout = QVBoxLayout(dialog)

            # Create a list widget to display channel names
            channel_list = QListWidget(dialog)
            channel_list.addItems(channel_names)  # Add channel names to the list
            layout.addWidget(channel_list)

            # Create OK and Cancel buttons
            ok_button = QPushButton("OK", dialog)
            cancel_button = QPushButton("Cancel", dialog)
            layout.addWidget(ok_button)
            layout.addWidget(cancel_button)

            # Define button actions
            def on_ok():
                selected_items = channel_list.selectedItems()
                if selected_items:
                    self.dataset_extra.selected_channel_name = selected_items[0].text()  # Get the selected channel name
                    self.dataset_extra.selected_channel_index = channel_names.index(self.dataset_extra.selected_channel_name)  # Get the index
                    self.channel_label_xdf.setText(f"Selected Channel: {self.dataset_extra.selected_channel_name}")
                    self.btn_plot_channel_xdf.setEnabled(True)
                    dialog.close()
                    self.btn_artifact_detect_xdf.setEnabled(True)
                    self.label_automatic_artifact_time_xdf.setVisible(True)
                    self.btn_manual_select_artifact_xdf.setEnabled(True)
                    self.label_manual_artifact_time_xdf.setVisible(True)
                    self.update_synchronize_button_state()  # Check if we can enable the button

            def on_cancel():
                dialog.close()

            ok_button.clicked.connect(on_ok)
            cancel_button.clicked.connect(on_cancel)

            dialog.setLayout(layout)
            dialog.show()


    def detect_artifacts_xdf(self):
        """Detect artifacts in .xdf data."""
        channel_data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
        self.dataset_extra.art_start = find_external_sync_artifact(data=self.detrend_data(channel_data), sf_external=self.dataset_extra.sf, times = self.dataset_extra.times, start_index=0)
        print(f"Artifact detected in BIP data at time: {self.dataset_extra.art_start} s")
        self.label_automatic_artifact_time_xdf.setText(f"Artifact start detected at: {self.dataset_extra.art_start} s")

        # Plot the channel with artifact
        self.plot_scatter_channel_xdf(art_start_BIP=self.dataset_extra.art_start)
        self.update_synchronize_button_state()  # Check if we can enable the button



    def plot_channel_xdf(self):
        """Plot the selected channel data from the .xdf file."""
        if self.dataset_extra.raw_data and self.dataset_extra.selected_channel_index is not None:
            self.canvas_xdf.setEnabled(True)
            self.toolbar_xdf.setEnabled(True)
            self.ax_xdf.clear()
            channel_data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
            times = self.dataset_extra.times
            # apply a high-pass filter to detrend the data if the channel to plot is a bipolar channel:
            if self.dataset_extra.selected_channel_name.startswith("BIP"):
                channel_data_to_plot = self.detrend_data(channel_data)
            else:
                channel_data_to_plot = channel_data
            self.ax_xdf.plot(times, channel_data_to_plot)
            self.ax_xdf.set_title(f"Channel {self.dataset_extra.selected_channel_index} data - {self.dataset_extra.selected_channel_name}")
            self.ax_xdf.set_xlabel("Time (s)")
            self.ax_xdf.set_ylabel("Amplitude")
            self.canvas_xdf.draw()




    def plot_scatter_channel_xdf(self, art_start_BIP=None):
        """Plot scatter plot of the selected channel data."""
        self.toolbar_xdf.setEnabled(True)
        self.canvas_xdf.setEnabled(True)
        self.ax_xdf.clear()

        # Plot the channel data
        channel_data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
        channel_data_to_plot = self.detrend_data(channel_data)
        times = self.dataset_extra.raw_data.times  # Time vector corresponding to the data points
        
        # Plot scatter points
        start = int(round(art_start_BIP * self.dataset_extra.sf)-round(self.dataset_extra.sf/50))
        end = int(round(art_start_BIP * self.dataset_extra.sf)+round(self.dataset_extra.sf/50))
        times_array = np.array(times)
        channel_data_array = np.array(channel_data_to_plot)
        self.ax_xdf.scatter(times_array[start:end], channel_data_array[start:end], s=5)


        # Highlight artifact start points if available
        if art_start_BIP is not None:
                self.ax_xdf.axvline(x=art_start_BIP, color='red', linestyle='--', label='Artifact Start')


        self.ax_xdf.legend()
        
        # Allow interactive features like zoom and pan
        self.canvas_xdf.draw()


    def manual_selection_xdf(self):
        self.toolbar_xdf.setEnabled(True)
        self.canvas_xdf.setEnabled(True)
        self.ax_xdf.clear()
        data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
        timescale = self.dataset_extra.times

        pos = []

        self.ax_xdf.scatter(timescale, data, s=8)
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
                    closest_value_y = data[closest_index_x]
                    plus_symbol.set_data([closest_value_x], [closest_value_y])
                    self.canvas_xdf.draw()
                    self.dataset_extra.art_start = closest_value_x
                    self.label_manual_artifact_time_xdf.setText(f"Selected Artifact start: {closest_value_x} s")

        self.canvas_xdf.mpl_connect("button_press_event", onclick)
        self.update_synchronize_button_state()
    


    def update_synchronize_button_state(self):
        """Enable or disable the synchronize button based on file selection."""
        if self.dataset_intra.art_start is not None and self.dataset_extra.art_start is not None:
            self.btn_sync_as_set.setEnabled(True)
            self.btn_sync_as_pickle.setEnabled(True)
            self.btn_all_as_pickle.setEnabled(True)
            self.switch_to_second_page_button.setEnabled(True)
            self.switch_to_second_page_button.setEnabled(True)
        else:
            self.btn_sync_as_set.setEnabled(False)
            self.btn_sync_as_pickle.setEnabled(False)
            self.btn_all_as_pickle.setEnabled(False)
            self.switch_to_second_page_button.setEnabled(False)
            self.switch_to_second_page_button.setEnabled(False)



    def synchronize_datasets_as_set(self):
        events, _ = mne.events_from_annotations(self.dataset_extra.raw_data)
        inv_dic = {v: k for k, v in _.items()}

        ## offset intracranial recording (crop everything that is more than 1s before the artifact)
        tmax_lfp = max(self.dataset_intra.times)
        new_start_intracranial = self.dataset_intra.art_start - 1
        lfp_rec_offset = self.dataset_intra.raw_data.copy().crop(tmin=new_start_intracranial, tmax=tmax_lfp)

        ## offset external recording (crop everything that is more than 1s before the artifact)
        tmax_external = max(self.dataset_extra.times)
        new_start_external = self.dataset_extra.art_start - 1
        TMSi_rec_offset = self.dataset_extra.raw_data.copy().crop(tmin=new_start_external, tmax=tmax_external)
        #TMSi_rec_offset.plot(title='TMSi_rec_offset')

        ## transfer of the events from the external to the intracranial recording
        # create a duplicate of the events to manipulate it without changing the external one
        events_lfp = deepcopy(events)

        # get the events from the external in time instead of samples to account for the different sampling frequencies
        events_in_time = events[:,0]/self.dataset_extra.sf #### CAREFUL HERE THE SAMPLING FREQUENCY MIGHT BE WRONG ####

        # then offset the events in time to the new start of the external recording
        events_in_time_offset = events_in_time - new_start_external

        # convert the events in time offset to samples corresponding to the sampling frequency of the intracranial recording
        # because the annotations object works with samples, not timings
        events_in_time_offset_lfp = events_in_time_offset * self.dataset_intra.sf
        events_lfp[:,0] = events_in_time_offset_lfp

        ## create an annotation object for the intracranial recording
        annotations_lfp = mne.annotations_from_events(events_lfp, sfreq=self.dataset_intra.sf, event_desc=inv_dic)

        lfp_rec_offset.set_annotations(None) # make sure that no annotations are present
        lfp_rec_offset.set_annotations(annotations_lfp) # set the new annotations

        external_title = ("SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".set")
        lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".set")

        if self.folder_path is not None:
            fname_external_out=join(self.folder_path, external_title)
            fname_lfp_out =join(self.folder_path, lfp_title)
        else:
            fname_external_out = external_title
            fname_lfp_out = lfp_title

        TMSi_rec_offset_annotations_onset= (TMSi_rec_offset.annotations.onset) - new_start_external
        lfp_rec_offset_annotations_onset= (lfp_rec_offset.annotations.onset) - new_start_intracranial

        write_set(fname_external_out, TMSi_rec_offset, TMSi_rec_offset_annotations_onset)
        write_set(fname_lfp_out, lfp_rec_offset, lfp_rec_offset_annotations_onset)

        QMessageBox.information(self, "Synchronization", "Synchronization done. Both files saved as .SET")

    

    def synchronize_datasets_as_pickles(self): ## THIS FUNCTION DOES NOT WORK PROPERLY
        ## Intracranial ##
        # Crop beginning of LFP intracranial recording 1 second before first artifact:
        time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
        index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

        LFP_array = self.dataset_intra.raw_data.get_data()

        LFP_cropped = LFP_array[:, int(index_start_LFP) :].T
        LFP_df_offset = pd.DataFrame(LFP_cropped)
        LFP_df_offset.columns = self.dataset_intra.ch_names
        LFP_timescale_offset_s = self.dataset_intra.times[int(index_start_LFP):] - time_start_LFP_0

        # Save as pickle file:
        LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
        LFP_df_offset["time_stamp"] = LFP_timescale_offset_s
        lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".pkl")
        if self.folder_path is not None:
            LFP_filename = join(self.folder_path, lfp_title)
        else: LFP_filename = lfp_title
        # Save the dataset to a pickle file
        with open(LFP_filename, "wb") as file:
            pickle.dump(LFP_df_offset, file)


        ## External##
        stream_names = []
        stream_ids = []
        streams_dict = {}

        filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
        xdf_datas = XdfData(filepath).resolve_streams()

        for streams in range(1, len(xdf_datas['name'])+1, 1):
            stream_names.append(xdf_datas['name'][streams])

        for name in stream_names:
            stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

        for stream_nb in zip(stream_names, stream_ids):
            # create a dictionnary associating each streaming name from stream_names list to its corresponding stream_id:
            streams_dict[stream_nb[0]] = stream_nb[1]

        # LOAD ALL STREAMS IN A DICTIONNARY

        streams = {}

        for idx, (name, stream_id) in enumerate(streams_dict.items(), start=1):
            stream_name = f"{name}_stream"
            streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

        # CREATE GLOBAL VARIABLES FOR EACH STREAM
        for stream_name, stream_data in streams.items():
            globals()[stream_name] = stream_data.data()
            print(stream_name)

        # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
        art_start_0 = self.dataset_extra.art_start - 1
        # Get the original timestamps from both sources
        timestamps_global = globals()['SAGA_stream']['time_stamp']
        times_real = self.dataset_extra.times

        # Find the index in self.dataset_extra.times corresponding to art_start_0
        art_start_index = (times_real >= art_start_0).argmax()

        # Filter the timestamps from the global clock based on this index
        filtered_timestamps_global = timestamps_global[art_start_index:]
        art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])
        print(f"Art start in global timestamps: {art_start_in_globals}")


        # Iterate over the dynamically created variables
        for stream_name in streams.keys():
            # Find the index corresponding to the value of art_start_in_globals
            index = np.argmax(globals()[stream_name]['time_stamp'] >= art_start_in_globals)
            print(f"Index corresponding to art_start_in_globals in {stream_name}: {index}")
            
            # Crop the stream data from the index onwards
            stream_offset = globals()[stream_name].iloc[index:]
            
            # Create a copy of the cropped stream data
            stream_offset_copy = stream_offset.copy()
            
            # Offset the 'time_stamp' column
            stream_offset_copy['time_stamp'] = stream_offset_copy['time_stamp'] - art_start_in_globals
            
            # Reset the index
            stream_offset_copy.reset_index(drop=True, inplace=True)
            
            # Update the global variable with the modified data
            globals()[stream_name] = stream_offset_copy

        # Create a dictionary to hold the extracted DataFrames
        extracted_streams = {}

        # Iterate over the dynamically created variables
        for stream_name in streams.keys():
            # Extract the current stream DataFrame
            extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()

        ## saving as pickle:
        # Iterate over the extracted DataFrames
        for df_name, df_data in extracted_streams.items():
            # Generate the filename
            external_title = (f"{df_name}SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".pkl")
            # Create the full path to the file
            if self.folder_path is not None:
                filepath = join(self.folder_path, external_title)
            else:
                filepath = external_title
            # Save the DataFrame to a pickle file
            df_data.to_pickle(filepath)
        QMessageBox.information(self, "Synchronization", "Synchronization done. All files saved separately as .pickle")



    def synchronize_datasets_as_one_pickle(self):
        ## Intracranial ##
        # Crop beginning of LFP intracranial recording 1 second before first artifact:
        time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
        index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

        LFP_array = self.dataset_intra.raw_data.get_data()

        LFP_cropped = LFP_array[:, int(index_start_LFP) :].T
        LFP_df_offset = pd.DataFrame(LFP_cropped)
        LFP_df_offset.columns = self.dataset_intra.ch_names
        LFP_timescale_offset_s = self.dataset_intra.times[int(index_start_LFP):] - time_start_LFP_0

        # Prepare LFP dataframe
        LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
        LFP_df_offset["time_stamp"] = LFP_timescale_offset_s


        ## External ##
        stream_names = []
        stream_ids = []
        streams_dict = {}

        filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
        xdf_datas = XdfData(filepath).resolve_streams()

        for streams in range(1, len(xdf_datas['name'])+1, 1):
            stream_names.append(xdf_datas['name'][streams])

        for name in stream_names:
            stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

        for stream_nb in zip(stream_names, stream_ids):
            # create a dictionnary associating each streaming name from stream_names list to its corresponding stream_id:
            streams_dict[stream_nb[0]] = stream_nb[1]

        # LOAD ALL STREAMS IN A DICTIONNARY
        streams = {}

        for idx, (name, stream_id) in enumerate(streams_dict.items(), start=1):
            stream_name = f"{name}_stream"
            streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

        # CREATE GLOBAL VARIABLES FOR EACH STREAM
        for stream_name, stream_data in streams.items():
            globals()[stream_name] = stream_data.data()
            print(stream_name)

        # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
        art_start_0 = self.dataset_extra.art_start - 1
        # Get the original timestamps from both sources
        timestamps_global = globals()['SAGA_stream']['time_stamp']
        times_real = self.dataset_extra.times

        # Find the index in self.dataset_extra.times corresponding to art_start_0
        art_start_index = (times_real >= art_start_0).argmax()

        # Filter the timestamps from the global clock based on this index
        filtered_timestamps_global = timestamps_global[art_start_index:]
        art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])
        print(f"Art start in global timestamps: {art_start_in_globals}")


        # Iterate over the dynamically created variables
        for stream_name in streams.keys():
            # Find the index corresponding to the value of art_start_in_globals
            index = np.argmax(globals()[stream_name]['time_stamp'] >= art_start_in_globals)
            print(f"Index corresponding to art_start_in_globals in {stream_name}: {index}")

            # Crop the stream data from the index onwards
            stream_offset = globals()[stream_name].iloc[index:]

            # Create a copy of the cropped stream data
            stream_offset_copy = stream_offset.copy()

            # Offset the 'time_stamp' column
            stream_offset_copy['time_stamp'] = stream_offset_copy['time_stamp'] - art_start_in_globals
            
            # Reset the index
            stream_offset_copy.reset_index(drop=True, inplace=True)
            
            # Update the global variable with the modified data
            globals()[stream_name] = stream_offset_copy

        # Create a dictionary to hold the extracted DataFrames
        extracted_streams = {}

        # Iterate over the dynamically created variables
        for stream_name in streams.keys():
            # Extract the current stream DataFrame
            extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()


        # Iterate over the extracted DataFrames
        for df_name, df_data in extracted_streams.items():
            # Create separate DataFrame variables with specific names
            globals()[df_name] = pd.DataFrame(df_data)


        # Create an empty list to hold the DataFrames
        all_dfs = []

        # Iterate over the extracted DataFrames
        for df_name, df_data in extracted_streams.items():
            # Create a DataFrame with a MultiIndex containing the df_name as the top level
            df = pd.DataFrame(df_data)
            df.columns = pd.MultiIndex.from_product([[df_name], df.columns])  # Add df_name as the top level of column index
            all_dfs.append(df)

        # Concatenate all DataFrames in the list along axis 1 (columns)
        LSL_df = pd.concat(all_dfs, axis=1)

        # Assuming LFP_df_offset is your new DataFrame
        # First, adjust its column names to include a MultiIndex with the header 'LFP'
        LFP_df_offset.columns = pd.MultiIndex.from_product([['df_LFP'], LFP_df_offset.columns])

        # Concatenate LFP_df_offset on top of big_df along axis 1 (columns)
        final_df = pd.concat([LFP_df_offset, LSL_df], axis=1)
        
        ## saving as pickle:
        # Generate the filename
        filename = f"{self.dataset_extra.file_name[:-4]}_synchronized_data.pkl"
        # Create the full path to the file
        if self.folder_path is not None:
            filepath = join(self.folder_path, filename)
        else:
            filepath = filename
        # Save the DataFrame to a pickle file
        final_df.to_pickle(filepath)
        print(f"DataFrame {filename} saved as pickle to {filepath}")
        QMessageBox.information(self, "Synchronization", "Synchronization done. Everything saved as one .pickle file")


    def plot_synced_channels(self):
        self.toolbar_synced.setEnabled(True)
        self.canvas_synced.setEnabled(True)
        self.ax_synced.clear()

        # Plot the external channel synchronized
        data_extra = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
        data_extra_detrended = self.detrend_data(data_extra)
        timescale_extra = self.dataset_extra.times
        art_start_0_extra = self.dataset_extra.art_start - 1
        # Find the index in self.dataset_extra.times corresponding to art_start_0
        art_start_index_extra = (timescale_extra >= art_start_0_extra).argmax()
        offset_data_extra = data_extra_detrended[art_start_index_extra:]
        offset_timescale_extra = timescale_extra[art_start_index_extra:] - art_start_0_extra
        self.dataset_extra.reset_timescale = offset_timescale_extra
        self.dataset_extra.reset_data = offset_data_extra
        self.ax_synced.scatter(offset_timescale_extra, offset_data_extra, s=8, color='pink', label='External')

        # Plot the intracranial channel synchronized
        data_intra = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        timescale_intra = self.dataset_intra.times
        art_start_0_intra = self.dataset_intra.art_start - 1
        # Find the index in self.dataset_intra.times corresponding to art_start_0
        art_start_index_intra = (timescale_intra >= art_start_0_intra).argmax()
        offset_data_intra = data_intra[art_start_index_intra:]
        offset_timescale_intra = timescale_intra[art_start_index_intra:] - art_start_0_intra
        self.dataset_intra.reset_timescale = offset_timescale_intra
        self.dataset_intra.reset_data = offset_data_intra
        self.ax_synced.scatter(offset_timescale_intra, offset_data_intra, s=8, color='purple', label='Intracranial')
        self.ax_synced.legend(loc='upper left')
        self.canvas_synced.draw()


    def select_last_artifact_mat(self):
        # Check if we're already in external selection mode and prevent interference
        if hasattr(self, 'cid_extra') and self.cid_extra is not None:
            self.canvas_synced.mpl_disconnect(self.cid_extra)
            self.cid_extra = None

        pos_intra = []

        self.ax_synced.set_title(
            'Right click on the plot to select the start of the last artifact in the intracranial recording (shown by the black "+")'
        )
        self.canvas_synced.draw()

        # Create or update the intracranial "+" symbol
        if not hasattr(self, 'plus_symbol_intra'):
            self.plus_symbol_intra, = self.ax_synced.plot([], [], "k+", markersize=10)

        # Disconnect previous intracranial event listener, if any
        if hasattr(self, 'cid_intra') and self.cid_intra is not None:
            self.canvas_synced.mpl_disconnect(self.cid_intra)

        # Define the click handler for the intracranial selection
        def onclick(event_intra):
            if event_intra.inaxes is not None:  # Check if the click is inside the axes
                if event_intra.button == MouseButton.RIGHT:
                    pos_intra.append([event_intra.xdata, event_intra.ydata])

                    closest_index_x_intra = np.argmin(np.abs(self.dataset_intra.reset_timescale - event_intra.xdata))
                    closest_value_x_intra = self.dataset_intra.reset_timescale[closest_index_x_intra]
                    closest_value_y_intra = self.dataset_intra.reset_data[closest_index_x_intra]
                    self.plus_symbol_intra.set_data([closest_value_x_intra], [closest_value_y_intra])
                    self.canvas_synced.draw()

                    self.dataset_intra.last_artifact = closest_value_x_intra
                    self.label_select_last_art_mat.setText(f"Selected last artifact start: {self.dataset_intra.last_artifact} s")

        # Connect and store the ID of the intracranial event listener
        self.cid_intra = self.canvas_synced.mpl_connect("button_press_event", onclick)


    def select_last_artifact_ext(self):
        # Check if we're already in intracranial selection mode and prevent interference
        if hasattr(self, 'cid_intra') and self.cid_intra is not None:
            self.canvas_synced.mpl_disconnect(self.cid_intra)
            self.cid_intra = None

        pos_extra = []

        self.ax_synced.set_title(
            'Right click on the plot to select the start of the last artifact in the external recording (shown by the red "+")'
        )
        self.canvas_synced.draw()

        # Create or update the external "+" symbol
        if not hasattr(self, 'plus_symbol_extra'):
            self.plus_symbol_extra, = self.ax_synced.plot([], [], "r+", markersize=10)

        # Disconnect previous external event listener, if any
        if hasattr(self, 'cid_extra') and self.cid_extra is not None:
            self.canvas_synced.mpl_disconnect(self.cid_extra)

        # Define the click handler for the external selection
        def onclick(event_extra):
            if event_extra.inaxes is not None:  # Check if the click is inside the axes
                if event_extra.button == MouseButton.RIGHT:
                    pos_extra.append([event_extra.xdata, event_extra.ydata])

                    closest_index_x_extra = np.argmin(np.abs(self.dataset_extra.reset_timescale - event_extra.xdata))
                    closest_value_x_extra = self.dataset_extra.reset_timescale[closest_index_x_extra]
                    closest_value_y_extra = self.dataset_extra.reset_data[closest_index_x_extra]
                    self.plus_symbol_extra.set_data([closest_value_x_extra], [closest_value_y_extra])
                    self.canvas_synced.draw()

                    self.dataset_extra.last_artifact = closest_value_x_extra
                    self.label_select_last_art_xdf.setText(f"Selected last artifact start: {self.dataset_extra.last_artifact} s")

        # Connect and store the ID of the external event listener
        self.cid_extra = self.canvas_synced.mpl_connect("button_press_event", onclick)


    def compute_timeshift(self):
        timeshift = (self.dataset_extra.last_artifact - self.dataset_intra.last_artifact)*1000
        self.label_timeshift.setText(f"Timeshift: {timeshift} ms")


    def detrend_data(self, channel_data):
        b, a = scipy.signal.butter(1, 0.05, "highpass")
        detrended_data = scipy.signal.filtfilt(b, a, channel_data)

        return detrended_data

    def find_EEG_stream(self, fpath_external, stream_name):
        """Find the EEG stream in the .xdf file."""
        xdf_datas = resolve_streams(fpath_external)
        streams_dict = {stream['name']: stream['stream_id'] for stream in xdf_datas}
        stream_id = streams_dict.get(stream_name)

        if stream_id is None:
            raise ValueError(f"Stream '{stream_name}' not found in the XDF file.")
        
        return stream_id


    def select_folder(self):
        # Open a QFileDialog to select a folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.folder_path = folder_path
        self.label_saving_folder.setText(f"Results will be saved in: {folder_path}")
        
        if folder_path:  # Check if the user selected a folder
            print(f"Selected folder: {folder_path}")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyncGUI()
    window.show()
    sys.exit(app.exec_())
