import sys
import matplotlib
matplotlib.use('Qt5Agg')
import PyQt5
from PyQt5.QtWidgets import QLabel, QLineEdit, QRadioButton, QComboBox, QApplication, QMainWindow, QListWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QInputDialog, QMessageBox, QStackedWidget
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_fieldtrip  # Import necessary functions
from mnelab.io.readers import read_raw
from os.path import join, basename, dirname, exists
from pyxdf import resolve_streams
import scipy
from scipy.io import savemat
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
import webbrowser

# import modules
from pyxdftools.xdfdata import XdfData
from functions.tmsi_poly5reader import Poly5Reader

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
        self.max_y_value = None
        self.first_art_start_time = None
        self.last_art_start_time = None
        self.first_art_start_idx = None
        self.last_art_start_idx = None
        self.eff_sf = None



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
            QPushButton[active="true"] {{
                font-weight: bold;  /* Bold font for active button */
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
        self.setWindowIcon(QIcon("logo_resized.png"))
        self.setGeometry(100, 100, 1000, 600)

        # Create a stacked widget to hold multiple pages
        self.stacked_widget = QStackedWidget()

        # Create the pages and add them to the stacked widget
        self.first_page = self.create_first_page()
        self.second_page = self.create_second_page()
        self.third_page = self.create_third_page()
        self.fourth_page = self.create_fourth_page()
        self.stacked_widget.addWidget(self.first_page)
        self.stacked_widget.addWidget(self.fourth_page) # ecg cleaning, should be the second page
        self.stacked_widget.addWidget(self.second_page)
        self.stacked_widget.addWidget(self.third_page)


        # Create the header with navigation buttons
        header_layout = QHBoxLayout()
        self.menu_label = QLabel("MENU")
        self.btn_first = Button("Home Page","#cd9ddc")
        self.btn_first.clicked.connect(self.show_first_page)
        self.btn_fourth = Button("ECG Cleaning", "#cd9ddc")
        self.btn_fourth.clicked.connect(self.show_fourth_page)
        self.btn_second = Button("Effective Sampling Frequency correction", "#cd9ddc")
        self.btn_second.clicked.connect(self.show_second_page)
        self.btn_third = Button("Timeshift Analysis", "#cd9ddc")
        self.btn_third.clicked.connect(self.show_third_page)
        self.btn_help = Button("Help", "#cd9ddc")
        self.btn_help.clicked.connect(self.show_help)

        # Add buttons to the header layout
        header_layout.addWidget(self.menu_label)
        header_layout.addWidget(self.btn_first)
        header_layout.addWidget(self.btn_fourth)
        header_layout.addWidget(self.btn_second)
        header_layout.addWidget(self.btn_third)
        header_layout.addWidget(self.btn_help)
        header_layout.addStretch()

        # Create a widget for the header layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #d1d1d1; border-top: 2px solid #d3d3d3; border-left: 2px solid #d3d3d3;")
        self.menu_label.setStyleSheet("border: none; font-weight: bold;")

        # Main vertical layout for the entire GUI
        main_layout = QVBoxLayout()
        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.stacked_widget)

        # Create the footer with RESET button
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()

        # Add reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_app)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: "#cd9ddc";
                color: black;
                font-size: 18px;
                border-radius: 10px;
                border: 1px solid lightgrey;
                padding: 5px 30px;
            }
            QPushButton:hover {
                background-color: "lightgray";
            }
        """)
        footer_layout.addWidget(self.reset_button)
        footer_layout.addStretch()

        footer_widget = QWidget()
        footer_widget.setLayout(footer_layout)
        footer_widget.setStyleSheet("background-color: #d1d1d1; border-top: 2px solid #d3d3d3; border-left: 2px solid #d3d3d3;")

        main_layout.addWidget(footer_widget)


        # Central widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.update_button_styles(self.btn_first)  # Highlight the first button

        # Initialize datasets objects
        self.dataset_intra = DataSet()  # Dataset for the intracranial recording (STN recordings from Percept). Should be .mat file
        self.dataset_extra = DataSet()  # Dataset for the extracranial recording (EEG for example) Should be .xdf or .Poly5 file

    def reset_app(self):
        # Close current instance of the window
        self.close()
        
        # Open a new instance of SyncGUI
        self.new_window = SyncGUI()  # Create a new window instance
        self.new_window.show()  # Show the new window



    def create_first_page(self):
        # Main vertical layout for the first page
        main_layout = QVBoxLayout()

        # Horizontal layout for intracranial and external recordings panels
        panel_layout = QHBoxLayout()


        # Left panel for intracranial file
        self.panel_intra = self.create_panel_intra()
        panel_layout.addLayout(self.panel_intra)

        # Right panel for .xdf file
        self.xdf_panel = self.create_xdf_panel()
        panel_layout.addLayout(self.xdf_panel)

        # Add the horizontal panel layout to the main layout
        main_layout.addLayout(panel_layout)

        saving_folder_layout = QHBoxLayout()

        # Create a button to select the folder where to save the results
        self.btn_select_folder = Button("Select folder to save results", "lightyellow")
        self.btn_select_folder.clicked.connect(self.select_folder)
        saving_folder_layout.addWidget(self.btn_select_folder)

        self.label_saving_folder = QLabel("No saving folder selected")
        self.label_saving_folder.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        saving_folder_layout.addWidget(self.label_saving_folder)

        main_layout.addLayout(saving_folder_layout)

        # Synchronize and save buttons
        self.label_sync_and_save = QLabel('Synchronize and save as:')
        self.label_sync_and_save.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.label_sync_and_save)

        # layout for saving from .xdf files
        saving_xdf_layout = QHBoxLayout()

        self.label_saving_xdf = QLabel("If the external file was .xdf:")
        saving_xdf_layout.addWidget(self.label_saving_xdf)

        self.btn_sync_as_set = Button("separately as .SET files", "lightyellow")
        self.btn_sync_as_set.setEnabled(False)
        self.btn_sync_as_set.clicked.connect(self.synchronize_datasets_as_set)
        saving_xdf_layout.addWidget(self.btn_sync_as_set)

        self.btn_sync_as_pickle = Button("separately as .pkl files", "lightyellow")
        self.btn_sync_as_pickle.setEnabled(False)
        self.btn_sync_as_pickle.clicked.connect(self.synchronize_datasets_as_pickles)
        saving_xdf_layout.addWidget(self.btn_sync_as_pickle)

        self.btn_all_as_pickle = Button("all as one .pkl", "lightyellow")
        self.btn_all_as_pickle.setEnabled(False)
        self.btn_all_as_pickle.clicked.connect(self.synchronize_datasets_as_one_pickle)
        saving_xdf_layout.addWidget(self.btn_all_as_pickle)

        main_layout.addLayout(saving_xdf_layout)

        # layout for saving from .Poly5 files
        saving_poly5_layout = QHBoxLayout()

        self.label_saving_poly5 = QLabel("If the external file was .Poly5:")
        saving_poly5_layout.addWidget(self.label_saving_poly5)

        self.btn_sync_as_mat = Button("separately as .mat files", "lightyellow")
        self.btn_sync_as_mat.setEnabled(False)
        self.btn_sync_as_mat.clicked.connect(self.synchronize_datasets_as_mat)
        saving_poly5_layout.addWidget(self.btn_sync_as_mat)

        self.btn_sync_as_pickle_from_poly5 = Button("separately as .pkl files", "lightyellow")
        self.btn_sync_as_pickle_from_poly5.setEnabled(False)
        #self.btn_sync_as_pickle_from_poly5.clicked.connect(self.synchronize_datasets_as_pickles_from_poly5)
        saving_poly5_layout.addWidget(self.btn_sync_as_pickle_from_poly5)

        main_layout.addLayout(saving_poly5_layout)        


        # Create the first page widget and set the layout
        first_page_widget = QWidget()
        first_page_widget.setLayout(main_layout)
        return first_page_widget



    def create_second_page(self):
        # Main vertical layout for the first page
        layout_second_page = QVBoxLayout()

        # Horizontal layout for intracranial and external recordings panels
        panel_layout = QHBoxLayout()

        # Left panel for intracranial file
        self.panel_intra_sf_correction = self.create_panel_intra_sf_correction()
        panel_layout.addLayout(self.panel_intra_sf_correction)

        # Right panel for .xdf file
        self.panel_extra_sf_correction = self.create_panel_extra_sf_correction()
        panel_layout.addLayout(self.panel_extra_sf_correction)

        # Add the horizontal panel layout to the main layout
        layout_second_page.addLayout(panel_layout)

        computing_sf_layout = QHBoxLayout()
        # Compute effective sampling frequency button
        self.btn_compute_eff_sf = Button("Compute effective sampling frequency", "lightyellow")
        self.btn_compute_eff_sf.clicked.connect(self.compute_eff_sf)
        self.btn_compute_eff_sf.setEnabled(False)
        computing_sf_layout.addWidget(self.btn_compute_eff_sf)

        self.label_eff_sf = QLabel("No effective sampling frequency computed yet. Select all requested time points first.")
        computing_sf_layout.addWidget(self.label_eff_sf)
        layout_second_page.addLayout(computing_sf_layout)

        # Create the third page widget and set the layout
        third_page_widget = QWidget()
        third_page_widget.setLayout(layout_second_page)
        return third_page_widget



    def create_third_page(self):
        # Second page layout
        layout_third_page = QVBoxLayout()

        self.btn_plot_synced_channels = Button("Plot synchronized channels", "lightyellow")
        self.btn_plot_synced_channels.clicked.connect(self.plot_synced_channels)
        self.btn_plot_synced_channels.setEnabled(False)
        layout_third_page.addWidget(self.btn_plot_synced_channels)

        # Set up canvas for matplotlib for synced datasets
        self.figure_synced, self.ax_synced = plt.subplots()
        self.canvas_synced = FigureCanvas(self.figure_synced)
        self.canvas_synced.setEnabled(False)  # Initially hidden

        # Set up the interactive toolbar to plot the synchronized signals together and check for timeshift
        self.toolbar_synced = NavigationToolbar(self.canvas_synced, self)
        self.toolbar_synced.setEnabled(False)
        layout_third_page.addWidget(self.toolbar_synced)
        layout_third_page.addWidget(self.canvas_synced)

        layout_third_page_selection = QHBoxLayout()
        layout_third_page_selection_intra = QVBoxLayout()
        layout_third_page_selection_xdf = QVBoxLayout()

        self.btn_select_last_art_intra = Button("Select last artifact in intracranial recording", "lightblue")
        self.btn_select_last_art_intra.clicked.connect(self.select_last_artifact_intra)
        self.btn_select_last_art_intra.setEnabled(False)
        layout_third_page_selection_intra.addWidget(self.btn_select_last_art_intra)

        self.label_select_last_art_intra = QLabel("No artifact selected")
        self.label_select_last_art_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_third_page_selection_intra.addWidget(self.label_select_last_art_intra)


        self.btn_select_last_art_xdf = Button("Select last artifact in extracranial recording", "lightgreen")
        self.btn_select_last_art_xdf.clicked.connect(self.select_last_artifact_ext)
        self.btn_select_last_art_xdf.setEnabled(False)
        layout_third_page_selection_xdf.addWidget(self.btn_select_last_art_xdf)

        self.label_select_last_art_xdf = QLabel("No artifact selected")
        self.label_select_last_art_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_third_page_selection_xdf.addWidget(self.label_select_last_art_xdf)

        layout_third_page_selection.addLayout(layout_third_page_selection_intra)
        layout_third_page_selection.addLayout(layout_third_page_selection_xdf)
        layout_third_page.addLayout(layout_third_page_selection)

        layout_timeshift = QHBoxLayout()
        self.btn_compute_timeshift = Button("Compute timeshift", "lightyellow")
        self.btn_compute_timeshift.clicked.connect(self.compute_timeshift)
        self.btn_compute_timeshift.setEnabled(False)
        layout_timeshift.addWidget(self.btn_compute_timeshift)

        self.label_timeshift = QLabel("No timeshift computed")
        layout_timeshift.addWidget(self.label_timeshift)
        layout_third_page.addLayout(layout_timeshift)


        # Create the third page widget
        third_page_widget = QWidget()
        third_page_widget.setLayout(layout_third_page)
        return third_page_widget


    def create_fourth_page(self):
        # Main vertical layout for the first page
        layout_fourth_page = QHBoxLayout()

        # Vertical layout on the left for the plotting and for the visualization of the cleaned data
        layout_left_fourth_page = QVBoxLayout()

        layout_channel_selection_cleaning = QHBoxLayout()
        # Create a button to choose a channel for cleaning
        self.btn_choose_channel = Button("Choose channel for cleaning", "lightyellow")
        self.btn_choose_channel.clicked.connect(self.choose_channel_for_cleaning)
        self.btn_choose_channel.setEnabled(False) # Should be enabled only when the file is loaded

        # add a label to show the selected channel name
        self.label_selected_channel = QLabel("No channel selected")

        layout_channel_selection_cleaning.addWidget(self.btn_choose_channel)
        layout_channel_selection_cleaning.addWidget(self.label_selected_channel)

        # Create a button to start the ECG cleaning process
        self.btn_start_ecg_cleaning = Button("Start ECG cleaning", "lightyellow")
        self.btn_start_ecg_cleaning.clicked.connect(self.start_ecg_cleaning)
        self.btn_start_ecg_cleaning.setEnabled(False) # Should be enabled only when the file is loaded

        # add a matplotlib canvas to visualize the raw data
        self.figure_ecg, self.ax_ecg = plt.subplots()
        self.canvas_ecg = FigureCanvas(self.figure_ecg)
        #self.canvas_ecg.setEnabled(False)  # Initially hidden

        # Set up the interactive toolbar to plot the signal
        self.toolbar_ecg = NavigationToolbar(self.canvas_ecg, self)
        self.toolbar_ecg.setEnabled(False)

        # create a vertical layout below the plot to enter start and end times for the cleaning
        layout_start_end_time = QHBoxLayout()
        self.label_start_time = QLabel("Start time for cleaning (s):")
        self.box_start_time = QLineEdit()
        self.box_start_time.setPlaceholderText("0")  # Set a placeholder text
        self.box_start_time.setEnabled(False)  # Initially disabled
        self.label_end_time = QLabel("End time for cleaning (s):")
        self.box_end_time = QLineEdit()
        self.box_end_time.setPlaceholderText("0")  # Set a placeholder text
        self.box_end_time.setEnabled(False)  # Initially disabled
        self.label_artifact_polarity = QLabel("Artifact polarity:")
        # add 2 radio buttons to select the artifact polarity:
        self.radio_button_up = QRadioButton("Up")
        self.radio_button_down = QRadioButton("Down")
        self.radio_button_up.setChecked(True)  # Set the default selection to "Up"
        self.radio_button_down.setChecked(False)  # Set the default selection to "Down"
        self.label_thresh_ecg = QLabel("Threshold:")
        self.box_thresh_ecg = QComboBox()
        self.box_thresh_ecg.addItems(['95', '96', '97', '98', '99'])
        self.box_thresh_ecg.setEnabled(False)
        self.btn_validate_start_end_time = Button("Validate", "lightyellow")
        self.btn_validate_start_end_time.clicked.connect(self.validate_start_end_time)
        self.btn_validate_start_end_time.setEnabled(False)  # Initially disabled

        layout_start_end_time.addWidget(self.label_start_time)
        layout_start_end_time.addWidget(self.box_start_time)
        layout_start_end_time.addWidget(self.label_end_time)
        layout_start_end_time.addWidget(self.box_end_time)
        layout_start_end_time.addWidget(self.label_artifact_polarity)
        layout_start_end_time.addWidget(self.radio_button_up)
        layout_start_end_time.addWidget(self.radio_button_down)
        layout_start_end_time.addWidget(self.label_thresh_ecg)
        layout_start_end_time.addWidget(self.box_thresh_ecg)
        layout_start_end_time.addWidget(self.btn_validate_start_end_time)

        # Create a button to plot the overlap raw and clean signal for inspection
        self.btn_plot_ecg_clean = Button("Plot overlap raw and clean signal for inspection", "lightyellow")
        #self.btn_plot_ecg_clean.clicked.connect(self.plot_ecg_clean)
        self.btn_plot_ecg_clean.setEnabled(False) # Should be enabled only when self.dataset_intra.ecg is not None

        # add another matplotlib canvas to visualize the cleaned data
        self.figure_ecg_clean, self.ax_ecg_clean = plt.subplots()
        self.canvas_ecg_clean = FigureCanvas(self.figure_ecg_clean)
        #self.canvas_ecg_clean.setEnabled(False)  # Initially hidden

        # Set up the interactive toolbar to plot the cleaned signal
        self.toolbar_ecg_clean = NavigationToolbar(self.canvas_ecg_clean, self)
        self.toolbar_ecg_clean.setEnabled(False)

        layout_left_fourth_page.addLayout(layout_channel_selection_cleaning)
        layout_left_fourth_page.addWidget(self.toolbar_ecg)
        layout_left_fourth_page.addWidget(self.canvas_ecg)  # Add the canvas to the layout
        layout_left_fourth_page.addLayout(layout_start_end_time)
        layout_left_fourth_page.addWidget(self.btn_start_ecg_cleaning)
        layout_left_fourth_page.addWidget(self.toolbar_ecg_clean)  # Add the toolbar to the layout
        layout_left_fourth_page.addWidget(self.canvas_ecg_clean)  # Add the canvas to the layout

        layout_right_fourth_page = QVBoxLayout()

        # Create a Header for the right panel
        self.label_ecg_artifact = QLabel("Verification tools")
        self.label_ecg_artifact.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_ecg_artifact.setStyleSheet("font-weight: bold; font-size: 18px;")

        # Create a button to show the detected peaks
        self.btn_show_detected_peaks = Button("Show detected peaks", "lightyellow")
        #self.btn_show_detected_peaks.clicked.connect(self.show_detected_peaks)
        self.btn_show_detected_peaks.setEnabled(False) # Should be enabled only when self.dataset_intra.ecg is not None

        # Add a canvas to visualize the detected peaks
        self.figure_detected_peaks, self.ax_detected_peaks = plt.subplots()
        self.canvas_detected_peaks = FigureCanvas(self.figure_detected_peaks)
        #self.canvas_detected_peaks.setEnabled(False)  # Initially hidden

        # Add a toolbar to visualize the detected peaks
        self.toolbar_detected_peaks = NavigationToolbar(self.canvas_detected_peaks, self)
        self.toolbar_detected_peaks.setEnabled(False)

        # Create a button to plot the ECG artifact detected after cleaning
        self.btn_plot_ecg_artifact = Button("Plot ECG artifact detected after cleaning", "lightyellow")
        #self.btn_plot_ecg_artifact.clicked.connect(self.plot_ecg_artifact)
        self.btn_plot_ecg_artifact.setEnabled(False) # Should be enabled only when self.dataset_intra.ecg is not None

        # add a matplotlib canvas to visualize the ECG artifact detected
        self.figure_ecg_artifact, self.ax_ecg_artifact = plt.subplots()
        self.canvas_ecg_artifact = FigureCanvas(self.figure_ecg_artifact)

        # Create a button to confirm the cleaning and continue with the synchronization
        self.btn_confirm_cleaning = Button("Confirm cleaning and use cleaned channel for synchronization", "lightyellow")
        self.btn_confirm_cleaning.clicked.connect(self.confirm_cleaning)
        self.btn_confirm_cleaning.setEnabled(False) # Should be enabled only when self.dataset_intra.ecg is not None

        layout_right_fourth_page.addWidget(self.label_ecg_artifact)
        layout_right_fourth_page.addWidget(self.toolbar_detected_peaks)  # Add the toolbar to the layout
        layout_right_fourth_page.addWidget(self.canvas_detected_peaks)  # Add the canvas to the layout
        layout_right_fourth_page.addWidget(self.canvas_ecg_artifact)  # Add the canvas to the layout
        layout_right_fourth_page.addWidget(self.btn_confirm_cleaning)

        # Left panel for intracranial file
        #self.panel_intra_ecg = self.create_panel_intra_ecg()
        #panel_layout.addLayout(self.panel_intra_ecg)

        # Add the horizontal panel layout to the main layout
        layout_fourth_page.addLayout(layout_left_fourth_page)
        layout_fourth_page.addLayout(layout_right_fourth_page)

        # Create the first page widget and set the layout
        fourth_page_widget = QWidget()
        fourth_page_widget.setLayout(layout_fourth_page)
        return fourth_page_widget

    def confirm_cleaning(self):
        if self.dataset_intra.selected_channel_index_ecg == 0:
            # Replace the corresponding channel's data
            self.dataset_intra.raw_data._data[0,:] = self.dataset_intra.cleaned_ecg_left
            #self.dataset_intra.raw_data.get_data()[0] = self.dataset_intra.cleaned_ecg_left
            print("Cleaning confirmed. Replacing raw data with cleaned data in the left channel.")
        elif self.dataset_intra.selected_channel_index_ecg == 1:
            # Replace the corresponding channel's data
            self.dataset_intra.raw_data._data[1,:] = self.dataset_intra.cleaned_ecg_right
            #self.dataset_intra.raw_data.get_data()[1] = self.dataset_intra.cleaned_ecg_right
            print("Cleaning confirmed. Replacing raw data with cleaned data in the right channel.")
        """
        if self.dataset_intra.cleaned_ecg_left is not None and self.dataset_intra.selected_channel_index_ecg == 0:
            print("replacing left channel with cleaned data")
        elif self.dataset_intra.cleaned_ecg_right is not None and self.dataset_intra.selected_channel_index_ecg == 1:
            print("replacing right channel with cleaned data")
        """


    def start_ecg_cleaning(self):
        """Start the ECG cleaning process."""
        if self.dataset_intra.raw_data is not None and self.dataset_intra.selected_channel_index_ecg is not None:
            # Perform the ECG cleaning process here
            try:
                self.clean_ecg()
                """
                if self.dataset_intra.selected_channel_index_ecg == 0:
                # Assuming you have a method to clean the ECG data
                    self.clean_ecg(self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index_ecg])
                    
                elif self.dataset_intra.selected_channel_index_ecg == 1:
                    self.clean_ecg(self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index_ecg])    
                """

                self.btn_plot_ecg_clean.setEnabled(True)  # Enable the button after cleaning
                self.btn_show_detected_peaks.setEnabled(True)  # Enable the button after cleaning
                self.btn_plot_ecg_artifact.setEnabled(True)  # Enable the button after cleaning

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")


    def clean_ecg(self):
        full_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
        times = self.dataset_intra.raw_data.times
        start_rec = self.dataset_intra.start_time_left if self.dataset_intra.selected_channel_index_ecg == 0 else self.dataset_intra.start_time_right
        end_rec = self.dataset_intra.end_time_left if self.dataset_intra.selected_channel_index_ecg == 0 else self.dataset_intra.end_time_right
        sf_lfp= round(self.dataset_intra.raw_data.info['sfreq'])
        if self.dataset_intra.artifact_polarity == "down":
            REVERSED = True # true if cardiac artifacts are going downwards
        else:
            REVERSED = False

        # crop data to remove sync pulses (amplitude is too big, it messes with template detection)
        data_crop = full_data[int(start_rec*sf_lfp):int(end_rec*sf_lfp)]
        # keep beginning and end:
        beginning_part = full_data[:int(start_rec*sf_lfp)]
        end_part = full_data[int(end_rec*sf_lfp):]

        times_crop = times[int(start_rec*sf_lfp):int(end_rec*sf_lfp)]
        
        if REVERSED:
            cropped_data = - data_crop
            full_data = - full_data
        else:
            cropped_data = data_crop
            full_data = full_data

        ecg = {'proc': {}, 'stats': {}, 'cleandata': None, 'detected': False}
        ns = len(cropped_data)
        
        # Segment the signal into overlapping windows
        dwindow = int(round(sf_lfp))  # 1s window
        dmove = sf_lfp  # 1s step
        n_segments = (ns - dwindow) // dmove + 1
        
        x = np.array([cropped_data[i * dmove: i * dmove + dwindow] for i in range(n_segments) if i * dmove + dwindow <= ns])
        
        detected_peaks = []  # Store peak indices in the original timescale
        
        # Loop through each segment and find peaks
        for i in range(n_segments):
            segment = x[i]
            peaks, _ = scipy.signal.find_peaks(segment, height=np.percentile(segment, 90), distance=sf_lfp//2)  # Adjust threshold & min distance
            real_peaks = peaks + (i * dmove)  # Convert to original timescale
            detected_peaks.extend(real_peaks)

        detected_peaks = np.array(detected_peaks)
        
        # Define epoch window (-0.5s to +0.5s)
        pre_samples = int(0.5 * sf_lfp)
        post_samples = int(0.5 * sf_lfp)
        epoch_length = pre_samples + post_samples  # Total length of each epoch

        epochs = []  # Store extracted heartbeats

        for peak in detected_peaks:
            start = peak - pre_samples
            end = peak + post_samples
            
            if start >= 0 and end < ns:  # Ensure we don't go out of bounds
                epochs.append(cropped_data[start:end])

        epochs = np.array(epochs)

        # Compute average heartbeat template
        mean_epoch = np.mean(epochs, axis=0)
        ecg['proc']['template1'] = mean_epoch  # First ECG template

        # Plot the detected ECG epochs
        time = np.linspace(-0.5, 0.5, epoch_length)  # Time in seconds

        self.canvas_ecg_artifact.setEnabled(True)
        #self.toolbar_ecg_artifact.setEnabled(True)
        self.ax_ecg_artifact.clear()
        self.ax_ecg_artifact.set_title("Detected ECG epochs")

        for epoch in epochs:
            if REVERSED:
                epoch = -epoch
            self.ax_ecg_artifact.plot(time, epoch, color='gray', alpha=0.3)
        
        if REVERSED:
            mean_epoch = - mean_epoch

        self.ax_ecg_artifact.plot(time, mean_epoch, color='red', linewidth=2, label='Average ECG Template')
        self.ax_ecg_artifact.set_xlabel("Time (s)")
        self.ax_ecg_artifact.set_ylabel("Amplitude")
        self.ax_ecg_artifact.legend()
        self.canvas_ecg_artifact.draw()


        # Temporal correlation for ECG detection
        r = np.correlate(cropped_data, ecg['proc']['template1'], mode='same')
        threshold = np.percentile(r, 95)
        detected_peaks, _ = scipy.signal.find_peaks(r, height=threshold)

        if len(detected_peaks) < 5:
            return ecg  # Not enough peaks detected

        ecg['proc']['r'] = r
        ecg['proc']['thresh'] = threshold

        # Second pass for refining detection
        refined_template = np.mean([cropped_data[p - dwindow//2 : p + dwindow//2] for p in detected_peaks if p - dwindow//2 > 0 and p + dwindow//2 < ns], axis=0)
        ecg['proc']['template2'] = refined_template
        r2 = np.correlate(cropped_data, refined_template, mode='same')
        threshold2 = np.percentile(r2, self.dataset_intra.ecg_tresh)
        final_peaks, _ = scipy.signal.find_peaks(r2, height=threshold2)

        ecg['proc']['r2'] = r2
        ecg['proc']['thresh2'] = threshold2

        # Estimate HR
        peak_intervals = np.diff(final_peaks) / sf_lfp  # Convert to seconds
        hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
        ecg['stats']['hr'] = hr
        ecg['stats']['pctartefact'] = (1 - len(final_peaks) / ns) * 100

        # Check ECG detection validity
        if 55 <= hr <= 120 and len(final_peaks) > 10:
            ecg['detected'] = True

        # Remove artifacts (simple interpolation)
        clean_data = np.copy(cropped_data)
        for p in final_peaks:
            clean_data[max(0, p - 5): min(ns, p + 5)] = np.nan  # NaN out artifacts
        clean_data = np.interp(np.arange(ns), np.arange(ns)[~np.isnan(clean_data)], clean_data[~np.isnan(clean_data)])

        if REVERSED:
            full_data = - full_data
            clean_data = - clean_data

        clean_data_full = np.concatenate([beginning_part, clean_data, end_part])
        ecg['cleandata'] = clean_data_full

        if self.dataset_intra.selected_channel_index_ecg == 0:
            self.dataset_intra.cleaned_ecg_left = clean_data_full
            self.dataset_intra.detected_peaks_left = final_peaks
            self.dataset_intra.mean_epoch_left = mean_epoch
            self.dataset_intra.epochs_left = epochs
            print("Left channel cleaned")

        elif self.dataset_intra.selected_channel_index_ecg == 1:
            self.dataset_intra.cleaned_ecg_right = clean_data_full
            self.dataset_intra.detected_peaks_right = final_peaks
            self.dataset_intra.mean_epoch_right = mean_epoch
            self.dataset_intra.epochs_right = epochs
            print("Right channel cleaned")

        # plot the detected peaks
        self.canvas_detected_peaks.setEnabled(True)
        self.toolbar_detected_peaks.setEnabled(True)
        self.ax_detected_peaks.clear()
        self.ax_detected_peaks.set_title('Detected Peaks')
        self.ax_detected_peaks.plot(cropped_data, label='Raw ECG')
        self.ax_detected_peaks.plot(final_peaks, cropped_data[final_peaks], 'ro', label='Detected Peaks')
        self.canvas_detected_peaks.draw()
    
        #plot an overlap of the raw and cleaned data
        self.canvas_ecg_clean.setEnabled(True)
        self.toolbar_ecg_clean.setEnabled(True)
        self.ax_ecg_clean.clear()
        self.ax_ecg_clean.set_title("Cleaned ECG Signal")
        self.ax_ecg_clean.plot(full_data, label='Raw data')
        self.ax_ecg_clean.plot(clean_data_full, label='Cleaned data')
        self.ax_ecg_clean.set_xlabel("Time (s)")
        self.ax_ecg_clean.set_ylabel("Amplitude")
        self.ax_ecg_clean.legend()
        self.canvas_ecg_clean.draw()

        self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning


    def validate_start_end_time(self):
        """Validate the start and end times for cleaning."""
        try:
            start_time = int(self.box_start_time.text())
            end_time = int(self.box_end_time.text())

            if start_time < 0 or end_time < 0:
                raise ValueError("Start and end times must be positive.")

            if start_time >= end_time:
                raise ValueError("Start time must be less than end time.")

            if self.dataset_intra.selected_channel_index_ecg == 0:
                self.dataset_intra.start_time_left = start_time
                self.dataset_intra.end_time_left = end_time
            elif self.dataset_intra.selected_channel_index_ecg == 1:
                self.dataset_intra.start_time_right = start_time
                self.dataset_intra.end_time_right = end_time

            self.dataset_intra.ecg_tresh = int(self.box_thresh_ecg.currentText())
            self.label_start_time.setText(f"Start time: {start_time} s")
            self.label_end_time.setText(f"End time: {end_time} s")
            self.btn_start_ecg_cleaning.setEnabled(True)  # Enable the button after validation

            if self.radio_button_down.isChecked():
                self.dataset_intra.artifact_polarity = "down"
            elif self.radio_button_up.isChecked():
                self.dataset_intra.artifact_polarity = "up"

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input, please enter an integer", str(e))


    def choose_channel_for_cleaning(self):
        """Prompt for channel name selection for intracranial file."""
        if self.dataset_intra.raw_data:
            try:
                channel_names = self.dataset_intra.ch_names  # List of channel names
                channel_name, ok = QInputDialog.getItem(self, "Channel Selection", "Select a channel:", channel_names, 0, False)

                if ok and channel_name:  # Check if a channel was selected
                    self.dataset_intra.selected_channel_name_ecg = channel_name
                    self.dataset_intra.selected_channel_index_ecg = channel_names.index(channel_name)  # Get the index of the selected channel
                    self.label_selected_channel.setText(f"Selected Channel: {channel_name}")                    

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")

        """Plot the selected channel data from the intracranial file."""
        if self.dataset_intra.raw_data and self.dataset_intra.selected_channel_index_ecg is not None:
            self.canvas_ecg.setEnabled(True)
            self.toolbar_ecg.setEnabled(True)
            self.ax_ecg.clear()
            channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
            times = self.dataset_intra.times
            self.ax_ecg.plot(times, channel_data)
            self.ax_ecg.set_title(f"Channel {self.dataset_intra.selected_channel_index_ecg} data - {self.dataset_intra.selected_channel_name_ecg}")
            self.ax_ecg.set_xlabel("Time (s)")
            self.ax_ecg.set_ylabel("Amplitude")
            self.canvas_ecg.draw()
            self.box_start_time.setEnabled(True)
            self.box_end_time.setEnabled(True)
            self.box_thresh_ecg.setEnabled(True)
            self.btn_validate_start_end_time.setEnabled(True)


    def show_first_page(self):
        self.stacked_widget.setCurrentIndex(0)
        self.update_button_styles(self.btn_first)

    def show_second_page(self):
        self.stacked_widget.setCurrentIndex(2)
        self.update_button_styles(self.btn_second)
        if self.dataset_intra.selected_channel_index is not None:
            self.plot_scatter_channel_intra_sf()
            self.button_select_first_intra.setEnabled(True)
            self.button_select_last_intra.setEnabled(True)
        if self.dataset_extra.selected_channel_index is not None:
            self.plot_scatter_channel_extra_sf()
            self.button_select_first_extra.setEnabled(True)
            self.button_select_last_extra.setEnabled(True)        

    def show_third_page(self):
        self.stacked_widget.setCurrentIndex(3)
        self.update_button_styles(self.btn_third)
        self.update_plot_sync_channels_state()

    def show_fourth_page(self):
        self.stacked_widget.setCurrentIndex(1)
        self.update_button_styles(self.btn_fourth)


    def show_help(self):
        # Path to the HTML file stored in the GUI folder
        help_file_folder = join(dirname(__file__), "help")
        help_file_path = join(help_file_folder, 'info.html')
        if exists(help_file_path):
            webbrowser.open(f'file://{help_file_path}')
        else:
            print("Help file not found.")

    def create_panel_intra_sf_correction(self):
        """Create the left panel for intracranial file processing."""
        layout = QVBoxLayout()

        # Set up canvas for matplotlib for intracranial data
        self.figure_intra_sf, self.ax_intra_sf = plt.subplots()
        self.canvas_intra_sf = FigureCanvas(self.figure_intra_sf)
        self.canvas_intra_sf.setEnabled(False)  # Initially hidden

        # Create a navigation toolbar and add it to the layout
        self.toolbar_intra_sf = NavigationToolbar(self.canvas_intra_sf, self)
        self.toolbar_intra_sf.setEnabled(False) 
        layout.addWidget(self.toolbar_intra_sf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_intra_sf)    # Add the canvas to the layout

        selection_layout = QHBoxLayout()
        selection_layout_first = QVBoxLayout()
        selection_layout_last = QVBoxLayout()

        self.button_select_first_intra = Button("Select first artifact in intracranial recording", "lightblue")
        self.button_select_first_intra.clicked.connect(self.select_first_artifact_intra_sf)    
        self.button_select_first_intra.setEnabled(False)
        self.label_time_select_first_intra = QLabel()
        self.label_time_select_first_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_first_intra = QLabel()
        self.label_sample_select_first_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        self.button_select_last_intra = Button("Select last artifact in intracranial recording", "lightblue")
        self.button_select_last_intra.clicked.connect(self.select_last_artifact_intra_sf)
        self.button_select_last_intra.setEnabled(False)
        self.label_time_select_last_intra = QLabel()
        self.label_time_select_last_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_last_intra = QLabel()
        self.label_sample_select_last_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        selection_layout_first.addWidget(self.button_select_first_intra)
        selection_layout_first.addWidget(self.label_time_select_first_intra)
        selection_layout_first.addWidget(self.label_sample_select_first_intra)
        selection_layout.addLayout(selection_layout_first)

        selection_layout_last.addWidget(self.button_select_last_intra)
        selection_layout_last.addWidget(self.label_time_select_last_intra)
        selection_layout_last.addWidget(self.label_sample_select_last_intra)        
        selection_layout.addLayout(selection_layout_last)
        layout.addLayout(selection_layout)

        return layout


    def plot_scatter_channel_intra_sf(self):
        """Plot scatter plot of the selected channel data."""
        
        self.toolbar_intra_sf.setEnabled(True)
        self.canvas_intra_sf.setEnabled(True)
        self.ax_intra_sf.clear()
        data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        timescale = self.dataset_intra.times

        self.ax_intra_sf.scatter(timescale, data, s=8)
        self.canvas_intra_sf.draw()


    def select_first_artifact_intra_sf(self):
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


    def select_last_artifact_intra_sf(self):
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


    def plot_scatter_channel_extra_sf(self):
        """Plot scatter plot of the selected channel data."""
        
        self.toolbar_extra_sf.setEnabled(True)
        self.canvas_extra_sf.setEnabled(True)
        self.ax_extra_sf.clear()
        data = self.detrend_data(self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index])
        timescale = self.dataset_extra.times

        self.ax_extra_sf.scatter(timescale, data, s=8)
        self.canvas_extra_sf.draw()


    def select_first_artifact_extra_sf(self):
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
        
        data = self.detrend_data(self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index])


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


    def select_last_artifact_extra_sf(self):
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

        data = self.detrend_data(self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index])
        
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


    def create_panel_extra_sf_correction(self):
        """Create the left panel for intracranial file processing."""
        layout = QVBoxLayout()

        # Set up canvas for matplotlib for intracranial data
        self.figure_extra_sf, self.ax_extra_sf = plt.subplots()
        self.canvas_extra_sf = FigureCanvas(self.figure_extra_sf)
        self.canvas_extra_sf.setEnabled(False)  # Initially hidden


        # Create a navigation toolbar and add it to the layout
        self.toolbar_extra_sf = NavigationToolbar(self.canvas_extra_sf, self)
        self.toolbar_extra_sf.setEnabled(False) 
        layout.addWidget(self.toolbar_extra_sf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_extra_sf)    # Add the canvas to the layout

        selection_layout = QHBoxLayout()
        selection_layout_first = QVBoxLayout()
        selection_layout_last = QVBoxLayout()

        self.button_select_first_extra = Button("Select first artifact in external recording", "lightgreen")
        self.button_select_first_extra.clicked.connect(self.select_first_artifact_extra_sf)    
        self.button_select_first_extra.setEnabled(False)
        self.label_time_select_first_extra = QLabel()
        self.label_time_select_first_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_first_extra = QLabel()
        self.label_sample_select_first_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        self.button_select_last_extra = Button("Select last artifact in external recording", "lightgreen")
        self.button_select_last_extra.clicked.connect(self.select_last_artifact_extra_sf)
        self.button_select_last_extra.setEnabled(False)
        self.label_time_select_last_extra = QLabel()
        self.label_time_select_last_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_last_extra = QLabel()
        self.label_sample_select_last_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        selection_layout_first.addWidget(self.button_select_first_extra)
        selection_layout_first.addWidget(self.label_time_select_first_extra)
        selection_layout_first.addWidget(self.label_sample_select_first_extra)
        selection_layout.addLayout(selection_layout_first)

        selection_layout_last.addWidget(self.button_select_last_extra)
        selection_layout_last.addWidget(self.label_time_select_last_extra)
        selection_layout_last.addWidget(self.label_sample_select_last_extra)        
        selection_layout.addLayout(selection_layout_last)
        layout.addLayout(selection_layout)


        return layout


    def create_panel_intra(self):
        """Create the left panel for intracranial file processing."""
        layout = QVBoxLayout()

        # File selection button for intracranial
        self.btn_load_file_intra = Button("Load intracranial file (supported format: .mat)", "lightblue")
        self.btn_load_file_intra.clicked.connect(self.load_mat_file)
        layout.addWidget(self.btn_load_file_intra)

        # Create a label to display the selected file name
        self.file_label_intra = QLabel("No file selected")
        self.file_label_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.addWidget(self.file_label_intra)

        # Set up canvas for matplotlib for intracranial data
        self.figure_intra, self.ax_intra = plt.subplots()
        self.canvas_intra = FigureCanvas(self.figure_intra)
        self.canvas_intra.setEnabled(False)  # Initially hidden


        # Create a navigation toolbar and add it to the layout
        self.toolbar_intra = NavigationToolbar(self.canvas_intra, self)
        self.toolbar_intra.setEnabled(False) 
        layout.addWidget(self.toolbar_intra)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_intra)    # Add the canvas to the layout

        # Button layout for intracranial channel selection and plotting
        self.channel_layout_intra = QVBoxLayout()
        self.channel_selection_layout_intra = QHBoxLayout()

        # Channel selection button for intracranial file (Initially hidden)
        self.btn_select_channel_intra = Button("Select Channel", "lightblue")
        self.btn_select_channel_intra.setEnabled(False)  # Initially inactive
        self.btn_select_channel_intra.clicked.connect(self.prompt_channel_name_intra)
        self.channel_selection_layout_intra.addWidget(self.btn_select_channel_intra)

        # Create a label to display the selected channel name
        self.channel_label_intra = QLabel("No channel selected")
        self.channel_label_intra.setEnabled(False) # Initially inactive
        self.channel_selection_layout_intra.addWidget(self.channel_label_intra)  
        self.channel_layout_intra.addLayout(self.channel_selection_layout_intra)      

        # Plot channel button for intracranial files (Initially hidden)
        self.btn_plot_channel_intra = Button("Plot Selected Channel", "lightblue")
        self.btn_plot_channel_intra.setEnabled(False)  # Initially inactive
        self.btn_plot_channel_intra.clicked.connect(self.plot_channel_intra)
        self.channel_layout_intra.addWidget(self.btn_plot_channel_intra)


        self.artifact_layout_intra = QHBoxLayout()
        self.automatic_artifact_layout_intra = QVBoxLayout()
        self.manual_artifact_layout_intra = QVBoxLayout()


        # Plot artifact detection button for intracranial (Initially hidden)
        self.btn_artifact_detect_intra = Button("Automatic detection synchronization artifact", "lightblue")
        self.btn_artifact_detect_intra.setEnabled(False)  # Initially hidden
        self.btn_artifact_detect_intra.clicked.connect(self.detect_artifacts_intra)
        self.automatic_artifact_layout_intra.addWidget(self.btn_artifact_detect_intra)
        self.label_automatic_artifact_time_intra = QLabel("No artifact automatically detected")
        self.label_automatic_artifact_time_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_automatic_artifact_time_intra.setVisible(False)  # Initially hidden
        self.automatic_artifact_layout_intra.addWidget(self.label_automatic_artifact_time_intra)

        self.btn_manual_select_artifact_intra = Button("Manual detection synchronization artifact", "lightblue") 
        self.btn_manual_select_artifact_intra.setEnabled(False)
        self.btn_manual_select_artifact_intra.clicked.connect(self.manual_selection_intra)
        self.manual_artifact_layout_intra.addWidget(self.btn_manual_select_artifact_intra)
        self.label_manual_artifact_time_intra = QLabel("No artifact manually selected")
        self.label_manual_artifact_time_intra.setVisible(False)
        self.label_manual_artifact_time_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.manual_artifact_layout_intra.addWidget(self.label_manual_artifact_time_intra)        
        
        self.artifact_layout_intra.addLayout(self.automatic_artifact_layout_intra)
        self.artifact_layout_intra.addLayout(self.manual_artifact_layout_intra)

        # Add channel layout to main layout for intracranial
        layout.addLayout(self.channel_layout_intra)
        layout.addLayout(self.artifact_layout_intra)


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
                self.dataset_intra.ch_names = raw_data.ch_names  # Assign channel names#
                self.dataset_intra.times = np.linspace(0, raw_data.get_data().shape[1]/self.dataset_intra.sf, raw_data.get_data().shape[1])
                self.file_label_intra.setText(f"Selected File: {basename(file_name)}")
                self.dataset_intra.file_name = basename(file_name)
                self.dataset_intra.file_path = dirname(file_name)

                # Show channel selection and plot buttons for intracranial
                self.btn_select_channel_intra.setEnabled(True)
                self.channel_label_intra.setEnabled(True)
                self.btn_choose_channel.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load .mat file: {e}")


    def prompt_channel_name_intra(self):
        """Prompt for channel name selection for intracranial file."""
        if self.dataset_intra.raw_data:
            try:
                channel_names = self.dataset_intra.ch_names  # List of channel names
                channel_name, ok = QInputDialog.getItem(self, "Channel Selection", "Select a channel:", channel_names, 0, False)

                if ok and channel_name:  # Check if a channel was selected
                    self.dataset_intra.selected_channel_name = channel_name
                    self.dataset_intra.selected_channel_index = channel_names.index(channel_name)  # Get the index of the selected channel
                    self.channel_label_intra.setText(f"Selected Channel: {channel_name}")
                    self.dataset_intra.max_y_value = np.nanmax(self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index])
                    # Enable the plot button since a channel has been selected
                    self.btn_plot_channel_intra.setEnabled(True)
                    self.btn_artifact_detect_intra.setEnabled(True)    
                    self.label_automatic_artifact_time_intra.setVisible(True)   
                    self.btn_manual_select_artifact_intra.setEnabled(True)
                    self.label_manual_artifact_time_intra.setVisible(True)
                    

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")


    def plot_channel_intra(self):
        """Plot the selected channel data from the intracranial file."""
        if self.dataset_intra.raw_data and self.dataset_intra.selected_channel_index is not None:
            self.canvas_intra.setEnabled(True)
            self.toolbar_intra.setEnabled(True)
            self.ax_intra.clear()
            channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
            times = self.dataset_intra.times
            self.ax_intra.plot(times, channel_data)
            self.ax_intra.set_title(f"Channel {self.dataset_intra.selected_channel_index} data - {self.dataset_intra.selected_channel_name}")
            self.ax_intra.set_xlabel("Time (s)")
            self.ax_intra.set_ylabel("Amplitude")
            self.canvas_intra.draw()


    def detect_artifacts_intra(self):
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
        self.plot_scatter_channel_intra(art_start_intra = self.dataset_intra.art_start)
        self.update_synchronize_button_state()  # Check if we can enable the button
        self.label_automatic_artifact_time_intra.setText(f"Artifact start: {self.dataset_intra.art_start} s")


    def plot_scatter_channel_intra(self, art_start_intra=None):
        """Plot scatter plot of the selected channel data."""
        
        self.toolbar_intra.setEnabled(True)
        self.canvas_intra.setEnabled(True)
        self.ax_intra.clear()
        
        # Plot the channel data
        channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index]
        times = self.dataset_intra.raw_data.times  # Time vector corresponding to the data points
        
        # Plot scatter points
        start = int(round(art_start_intra * self.dataset_intra.sf)-round(self.dataset_intra.sf/10))
        end = int(round(art_start_intra * self.dataset_intra.sf)+round(self.dataset_intra.sf/10))
        times_array = np.array(times)
        channel_data_array = np.array(channel_data)
        self.ax_intra.scatter(times_array[start:end], channel_data_array[start:end], s=5)


        # Highlight artifact start points if available
        if art_start_intra is not None:
                self.ax_intra.axvline(x=art_start_intra, color='red', linestyle='--', label='Artifact Start')

        self.ax_intra.legend()
        
        # Allow interactive features like zoom and pan
        self.canvas_intra.draw()



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
                    self.label_manual_artifact_time_intra.setText(f"Selected Artifact start: {closest_value_x} s")
                    self.update_synchronize_button_state()

        self.canvas_intra.mpl_connect("button_press_event", onclick)
        
    





    def create_xdf_panel(self):
        """Create the right panel for .xdf file processing."""
        layout = QVBoxLayout()

        # File selection button for .xdf
        self.btn_load_file_xdf = Button("Load external file (supported formats: .Poly5, .xdf)", "lightgreen")
        self.btn_load_file_xdf.clicked.connect(self.load_ext_file)
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


    def load_ext_file(self):
        """Load external file. Supported file formats are .xdf, .poly5"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select External File", "", "All Files (*);;XDF Files (*.xdf);;Poly5 Files (*.Poly5)")
        self.file_label_xdf.setText(f"Selected File: {basename(file_name)}")
        self.dataset_extra.file_name = basename(file_name)
        self.dataset_extra.file_path = dirname(file_name)
        
        if file_name.endswith(".xdf"):
            self.load_xdf_file(file_name)

        elif file_name.endswith(".Poly5"):
                self.load_poly5_file(file_name)


    def load_poly5_file(self, file_name):
        """Load .poly5 file."""
        try:
            TMSi_data = Poly5Reader(file_name)
            toMNE = True
            TMSi_rec = TMSi_data.read_data_MNE()
            self.dataset_extra.raw_data = TMSi_rec
            self.dataset_extra.sf = TMSi_rec.info["sfreq"]  # Get the sampling frequency
            self.dataset_extra.ch_names = TMSi_rec.ch_names  # Get the channel names
            self.dataset_extra.times = TMSi_rec.times # Get the timescale

            # Show channel selection and plot buttons for .xdf
            self.channel_label_xdf.setEnabled(True)
            self.btn_select_channel_xdf.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .poly5 file: {e}")


    def load_xdf_file(self, file_name):
        """Load .xdf file."""
        try:
            # Load the .xdf file using the read_raw function
            stream_id = self.find_EEG_stream(file_name, stream_name='SAGA')
            raw_data = read_raw(file_name, stream_ids=[stream_id], preload=True)
            self.dataset_extra.raw_data = raw_data
            self.dataset_extra.sf = raw_data.info["sfreq"]  # Get the sampling frequency
            self.dataset_extra.ch_names = raw_data.ch_names  # Get the channel names
            self.dataset_extra.times = raw_data.times # Get the timescale

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
                    channel_data = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
                    detrended_data = self.detrend_data(channel_data)
                    self.dataset_extra.min_y_value = detrended_data.min()
                    self.dataset_extra.max_y_value = detrended_data.max()
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
        channel_data_to_plot = self.detrend_data(data)
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
                    self.update_synchronize_button_state()

        self.canvas_xdf.mpl_connect("button_press_event", onclick)
        
    


    def synchronize_datasets_as_set(self):
        print("events from annotations extraction")
        events, _ = mne.events_from_annotations(self.dataset_extra.raw_data)
        inv_dic = {v: str(k) for k, v in _.items()}

        ## offset intracranial recording (crop everything that is more than 1s before the artifact)
        #tmax_lfp = self.dataset_intra.times[-2]
        tmax_lfp = max(self.dataset_intra.raw_data.times)
        new_start_intracranial = self.dataset_intra.art_start - 1
        lfp_rec_offset = self.dataset_intra.raw_data.copy().crop(tmin=new_start_intracranial, tmax=tmax_lfp)
        print(f"tmax_lfp for cropping is: {tmax_lfp}")

        ## offset external recording (crop everything that is more than 1s before the artifact)
        tmax_external = max(self.dataset_extra.times)
        new_start_external = self.dataset_extra.art_start - 1
        TMSi_rec_offset = self.dataset_extra.raw_data.copy().crop(tmin=new_start_external, tmax=tmax_external)

        ## transfer of the events from the external to the intracranial recording
        # create a duplicate of the events to manipulate it without changing the external one
        events_lfp = deepcopy(events)

        # get the events from the external in time instead of samples to account for the different sampling frequencies
        events_in_time = events[:,0]/self.dataset_extra.sf

        # then offset the events in time to the new start of the external recording
        events_in_time_offset = events_in_time - new_start_external

        # convert the events in time offset to samples corresponding to the sampling frequency of the intracranial recording
        # because the annotations object works with samples, not timings
        events_in_time_offset_lfp = events_in_time_offset * self.dataset_intra.sf
        events_lfp[:,0] = events_in_time_offset_lfp

        # Recast event descriptions to standard Python strings
        #events_lfp[:, 2] = events_lfp[:, 2].astype(str)

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

        if self.dataset_intra.eff_sf is not None:
            lfp_sf = self.dataset_intra.eff_sf
        else:
            lfp_sf = self.dataset_intra.sf
        
        lfp_timescale = np.linspace(0, self.dataset_intra.raw_data.get_data().shape[1]/lfp_sf, self.dataset_intra.raw_data.get_data().shape[1])

        write_set(
            fname_external_out, 
            TMSi_rec_offset, 
            TMSi_rec_offset_annotations_onset,
            TMSi_rec_offset.info['sfreq'],
            TMSi_rec_offset.times
            )
        write_set(
            fname_lfp_out, 
            lfp_rec_offset, 
            lfp_rec_offset_annotations_onset,
            lfp_sf,
            lfp_timescale
            )

        QMessageBox.information(self, "Synchronization", "Synchronization done. Both files saved as .SET")

    

    def synchronize_datasets_as_pickles(self): 
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



    def synchronize_datasets_as_mat(self):
        index_start_LFP = (self.dataset_intra.art_start - 1) * self.dataset_intra.sf
        LFP_array = self.dataset_intra.raw_data.get_data()
        LFP_cropped = LFP_array[:, int(index_start_LFP) :].T

        ## External ##
        # Crop beginning of external recordings 1s before first artifact:
        time_start_external = (self.dataset_extra.art_start) - 1
        index_start_external = time_start_external * self.dataset_extra.sf
        external_file = self.dataset_extra.raw_data.get_data()
        external_cropped = external_file[:, int(index_start_external) :].T

        # Check which recording is the longest,
        # crop it to give it the same duration as the other one:
        LFP_rec_duration = len(LFP_cropped) / self.dataset_intra.sf
        external_rec_duration = len(external_cropped) / self.dataset_extra.sf

        if LFP_rec_duration > external_rec_duration:
            index_stop_LFP = external_rec_duration * self.dataset_intra.sf
            LFP_synchronized = LFP_cropped[: int(index_stop_LFP), :]
            external_synchronized = external_cropped
        elif external_rec_duration > LFP_rec_duration:
            index_stop_external = LFP_rec_duration * self.dataset_extra.sf
            external_synchronized = external_cropped[: int(index_stop_external), :]
            LFP_synchronized = LFP_cropped
        else:
            LFP_synchronized = LFP_cropped
            external_synchronized = external_cropped  

        # save the synchronized data in mat format          
        LFP_df_offset = pd.DataFrame(LFP_synchronized)
        LFP_df_offset.columns = self.dataset_intra.ch_names
        external_df_offset = pd.DataFrame(external_synchronized)
        external_df_offset.columns = self.dataset_extra.ch_names

        lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".mat")
        external_title = (f"SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-6]) + ".mat")
        
        if self.folder_path is not None:
            LFP_filename = join(self.folder_path, lfp_title)
            external_filename = join(self.folder_path, external_title)
        else:
            LFP_filename = lfp_title
            external_filename = external_title

        savemat(
            LFP_filename,
            {
                "data": LFP_df_offset.T,
                "fsample": self.dataset_intra.sf,
                "label": np.array(
                    LFP_df_offset.columns.tolist(), dtype=object
                ).reshape(-1, 1),
            },
        )
        savemat(
            external_filename,
            {
                "data": external_df_offset.T,
                "fsample": self.dataset_extra.sf,
                "label": np.array(
                    external_df_offset.columns.tolist(), dtype=object
                ).reshape(-1, 1),
            },
        )
        QMessageBox.information(self, "Synchronization", "Synchronization done. Both files saved as .mat files")       


    def plot_synced_channels(self):
        self.toolbar_synced.setEnabled(True)
        self.canvas_synced.setEnabled(True)
        self.ax_synced.clear()

        # scale y-axis to the same range for both channels by modifying the ylim for the external channel:
        y_max_factor = self.dataset_intra.max_y_value / self.dataset_extra.max_y_value

        # Plot the external channel synchronized
        data_extra = self.dataset_extra.raw_data.get_data()[self.dataset_extra.selected_channel_index]
        data_extra_scaled = data_extra * y_max_factor

        data_extra_detrended = self.detrend_data(data_extra_scaled)
        timescale_extra = self.dataset_extra.times
        art_start_0_extra = self.dataset_extra.art_start - 1

        # Find the index in self.dataset_extra.times corresponding to art_start_0
        art_start_index_extra = (timescale_extra >= art_start_0_extra).argmax()
        offset_data_extra = data_extra_detrended[art_start_index_extra:]
        offset_timescale_extra = timescale_extra[art_start_index_extra:] - art_start_0_extra
        self.dataset_extra.reset_timescale = offset_timescale_extra
        self.dataset_extra.reset_data = offset_data_extra
        self.ax_synced.scatter(offset_timescale_extra, offset_data_extra, s=8, color='#90EE90', label='External')

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
        self.ax_synced.scatter(offset_timescale_intra, offset_data_intra, s=8, color='#6495ED', label='Intracranial')
        self.ax_synced.legend(loc='upper left')
        self.canvas_synced.draw()
        self.btn_select_last_art_intra.setEnabled(True)
        self.btn_select_last_art_xdf.setEnabled(True)



    def select_last_artifact_intra(self):
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
                    self.label_select_last_art_intra.setText(f"Selected last artifact start: {self.dataset_intra.last_artifact} s")
                    self.update_timeshift_button_state()

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
                    self.update_timeshift_button_state()
        # Connect and store the ID of the external event listener
        self.cid_extra = self.canvas_synced.mpl_connect("button_press_event", onclick)
        


    def compute_timeshift(self):
        timeshift = (self.dataset_extra.last_artifact - self.dataset_intra.last_artifact)*1000
        self.label_timeshift.setText(f"Timeshift: {timeshift} ms")


    def compute_eff_sf(self):
        time_interval = self.dataset_extra.last_art_start_time - self.dataset_extra.first_art_start_time
        print(f"time interval: {time_interval}")
        sample_interval = self.dataset_intra.last_art_start_idx - self.dataset_intra.first_art_start_idx
        print(f"sample interval: {sample_interval}")
        self.dataset_intra.eff_sf = sample_interval/time_interval
        self.dataset_intra.sf = self.dataset_intra.eff_sf
        self.dataset_intra.times = np.linspace(0, self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.raw_data.get_data().shape[1])
        print(f"eff_sf: {self.dataset_intra.eff_sf}")
        self.label_eff_sf.setText(f"The effective sampling frequency of the intracranial recording is actually {self.dataset_intra.eff_sf} and should be used for synchronization.")


    def update_synchronize_button_state(self):
        """Enable or disable the synchronize button based on file selection."""
        if self.dataset_intra.art_start is not None and self.dataset_extra.art_start is not None:
            if self.dataset_extra.file_name.endswith(".xdf"):
                self.btn_sync_as_set.setEnabled(True)
                self.btn_sync_as_pickle.setEnabled(True)
                self.btn_all_as_pickle.setEnabled(True)
            elif self.dataset_extra.file_name.endswith(".Poly5"):
                self.btn_sync_as_mat.setEnabled(True)
        else:
            self.btn_sync_as_set.setEnabled(False)
            self.btn_sync_as_pickle.setEnabled(False)
            self.btn_all_as_pickle.setEnabled(False)
            self.btn_sync_as_mat.setEnabled(False)

    
    def update_compute_eff_sf_button_state(self):
        if (
            self.dataset_intra.first_art_start_idx is not None
            and self.dataset_intra.last_art_start_idx is not None
            and self.dataset_extra.first_art_start_time is not None
            and self.dataset_extra.last_art_start_time is not None
        ):
            self.btn_compute_eff_sf.setEnabled(True)
        else:
            self.btn_compute_eff_sf.setEnabled(False)


    def update_timeshift_button_state(self):
        """Enable or disable the timeshift button based on artifact selection."""
        if self.dataset_intra.last_artifact is not None and self.dataset_extra.last_artifact is not None:
            self.btn_compute_timeshift.setEnabled(True)
        else:
            self.btn_compute_timeshift.setEnabled(False)

    def update_plot_sync_channels_state(self):
        if self.dataset_intra.art_start is not None and self.dataset_extra.art_start is not None:
            self.btn_plot_synced_channels.setEnabled(True)
        else:
            self.btn_plot_synced_channels.setEnabled(False)


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



    def update_button_styles(self, active_button):
        # Set custom property to determine which button is active
        buttons = [self.btn_first, self.btn_second, self.btn_third, self.btn_fourth]
        for button in buttons:
            button.setProperty("active", button == active_button)
            button.style().unpolish(button)
            button.style().polish(button)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyncGUI()
    window.show()
    sys.exit(app.exec_())
